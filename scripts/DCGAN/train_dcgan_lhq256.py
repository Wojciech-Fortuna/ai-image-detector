import os
import sys
import io
import re
import random
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils
from PIL import Image
from tqdm import tqdm


IMG_SIZE = 256
NZ = 100
NGF = 64
NDF = 64
BETA1 = 0.5
LR_G = 2e-4
LR_D = 1e-4
JPEG_QUALITY = 95
SEED = 1234


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rclone_available() -> bool:
    try:
        subprocess.run(["rclone", "version"], capture_output=True, check=False)
        return True
    except Exception:
        return False

def rclone_mkdir(remote_path: str):
    try:
        subprocess.run(["rclone", "mkdir", remote_path], check=False)
    except Exception as e:
        print(f"[warn] rclone mkdir {remote_path} failed: {e}")

def rclone_ls(remote_path: str) -> List[str]:
    p = subprocess.run(
        ["rclone", "lsf", "--format", "p", "--files-only", remote_path],
        capture_output=True, text=True, check=False
    )
    if p.returncode != 0:
        raise RuntimeError(f"rclone lsf failed for {remote_path}: {p.stderr}")
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]

def rclone_cat(remote_file: str) -> bytes:
    p = subprocess.run(["rclone", "cat", remote_file], capture_output=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"rclone cat failed for {remote_file}: {p.stderr.decode('utf-8', 'ignore')}")
    return p.stdout

def find_last_epoch(remote_base: str) -> Optional[int]:
    files = rclone_ls(remote_base)
    ep_pat = re.compile(r"^(G|D|G_EMA|state)_epoch_(\d{3})\.(pt|pth)$")
    by_epoch = {}
    for name in files:
        m = ep_pat.match(name)
        if not m:
            continue
        kind, ep = m.group(1), int(m.group(2))
        by_epoch.setdefault(ep, set()).add(kind)
    candidates = [ep for ep, kinds in by_epoch.items() if {"G", "D"}.issubset(kinds) or "state" in kinds]
    return max(candidates) if candidates else None

def load_state_dict_from_remote(remote_file: str, map_location="cpu"):
    data = rclone_cat(remote_file)
    buff = io.BytesIO(data)
    return torch.load(buff, map_location=map_location)

class RCloneStreamWriter:
    def __init__(self, remote_path: str):
        self.remote_path = remote_path
        self.proc = None
        self.stdin = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["rclone", "rcat", self.remote_path, "--retries", "3", "--low-level-retries", "10"],
            stdin=subprocess.PIPE
        )
        if self.proc.stdin is None:
            raise RuntimeError("Failed to open stdin for `rclone rcat`")
        self.stdin = self.proc.stdin
        return self.stdin

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.stdin:
                self.stdin.close()
        finally:
            ret = self.proc.wait() if self.proc else 0
            if ret != 0:
                raise RuntimeError(f"`rclone rcat` exited with code {ret} for {self.remote_path}")

def save_tensor_png_to_stream(tensor_0_1: torch.Tensor, stream, nrow=8):
    grid = vutils.make_grid(tensor_0_1, nrow=nrow)
    to_pil = transforms.ToPILImage()
    img = to_pil(grid)
    img.save(stream, format="PNG", quality=JPEG_QUALITY)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class FlatImageDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.files: List[Path] = []
        for p in self.root.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                self.files.append(p)
        if not self.files:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        with Image.open(fp) as im:
            im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, 0

def make_loader_flat(
    root: Path,
    img_size=256,
    batch_size=64,
    workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
):
    is_windows = (os.name == "nt")

    if workers is None:
        workers = 0 if is_windows else 4
    if pin_memory is None:
        pin_memory = (torch.cuda.is_available() and not is_windows)
    if prefetch_factor is None:
        prefetch_factor = 1 if is_windows else 2
    if persistent_workers is None:
        persistent_workers = (workers > 0)

    tfm = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = FlatImageDataset(root, transform=tfm)

    loader_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs), len(ds)


def sn(m): return nn.utils.spectral_norm(m)

class Generator256(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz,    ngf*16, 4, 1, 0, bias=False),  # 1 -> 4
            nn.BatchNorm2d(ngf*16), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),  # 4 -> 8
            nn.BatchNorm2d(ngf*8),  nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8,  ngf*4, 4, 2, 1, bias=False),  # 8 -> 16
            nn.BatchNorm2d(ngf*4),  nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4,  ngf*2, 4, 2, 1, bias=False),  # 16 -> 32
            nn.BatchNorm2d(ngf*2),  nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2,    ngf, 4, 2, 1, bias=False),  # 32 -> 64
            nn.BatchNorm2d(ngf),    nn.ReLU(True),

            nn.ConvTranspose2d(ngf,    ngf//2, 4, 2, 1, bias=False), # 64 -> 128
            nn.BatchNorm2d(ngf//2), nn.ReLU(True),

            nn.ConvTranspose2d(ngf//2,   3,   4, 2, 1, bias=False),  # 128 -> 256
            nn.Tanh(),
        )

    def forward(self, z): return self.main(z)

class Discriminator256(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            sn(nn.Conv2d(nc,     ndf//2, 4, 2, 1, bias=False)), nn.LeakyReLU(0.2, True),  # 256 -> 128
            sn(nn.Conv2d(ndf//2, ndf,    4, 2, 1, bias=False)), nn.LeakyReLU(0.2, True),  # 128 -> 64
            sn(nn.Conv2d(ndf,    ndf*2,  4, 2, 1, bias=False)), nn.LeakyReLU(0.2, True),  # 64 -> 32
            sn(nn.Conv2d(ndf*2,  ndf*4,  4, 2, 1, bias=False)), nn.LeakyReLU(0.2, True),  # 32 -> 16
            sn(nn.Conv2d(ndf*4,  ndf*8,  4, 2, 1, bias=False)), nn.LeakyReLU(0.2, True),  # 16 -> 8
            sn(nn.Conv2d(ndf*8,  ndf*16, 4, 2, 1, bias=False)), nn.LeakyReLU(0.2, True),  # 8  -> 4
            sn(nn.Conv2d(ndf*16, 1,      4, 1, 0, bias=False)),                           # 4  -> 1
        )

    def forward(self, x): return self.main(x).view(x.size(0), 1)

def init_weights(m):
    name = m.__class__.__name__
    if "Conv" in name:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif "BatchNorm" in name:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


class HingeLossD(nn.Module):
    def forward(self, out_real, out_fake):
        return torch.mean(nn.ReLU()(1.0 - out_real)) + torch.mean(nn.ReLU()(1.0 + out_fake))

class HingeLossG(nn.Module):
    def forward(self, out_fake):
        return -torch.mean(out_fake)

class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.model = model
        self.ema = type(model)().to(next(model.parameters()).device)
        self.ema.load_state_dict(model.state_dict(), strict=True)
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self):
        msd = self.model.state_dict()
        esd = self.ema.state_dict()
        for k in msd.keys():
            if msd[k].dtype.is_floating_point:
                esd[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)
            else:
                esd[k] = msd[k]


def train_lhq256_only(
    data_dir: Path,
    ckpt_remote: str,
    epochs=25,
    batch_size=64,
    nz=NZ,
    ngf=NGF,
    ndf=NDF,
    seed=SEED,
    workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    resume: bool = False,
):
    if not rclone_available():
        raise RuntimeError("rclone is required for remote saving (MEGA). Please install and configure it.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    loader, nimgs = make_loader_flat(
        data_dir, img_size=IMG_SIZE, batch_size=batch_size,
        workers=workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor
    )
    print(f"[nature/LHQ] Training start | images: {nimgs} | img_size={IMG_SIZE} | device={device}")

    G, D = Generator256(nz=nz, ngf=ngf).to(device), Discriminator256(ndf=ndf).to(device)
    G.apply(init_weights); D.apply(init_weights)

    optG = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, 0.999))
    optD = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    lossD_fn, lossG_fn = HingeLossD(), HingeLossG()

    ema = EMA(G, decay=0.999)

    fixed_z = torch.randn(64, nz, 1, 1, device=device)

    remote_base = ckpt_remote.rstrip("/")
    rclone_mkdir(remote_base)
    print(f"[nature/LHQ] Streaming saves to remote: {remote_base}")

    start_epoch = 1
    if resume:
        last_epoch = find_last_epoch(remote_base)
        if last_epoch is not None:
            print(f"[nature/LHQ] Resuming from remote epoch {last_epoch}")
            state_remote = f"{remote_base}/state_epoch_{last_epoch:03d}.pth"
            try:
                ckpt = load_state_dict_from_remote(state_remote, map_location=device)
                G.load_state_dict(ckpt["G"])
                D.load_state_dict(ckpt["D"])
                ema.ema.load_state_dict(ckpt["G_EMA"])
                optG.load_state_dict(ckpt["optG"])
                optD.load_state_dict(ckpt["optD"])
                if "fixed_z" in ckpt and isinstance(ckpt["fixed_z"], torch.Tensor):
                    fixed_z = ckpt["fixed_z"].to(device)
                start_epoch = int(ckpt.get("epoch", last_epoch)) + 1
                print("[nature/LHQ] Loaded full optimizer state.")
            except Exception as e:
                print(f"[warn] Could not load full state ({e}). Falling back to weights only.")
                G.load_state_dict(load_state_dict_from_remote(f"{remote_base}/G_epoch_{last_epoch:03d}.pt", map_location=device))
                D.load_state_dict(load_state_dict_from_remote(f"{remote_base}/D_epoch_{last_epoch:03d}.pt", map_location=device))
                try:
                    ema_state = load_state_dict_from_remote(f"{remote_base}/G_EMA_epoch_{last_epoch:03d}.pt", map_location=device)
                    ema.ema.load_state_dict(ema_state, strict=True)
                except Exception as ee:
                    print(f"[warn] EMA for epoch {last_epoch} not found or failed to load ({ee}); copying from G.")
                    ema.ema.load_state_dict(G.state_dict(), strict=True)
                start_epoch = last_epoch + 1
        else:
            print("[nature/LHQ] No previous checkpoints found on remote; starting from scratch.")

    for epoch in range(start_epoch, epochs + 1):
        G.train(); D.train()
        pbar = tqdm(loader, desc=f"[nature/LHQ] epoch {epoch}/{epochs}")
        for real, _ in pbar:
            real = real.to(device)
            b = real.size(0)

            z = torch.randn(b, nz, 1, 1, device=device)
            with torch.no_grad():
                fake = G(z)
            out_real, out_fake = D(real), D(fake)
            lossD = lossD_fn(out_real, out_fake)
            optD.zero_grad(set_to_none=True); lossD.backward(); optD.step()

            z = torch.randn(b, nz, 1, 1, device=device)
            fake = G(z)
            out_fake = D(fake)
            lossG = lossG_fn(out_fake)
            optG.zero_grad(set_to_none=True); lossG.backward(); optG.step()

            ema.update()

            pbar.set_postfix(lossD=float(lossD.item()), lossG=float(lossG.item()))

        G.eval(); ema.ema.eval()
        with torch.no_grad():
            samples = (ema.ema(fixed_z).cpu().clamp(-1, 1) + 1) * 0.5  # [0,1]
        png_name = f"samples_epoch_{epoch:03d}.png"
        with RCloneStreamWriter(f"{remote_base}/{png_name}") as f:
            save_tensor_png_to_stream(samples.clamp(0, 1), f, nrow=8)

        with RCloneStreamWriter(f"{remote_base}/G_epoch_{epoch:03d}.pt") as f:
            torch.save(G.state_dict(), f, _use_new_zipfile_serialization=True)
        with RCloneStreamWriter(f"{remote_base}/G_EMA_epoch_{epoch:03d}.pt") as f:
            torch.save(ema.ema.state_dict(), f, _use_new_zipfile_serialization=True)
        with RCloneStreamWriter(f"{remote_base}/D_epoch_{epoch:03d}.pt") as f:
            torch.save(D.state_dict(), f, _use_new_zipfile_serialization=True)

        with RCloneStreamWriter(f"{remote_base}/state_epoch_{epoch:03d}.pth") as f:
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "G_EMA": ema.ema.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "fixed_z": fixed_z.detach().cpu(),
                "rng": {
                    "python": random.getstate(),
                    "torch": torch.random.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                }
            }, f, _use_new_zipfile_serialization=True)

        print(f"[nature/LHQ] Saved epoch {epoch} -> remote {remote_base}")


def main():
    import argparse
    ap = argparse.ArgumentParser("Train DCGAN @256×256 on LHQ nature (save to MEGA via rclone)")
    ap.add_argument("--data_dir", type=str, required=True, help="folder z obrazami LHQ (np. lhq_256)")
    ap.add_argument("--ckpt_remote", type=str, required=True, help="np. mega:dcgan_runs/nature_lhq256")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--nz", type=int, default=NZ)
    ap.add_argument("--ngf", type=int, default=NGF)
    ap.add_argument("--ndf", type=int, default=NDF)
    ap.add_argument("--seed", type=int, default=SEED)

    ap.add_argument("--workers", type=int, default=None, help="DataLoader workers (auto-tuned for Windows)")
    ap.add_argument("--prefetch_factor", type=int, default=None, help="DataLoader prefetch_factor (gdy workers>0)")
    pin_group = ap.add_mutually_exclusive_group()
    pin_group.add_argument("--pin_memory", dest="pin_memory_flag", action="store_true", help="Force pin_memory=True")
    pin_group.add_argument("--no_pin_memory", dest="no_pin_memory_flag", action="store_true", help="Force pin_memory=False")

    ap.add_argument("--resume", action="store_true", help="Resume from latest epoch found on remote")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}", file=sys.stderr); sys.exit(1)

    if not rclone_available():
        print("rclone is not available. Please install and configure it (rclone config).", file=sys.stderr)
        sys.exit(1)

    pin_mem: Optional[bool]
    if args.pin_memory_flag:
        pin_mem = True
    elif args.no_pin_memory_flag:
        pin_mem = False
    else:
        pin_mem = None

    train_lhq256_only(
        data_dir=data_dir,
        ckpt_remote=args.ckpt_remote,
        epochs=args.epochs,
        batch_size=args.batch_size,
        nz=args.nz,
        ngf=args.ngf,
        ndf=args.ndf,
        seed=args.seed,
        workers=args.workers,
        pin_memory=pin_mem,
        prefetch_factor=args.prefetch_factor,
        resume=args.resume,
    )

if __name__ == "__main__":
    main()
