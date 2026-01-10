import os
import io
import argparse
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils

def sn(m):
    return nn.utils.spectral_norm(m)

class Generator256(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz,    ngf*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*16), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),  nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8,  ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),  nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4,  ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),  nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2,    ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),    nn.ReLU(True),

            nn.ConvTranspose2d(ngf,    ngf//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//2), nn.ReLU(True),

            nn.ConvTranspose2d(ngf//2,   3,   4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)

def rclone_available() -> bool:
    try:
        subprocess.run(["rclone", "version"], capture_output=True, check=False)
        return True
    except Exception:
        return False

def rclone_cat(remote_file: str) -> bytes:
    p = subprocess.run(["rclone", "cat", remote_file], capture_output=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"rclone cat failed for {remote_file}: {p.stderr.decode('utf-8','ignore')}")
    return p.stdout

def load_state_dict_from_remote(remote_file: str, map_location="cpu"):
    data = rclone_cat(remote_file)
    buff = io.BytesIO(data)
    return torch.load(buff, map_location=map_location)

def is_too_flat(
    img_01: torch.Tensor,
    std_thresh: float = 0.06,
    range_thresh: float = 0.20,
    edge_mean_thresh: float = 0.04,
) -> bool:
    x = img_01.detach().cpu()
    std = x.std().item()
    rng = (x.max() - x.min()).item()

    r, g, b = x[0], x[1], x[2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = gray.unsqueeze(0).unsqueeze(0)

    sobel_x = torch.tensor(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        dtype=gray.dtype, device=gray.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[1,  2,  1],
         [0,  0,  0],
         [-1, -2, -1]],
        dtype=gray.dtype, device=gray.device
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    edge_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    edge_mean = edge_mag.mean().item()

    if std < std_thresh or rng < range_thresh or edge_mean < edge_mean_thresh:
        return True
    return False


@torch.no_grad()
def generate_images(
    ckpt_remote: str,
    epoch: int = 182,
    out_dir: Path = Path("samples_epoch_182"),
    total: int = 1000,
    batch_size: int = 64,
    nz: int = 100,
    ngf: int = 64,
    seed: int = 1234,
):
    if not rclone_available():
        raise RuntimeError(
            "rclone is required to download remote weights (from MEGA). "
            "Install and configure `rclone`."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    remote_base = ckpt_remote.rstrip("/")
    ema_path = f"{remote_base}/G_EMA_epoch_{epoch:03d}.pt"
    g_path   = f"{remote_base}/G_epoch_{epoch:03d}.pt"

    G = Generator256(nz=nz, ngf=ngf).to(device).eval()
    try:
        sd = load_state_dict_from_remote(ema_path, map_location=device)
        G.load_state_dict(sd, strict=True)
        print(f"[ok] Loaded G_EMA from epoch {epoch} ({ema_path})")
    except Exception as e:
        print(f"[warn] Failed to load G_EMA ({e}). Trying G...")
        sd = load_state_dict_from_remote(g_path, map_location=device)
        G.load_state_dict(sd, strict=True)
        print(f"[ok] Loaded G from epoch {epoch} ({g_path})")

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    img_idx = 1
    attempts = 0
    max_attempts = total * 50

    while saved < total and attempts < max_attempts:
        remaining = total - saved
        b = min(batch_size, remaining * 2)
        attempts += b

        z = torch.randn(b, nz, 1, 1, device=device)
        fake = G(z).clamp(-1, 1)
        fake_01 = (fake + 1) * 0.5

        for i in range(b):
            if saved >= total:
                break

            img = fake_01[i]

            if is_too_flat(img):
                continue

            fn = out_dir / f"img_{img_idx:05d}.png"
            vutils.save_image(img, fn)
            img_idx += 1
            saved += 1

    if saved < total:
        print(
            f"[warn] Saved only {saved} images (target: {total}). "
            "The model likely generates too many near-uniform images."
        )
    else:
        print(f"[done] Saved {saved} images to: {out_dir.resolve()}")

    
def main():
    ap = argparse.ArgumentParser(
        "Sample images from DCGAN@256 (MEGA/rclone) while rejecting 'gray' / empty images (std+range+Sobel)"
    )
    ap.add_argument("--ckpt_remote", type=str, required=True, help="e.g. mega:dcgan_runs/nature_lhq256")
    ap.add_argument("--epoch", type=int, default=182)
    ap.add_argument("--out_dir", type=Path, default=Path("samples_epoch_182"))
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--nz", type=int, default=100)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    generate_images(
        ckpt_remote=args.ckpt_remote,
        epoch=args.epoch,
        out_dir=args.out_dir,
        total=args.total,
        batch_size=args.batch_size,
        nz=args.nz,
        ngf=args.ngf,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
