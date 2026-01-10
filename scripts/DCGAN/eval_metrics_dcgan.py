import os
import io
import csv
import random
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import gc

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 256
NZ = 100
NGF = 64
SEED = 1234

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rclone_cat(remote_file: str) -> bytes:
    p = subprocess.run(["rclone", "cat", remote_file], capture_output=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"rclone cat failed for {remote_file}: {p.stderr.decode('utf-8','ignore')}")
    return p.stdout

def load_state_dict_from_remote(remote_file: str, map_location="cpu"):
    data = rclone_cat(remote_file)
    buff = io.BytesIO(data)
    return torch.load(buff, map_location=map_location, weights_only=False)

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

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

def list_png_files(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*.png")]
    if not files:
        raise RuntimeError(f"No PNG images found under {root}")
    return files

def load_image_rgb(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None

class FidRealState:
    def __init__(self, mode: str, payload: Dict[str, Any], n: int):
        self.mode = mode
        self.payload = payload
        self.n = n

def _get_attr(obj, names: List[str]):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None

@torch.inference_mode()
def prepare_real_feature_buffers(
    data_dir: Path,
    device: str,
    resize_to_inception: bool,
    real_max: Optional[int],
    real_batch: int,
) -> Tuple[int, FidRealState, Optional[torch.Tensor]]:
    fid_tmp = FrechetInceptionDistance(feature=2048).to(device)
    kid_tmp = KernelInceptionDistance(subset_size=1000).to(device)

    tfm_inception = transforms.Compose([
        transforms.Resize((299, 299)) if resize_to_inception else transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.PILToTensor(),
    ])

    files = list_png_files(data_dir)
    if real_max is not None:
        files = files[:real_max]

    real_cnt = 0
    print(">> Preparing real features (FID/KID) ...")
    for i in tqdm(range(0, len(files), real_batch), desc="Real images"):
        batch_paths = files[i:i + real_batch]
        batch = []
        for p in batch_paths:
            im = load_image_rgb(p)
            if im is None:
                continue
            batch.append(tfm_inception(im))
        if not batch:
            continue
        xb = torch.stack(batch, dim=0).to(device)
        fid_tmp.update(xb, real=True)
        kid_tmp.update(xb, real=True)
        real_cnt += xb.size(0)

    real_feats_old = _get_attr(fid_tmp, ["_real_features"])
    if isinstance(real_feats_old, torch.Tensor) and real_feats_old.numel() > 0:
        fid_state = FidRealState(
            mode="features",
            payload={"features": real_feats_old.detach()},
            n=real_cnt,
        )
    else:
        real_sum = _get_attr(fid_tmp, ["real_features_sum", "_real_features_sum"])
        real_cov_sum = _get_attr(fid_tmp, ["real_features_cov_sum", "_real_features_cov_sum"])
        real_n = _get_attr(fid_tmp, ["real_features_num_samples", "_num_real_features"])
        if real_sum is None or real_cov_sum is None or real_n is None:
            raise RuntimeError("TorchMetrics FID: could not find real_* buffers (API changed).")
        if torch.is_tensor(real_n):
            n_val = int(real_n.item())
        else:
            n_val = int(real_n)
        fid_state = FidRealState(
            mode="moments",
            payload={"sum": real_sum.detach(), "cov_sum": real_cov_sum.detach()},
            n=n_val,
        )

    kid_state = _get_attr(kid_tmp, ["real_features", "_real_features"])
    if isinstance(kid_state, list) and len(kid_state) > 0:
        kid_real_feats = torch.cat([t.detach() for t in kid_state if isinstance(t, torch.Tensor)], dim=0)
    elif isinstance(kid_state, torch.Tensor) and kid_state.numel() > 0:
        kid_real_feats = kid_state.detach()
    else:
        kid_real_feats = None

    del fid_tmp, kid_tmp
    torch.cuda.empty_cache()
    gc.collect()

    print(f">> Real images used: {real_cnt}")
    return real_cnt, fid_state, kid_real_feats

def _inject_fid_real_state(fid_metric: FrechetInceptionDistance, state: FidRealState):
    if state.mode == "features":
        feats = state.payload["features"]
        if hasattr(fid_metric, "_real_features"):
            fid_metric._real_features = feats
            if hasattr(fid_metric, "_num_real_features"):
                fid_metric._num_real_features = state.n
            if hasattr(fid_metric, "real_features_num_samples"):
                fid_metric.real_features_num_samples = torch.tensor(state.n, device=feats.device)
        else:
            x = feats.float()
            mu = x.mean(dim=0)
            xc = x - mu
            cov = (xc.t() @ xc) / (x.size(0) - 1)
            if hasattr(fid_metric, "real_features_sum"):
                fid_metric.real_features_sum = (mu * x.size(0))
            if hasattr(fid_metric, "real_features_cov_sum"):
                uncentered_sum = cov * (x.size(0) - 1) + x.size(0) * (mu[:, None] @ mu[None, :])
                fid_metric.real_features_cov_sum = uncentered_sum
            if hasattr(fid_metric, "real_features_num_samples"):
                fid_metric.real_features_num_samples = torch.tensor(x.size(0), device=x.device)
            if hasattr(fid_metric, "_num_real_features"):
                fid_metric._num_real_features = x.size(0)
    else:
        if hasattr(fid_metric, "real_features_sum"):
            fid_metric.real_features_sum = state.payload["sum"]
        if hasattr(fid_metric, "real_features_cov_sum"):
            fid_metric.real_features_cov_sum = state.payload["cov_sum"]
        if hasattr(fid_metric, "real_features_num_samples"):
            n_tensor = (state.payload["sum"].new_tensor(state.n)
                        if isinstance(state.payload["sum"], torch.Tensor) else torch.tensor(state.n))
            fid_metric.real_features_num_samples = n_tensor
        if hasattr(fid_metric, "_num_real_features"):
            fid_metric._num_real_features = state.n

@torch.inference_mode()
def evaluate_epochs(
    data_dir: Path,
    ckpt_remote: str,
    epochs: List[int],
    n_gen: int = 2000,
    batch_gen: int = 64,
    nz: int = NZ,
    ngf: int = NGF,
    seed: int = SEED,
    resize_to_inception: bool = True,
    real_max: Optional[int] = None,
    device: Optional[str] = None,
    out_csv: Optional[Path] = None,
    real_batch: int = 128,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    real_cnt, fid_state, kid_real_feats = prepare_real_feature_buffers(
        data_dir=data_dir,
        device=device,
        resize_to_inception=resize_to_inception,
        real_max=real_max,
        real_batch=real_batch,
    )

    rows = []
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    for ep in epochs:
        print(f"\n===== Epoch {ep} =====")
        ema_path = f"{ckpt_remote.rstrip('/')}/G_EMA_epoch_{ep:03d}.pt"
        print(f">> Loading EMA weights: {ema_path}")
        state = load_state_dict_from_remote(ema_path, map_location=device)

        G = Generator256(nz=nz, ngf=ngf).to(device)
        G.load_state_dict(state, strict=True)
        G.eval()

        fid_ep = FrechetInceptionDistance(feature=2048).to(device)
        kid_ep = KernelInceptionDistance(subset_size=1000).to(device)
        is_ep = InceptionScore().to(device)

        _inject_fid_real_state(fid_ep, fid_state)

        if kid_real_feats is not None:
            if hasattr(kid_ep, "real_features") and isinstance(kid_ep.real_features, list):
                kid_ep.real_features = [kid_real_feats]
            if hasattr(kid_ep, "_real_features"):
                kid_ep._real_features = kid_real_feats
            if hasattr(kid_ep, "_num_real_features"):
                kid_ep._num_real_features = real_cnt

        left = n_gen
        pbar = tqdm(total=n_gen, desc=f"Generate+metrics (ep {ep})")
        while left > 0:
            b = min(batch_gen, left)
            z = torch.randn(b, nz, 1, 1, device=device)
            fake = (G(z).clamp(-1, 1) + 1) * 0.5
            if resize_to_inception:
                fake = torch.nn.functional.interpolate(
                    fake, size=(299, 299), mode="bilinear", align_corners=False
                )
            fake_uint8 = (fake * 255.0).round().clamp(0, 255).to(torch.uint8)

            fid_ep.update(fake_uint8, real=False)
            kid_ep.update(fake_uint8, real=False)
            is_ep.update(fake_uint8)

            left -= b
            pbar.update(b)
        pbar.close()

        fid_score = fid_ep.compute().item()
        kid_mean, kid_std = kid_ep.compute()
        kid_mean, kid_std = kid_mean.item(), kid_std.item()
        is_mean_t, is_std_t = is_ep.compute()
        is_mean, is_std = is_mean_t.item(), is_std_t.item()

        print(f"Epoch {ep:>3} | FID: {fid_score:.3f} | KID: {kid_mean:.5f} ± {kid_std:.5f} | IS: {is_mean:.3f} ± {is_std:.3f}")

        rows.append({
            "epoch": ep,
            "n_gen": n_gen,
            "real_used": real_cnt,
            "FID": fid_score,
            "KID_mean": kid_mean,
            "KID_std": kid_std,
            "IS_mean": is_mean,
            "IS_std": is_std,
        })

        del G, state, fid_ep, kid_ep, is_ep
        torch.cuda.empty_cache()
        gc.collect()

    if out_csv is not None and len(rows) > 0:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n>> Saved CSV: {out_csv}")

def parse_epochs_list(raw: List[str]) -> List[int]:
    out = []
    for tok in raw:
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(tok))
    return sorted(set(out))

def main():
    ap = argparse.ArgumentParser(
        "Evaluate FID/KID/IS for DCGAN G_EMA checkpoints (PNG dataset, compatible with different TorchMetrics versions)"
    )
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Folder with real PNG images (e.g. lhq_256)")
    ap.add_argument("--ckpt_remote", type=str, required=True,
                    help="e.g. mega:dcgan_runs/nature_lhq256")
    ap.add_argument("--epochs", nargs="+", required=True,
                    help="List of epochs, e.g. 140-160")
    ap.add_argument("--n_gen", type=int, default=2000,
                    help="Number of generated images per epoch")
    ap.add_argument("--batch_gen", type=int, default=64,
                    help="Batch size for FAKE generation")
    ap.add_argument("--nz", type=int, default=NZ)
    ap.add_argument("--ngf", type=int, default=NGF)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no_resize299", action="store_true",
                    help="Do not resize to 299x299")
    ap.add_argument("--real_max", type=int, default=None,
                    help="Maximum number of real images (e.g. 50000)")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda|cpu")
    ap.add_argument("--out_csv", type=str, default="metrics_fid_kid_is.csv")
    ap.add_argument("--real_batch", type=int, default=128,
                    help="Batch size for REAL (FID/KID)")
    args = ap.parse_args()

    epochs = parse_epochs_list(args.epochs)
    evaluate_epochs(
        data_dir=Path(args.data_dir),
        ckpt_remote=args.ckpt_remote,
        epochs=epochs,
        n_gen=args.n_gen,
        batch_gen=args.batch_gen,
        nz=args.nz,
        ngf=args.ngf,
        seed=args.seed,
        resize_to_inception=not args.no_resize299,
        real_max=args.real_max,
        device=args.device,
        out_csv=Path(args.out_csv),
        real_batch=args.real_batch,
    )

if __name__ == "__main__":
    main()
