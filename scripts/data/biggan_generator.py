import os
import time
import random
from typing import List, Dict

import numpy as np
from PIL import Image
import torch

from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample

queries: List[str] = ["people", "city", "nature", "animals", "food", "architecture"]
NUM_PER_CATEGORY = 30
OUT_DIR = "biggan"

MODEL_NAME = "biggan-deep-256"
TRUNCATION_BASE = 0.4
TRUNCATION_JITTER = 0.08

REQUEST_DELAY = 0.03
SEED_BASE = 14579

SAFE_POOL: Dict[str, List[str]] = {
    "people": [
        "scuba diver", "baseball player", "bridegroom", "parachute", "mountain bike",
        "snowmobile", "horse cart", "drum", "stage", "gown",
    ],
    "city": [
        "streetcar", "trolleybus", "traffic light", "parking meter", "bookshop",
        "school bus", "garbage truck", "tow truck", "street sign", "tram",
    ],
    "nature": [
        "valley", "lakeside", "seashore", "cliff", "geyser",
        "alp", "volcano", "coral reef", "lakeshore", "sandbar",
    ],
    "animals": [
        "tiger", "zebra", "dalmatian", "tabby cat", "goldfinch",
        "husky", "flamingo", "peacock", "lion", "elephant",
    ],
    "food": [
        "pizza", "cheeseburger", "hotdog", "pretzel", "bagel",
        "ice cream", "French loaf", "guacamole", "carbonara", "burrito",
    ],
    "architecture": [
        "castle", "church", "mosque", "palace", "monastery",
        "stupa", "bell cote", "triumphal arch", "lighthouse", "dome",
    ],
}

CLASS_BLEND_MIN = 1
CLASS_BLEND_MAX = 3


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def init_biggan(device: str) -> BigGAN:
    model = BigGAN.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return model


def pil_from_biggan_tensor(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    t = (t + 1.0) / 2.0
    t = torch.clamp(t, 0, 1)
    arr = (t.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def save_png(img: Image.Image, dest_path: str) -> None:
    ensure_dir(os.path.dirname(dest_path))
    img.save(dest_path, format="PNG")


def resolve_one_hot_safe(name: str):
    try:
        return one_hot_from_names([name], batch_size=1)
    except AssertionError:
        return None


def build_class_vector(names: List[str], weights: List[float]) -> np.ndarray:
    weights = np.array(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)

    vec = None
    for nm, w in zip(names, weights):
        oh = resolve_one_hot_safe(nm)
        if oh is None:
            continue
        v = oh.astype(np.float32) * w
        vec = v if vec is None else (vec + v)

    return vec


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {MODEL_NAME}")

    random.seed(SEED_BASE)
    np.random.seed(SEED_BASE)
    torch.manual_seed(SEED_BASE)

    model = init_biggan(device)
    ensure_dir(OUT_DIR)

    for category in queries:
        pool = SAFE_POOL.get(category, [])
        if not pool:
            print(f"[warn] Empty pool for category='{category}', skipping.")
            continue

        print(f"\n=== Category (BigGAN PNG): {category} — target: {NUM_PER_CATEGORY} ===")
        cat_dir = os.path.join(OUT_DIR, category.replace(" ", "_"))
        ensure_dir(cat_dir)

        for i in range(NUM_PER_CATEGORY):
            seed = random.randint(0, 2**31 - 1)
            np.random.seed(seed)
            torch.manual_seed(seed)

            k = random.randint(CLASS_BLEND_MIN, CLASS_BLEND_MAX)
            names = random.sample(pool, k=k)

            ws = np.random.rand(k).tolist()
            class_vec_np = build_class_vector(names, ws)
            if class_vec_np is None:
                print(f"[skip] Could not build class vector from names={names}")
                continue

            class_vec = torch.from_numpy(class_vec_np).to(device)

            trunc = float(np.clip(
                TRUNCATION_BASE + np.random.uniform(-TRUNCATION_JITTER, TRUNCATION_JITTER),
                0.2, 0.8
            ))

            noise_np = truncated_noise_sample(truncation=trunc, batch_size=1)
            noise = torch.from_numpy(noise_np).to(device)

            with torch.no_grad():
                out = model(noise, class_vec, trunc)

            img = pil_from_biggan_tensor(out)

            filename = f"{category.replace(' ', '_')}_{i+1:03d}.png"
            dest_path = os.path.join(cat_dir, filename)

            save_png(img, dest_path)

            print(f"{i+1}/{NUM_PER_CATEGORY} — {dest_path}  |  mix={names}  |  trunc={trunc:.2f}")

            time.sleep(REQUEST_DELAY)

        print(f"Done: {category} — images saved to {cat_dir}")

    print(f"\nAll set! BigGAN PNG images saved in folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
