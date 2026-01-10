import os
import time
import random
import base64
from typing import List
from openai import OpenAI

API_KEY = "PUT_YOUR_KEY_HERE"

if not API_KEY or API_KEY == "PUT_YOUR_KEY_HERE":
    print(
        "Missing API key. Paste it into the API_KEY variable at the top of the file."
    )
    exit(1)

client = OpenAI(api_key=API_KEY)

NUM_ARTISTIC = 30
NUM_VECTOR = 30
NUM_STRUCTURED3D = 30

WIDTH = 512
HEIGHT = 512
OUT_DIR = "dalle2"

NEGATIVE_COMMON = (
    "no text, no watermark, no logo, no caption, no border, "
    "no frame, no low quality, no blur"
)

REQUEST_DELAY = 0.1


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


BASE_SUBJECTS_ARTVECTOR: List[str] = [
    "abstract geometric composition with strong shapes",
    "surreal dreamlike landscape with floating islands",
    "expressionist portrait with strong emotions",
    "minimalist composition with lots of negative space",
    "retro-futuristic cityscape full of lights",
    "conceptual artwork focused on contrast and symbolism",
    "digital painting inspired by impressionism",
    "dynamic scene with strong motion and energy",
]

BASE_SUBJECTS_STRUCTURED3D: List[str] = [
    "modern living room interior with sofa and table",
    "minimalist bedroom interior with bed and large window",
    "simple office room with desk and chair",
    "corridor with doors and ceiling lights",
    "kitchen interior with cabinets and island",
    "open space studio apartment interior",
    "clean staircase area in a building",
    "small study room with shelves and desk",
]


def build_artistic_prompt(variant_idx: int) -> str:
    styles = [
        "highly detailed, painterly style",
        "textured brush strokes, rich details",
        "cinematic lighting, dramatic shadows",
        "soft lighting, subtle gradients",
        "vibrant colors, high contrast",
        "muted palette, moody atmosphere",
        "dynamic composition, strong leading lines",
        "experimental, glitch aesthetic",
    ]
    base = BASE_SUBJECTS_ARTVECTOR[variant_idx % len(BASE_SUBJECTS_ARTVECTOR)]
    style = styles[variant_idx % len(styles)]
    safety = "no celebrities, no trademarks, no identifiable private individuals"
    prompt = (
        f"Artistic digital artwork of {base}, {style}, "
        f"{safety}, {NEGATIVE_COMMON}"
    )
    return prompt


def build_vector_prompt(variant_idx: int) -> str:
    base = BASE_SUBJECTS_ARTVECTOR[variant_idx % len(BASE_SUBJECTS_ARTVECTOR)]
    style = (
        "flat vector illustration, clean lines, solid colors, sharp edges, "
        "minimal shading, simplified geometric shapes, svg style, high contrast"
    )
    safety = "no celebrities, no trademarks, no identifiable private individuals"
    prompt = f"{base}, {style}, {safety}, {NEGATIVE_COMMON}"
    return prompt


def build_structured3d_prompt(variant_idx: int) -> str:
    base = BASE_SUBJECTS_STRUCTURED3D[
        variant_idx % len(BASE_SUBJECTS_STRUCTURED3D)
    ]
    style = (
        "synthetic 3d indoor scene, clean CGI, smooth flat materials, "
        "neutral soft global illumination, no detailed textures, simple geometry, "
        "orthographic or isometric look, uniform colors, dataset style render, "
        "sharp edges, structured3d-like preview"
    )
    safety = "no celebrities, no trademarks, no identifiable private individuals"
    prompt = f"{base}, {style}, {safety}, {NEGATIVE_COMMON}"
    return prompt


def generate_image_dalle2(prompt: str, size: str = "512x512") -> bytes:
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        n=1,
        size=size,
        response_format="b64_json",
    )
    b64_data = response.data[0].b64_json
    return base64.b64decode(b64_data)


def generate_batch(
    name: str,
    count: int,
    build_fn,
    out_subdir: str,
    prefix: str,
) -> None:
    if count <= 0:
        return

    target_dir = os.path.join(OUT_DIR, out_subdir)
    ensure_dir(target_dir)

    print(f"\n=== Generating dataset: {name} — image count: {count} ===")

    for i in range(count):
        prompt = build_fn(i)
        try:
            image_bytes = generate_image_dalle2(
                prompt=prompt,
                size=f"{WIDTH}x{HEIGHT}",
            )

            filename = f"{prefix}_{i+1:03d}.png"
            dest_path = os.path.join(target_dir, filename)

            with open(dest_path, "wb") as f:
                f.write(image_bytes)

            print(f"{i+1}/{count} — {dest_path}")
            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"Error during generation: {e}")
            time.sleep(0.5)
            continue

    print(f"Done: {name} — images saved to {target_dir}")


def main():
    print("Using OpenAI DALL·E 2 API")
    ensure_dir(OUT_DIR)

    generate_batch(
        name="artistic",
        count=NUM_ARTISTIC,
        build_fn=build_artistic_prompt,
        out_subdir="artistic",
        prefix="art",
    )

    generate_batch(
        name="vector",
        count=NUM_VECTOR,
        build_fn=build_vector_prompt,
        out_subdir="vector",
        prefix="vector",
    )

    generate_batch(
        name="structured3d_like",
        count=NUM_STRUCTURED3D,
        build_fn=build_structured3d_prompt,
        out_subdir="structured3d_like",
        prefix="structured3d",
    )

    print(f"\nAll done! Images saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
