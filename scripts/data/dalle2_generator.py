import os
import time
import random
from typing import List
from openai import OpenAI

API_KEY = "PUT_YOUR_KEY_HERE"

if API_KEY == "PUT_YOUR_KEY_HERE":
    print(
        "Missing API key. Paste it into the API_KEY variable at the top of the file."
    )
    exit(1)

client = OpenAI(api_key=API_KEY)

queries: List[str] = [
    "people",
    "city",
    "nature",
    "animals",
    "food",
    "architecture",
]

NUM_PER_CATEGORY = 30

WIDTH = 512
HEIGHT = 512
OUT_DIR = "dalle2"

NEGATIVE_SUFFIX = (
    "no text, no watermark, no logo, no caption, no border, "
    "no frame, no low quality, no blur"
)

REQUEST_DELAY = 0.1


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_prompt(category: str, variant_idx: int) -> str:
    styles = [
        "photorealistic, natural lighting",
        "cinematic lighting, shallow depth of field",
        "wide angle shot, environmental context",
        "close-up, detailed texture",
        "golden hour ambience",
        "overcast, soft light",
        "dynamic composition, dramatic scene",
        "balanced composition, rule of thirds",
        "minimalist background",
        "vibrant colors, high detail",
    ]

    style = styles[variant_idx % len(styles)]
    safety = "no celebrities, no trademarks, no identifiable private individuals"

    base = f"High-quality detailed image of {category}, {style}, {safety}"
    return f"{base}, {NEGATIVE_SUFFIX}"


def generate_image_dalle2(prompt: str, size: str = "512x512") -> bytes:
    """
    Generates a DALL·E 2 image and returns PNG file bytes.
    """
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        n=1,
        size=size,
        response_format="b64_json",
    )

    import base64
    b64_data = response.data[0].b64_json
    return base64.b64decode(b64_data)


def main():
    print("Using OpenAI DALL·E 2 API")

    ensure_dir(OUT_DIR)

    for category in queries:
        print(
            f"\n=== Category: {category} — target images: {NUM_PER_CATEGORY} ==="
        )

        cat_dir = os.path.join(OUT_DIR, category.replace(" ", "_"))
        ensure_dir(cat_dir)

        for i in range(NUM_PER_CATEGORY):
            prompt = build_prompt(category, i)

            try:
                image_bytes = generate_image_dalle2(
                    prompt=prompt,
                    size=f"{WIDTH}x{HEIGHT}",
                )

                filename = f"{category.replace(' ', '_')}_{i+1:03d}.png"
                dest_path = os.path.join(cat_dir, filename)

                with open(dest_path, "wb") as f:
                    f.write(image_bytes)

                print(f"{i+1}/{NUM_PER_CATEGORY} — {dest_path}")
                time.sleep(REQUEST_DELAY)

            except Exception as e:
                print(f"Error during generation: {e}")
                time.sleep(0.5)
                continue

        print(f"Done: {category} — images saved to {cat_dir}")

    print(f"\nAll done! Images saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
