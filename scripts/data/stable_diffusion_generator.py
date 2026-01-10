import os
import time
import random
from typing import List
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

queries: List[str] = ["people", "city", "nature", "animals", "food", "architecture"]

NUM_PER_CATEGORY = 30

WIDTH = 512
HEIGHT = 512

OUT_DIR = "stable_diffusion"

NUM_INFERENCE_STEPS = 35
GUIDANCE_SCALE = 6.5

MODEL_ID = "runwayml/stable-diffusion-v1-5"

NEGATIVE_PROMPT = "low quality, blurry, watermark, text, logo, jpeg artifacts, deformed, extra limbs, poorly drawn"

REQUEST_DELAY = 0.05

FORCE_FULL_FP32 = True


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
    return f"High-quality detailed image of {category}, {style}, {safety}"


def init_pipeline(device: str) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32 if FORCE_FULL_FP32 or device == "cpu" else None,
        use_safetensors=True,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.to(device)

    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    return pipe


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  FORCE_FULL_FP32={FORCE_FULL_FP32}")

    pipe = init_pipeline(device)
    ensure_dir(OUT_DIR)

    for category in queries:
        print(f"\n=== Category (AI): {category} — target: {NUM_PER_CATEGORY} ===")
        cat_dir = os.path.join(OUT_DIR, category.replace(" ", "_"))
        ensure_dir(cat_dir)

        for i in range(NUM_PER_CATEGORY):
            prompt = build_prompt(category, i)

            seed = random.randint(0, 2**31 - 1)
            generator = torch.Generator(device=device).manual_seed(seed)

            try:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=WIDTH,
                    height=HEIGHT,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    generator=generator,
                )
                image = result.images[0]

                filename = f"{category.replace(' ', '_')}_{i+1:03d}.png"
                dest_path = os.path.join(cat_dir, filename)
                image.save(dest_path)

                print(f"{i+1}/{NUM_PER_CATEGORY} — {dest_path} (seed={seed})")
                time.sleep(REQUEST_DELAY)

            except RuntimeError as e:
                emsg = str(e).lower()
                if "out of memory" in emsg and device == "cuda":
                    print("CUDA OOM — try a smaller resolution (e.g., 448x448) or fewer steps.")
                else:
                    print(f"Error during generation: {e}")
                time.sleep(0.5)
                continue

        print(f"Done: {category} — images saved to {cat_dir}")

    print(f"\nAll set! All AI images saved in folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
