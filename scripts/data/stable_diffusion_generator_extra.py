import os
import time
import random
from typing import List
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

NUM_ARTISTIC = 30
NUM_VECTOR = 30
NUM_STRUCTURED3D = 30

WIDTH = 512
HEIGHT = 512

OUT_DIR = "stable_diffusion"

NUM_INFERENCE_STEPS = 35
GUIDANCE_SCALE = 6.5

MODEL_ID = "runwayml/stable-diffusion-v1-5"

NEGATIVE_COMMON = (
    "low quality, blurry, watermark, text, logo, jpeg artifacts, deformed, extra limbs, poorly drawn"
)

NEGATIVE_VECTOR = NEGATIVE_COMMON + ", photo, 3d render, realistic shading, complex textures"
NEGATIVE_ARTISTIC = NEGATIVE_COMMON
NEGATIVE_STRUCTURED3D = NEGATIVE_COMMON + ", noise, film grain, photo, clutter, people"

REQUEST_DELAY = 0.05

FORCE_FULL_FP32 = True


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


def build_artistic_prompt(idx: int) -> str:
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
    base = BASE_SUBJECTS_ARTVECTOR[idx % len(BASE_SUBJECTS_ARTVECTOR)]
    style = styles[idx % len(styles)]
    safety = "no celebrities, no trademarks, no identifiable private individuals"
    return f"Artistic digital artwork of {base}, {style}, {safety}"


def build_vector_prompt(idx: int) -> str:
    base = BASE_SUBJECTS_ARTVECTOR[idx % len(BASE_SUBJECTS_ARTVECTOR)]
    style = (
        "flat vector illustration, clean lines, solid colors, sharp edges, "
        "minimal shading, simplified geometric shapes, svg style, high contrast"
    )
    safety = "no celebrities, no trademarks, no identifiable private individuals"
    return f"{base}, {style}, {safety}"


def build_structured3d_prompt(idx: int) -> str:
    base = BASE_SUBJECTS_STRUCTURED3D[idx % len(BASE_SUBJECTS_STRUCTURED3D)]
    style = (
        "synthetic 3d indoor scene, clean CGI, smooth flat materials, "
        "neutral soft global illumination, no detailed textures, simple geometry, "
        "orthographic or isometric look, uniform colors, dataset style render, "
        "sharp edges, structured3d-like preview"
    )
    safety = "no celebrities, no trademarks, no identifiable private individuals"
    return f"{base}, {style}, {safety}"


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


def generate_batch(
    pipe: StableDiffusionPipeline,
    count: int,
    build_fn,
    negative_prompt: str,
    out_subdir: str,
    prefix: str,
    device: str,
) -> None:
    if count <= 0:
        return

    target_dir = os.path.join(OUT_DIR, out_subdir)
    ensure_dir(target_dir)

    print(f"\n=== Generuję: {out_subdir} — liczba obrazów: {count} ===")

    for i in range(count):
        prompt = build_fn(i)
        seed = random.randint(0, 2**31 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)

        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=WIDTH,
                height=HEIGHT,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=generator,
            )
            image = result.images[0]

            filename = f"{prefix}_{i+1:03d}.png"
            dest_path = os.path.join(target_dir, filename)
            image.save(dest_path)

            print(f"{i+1}/{count} — {dest_path} (seed={seed})")
            print(f"  prompt: {prompt}")
            time.sleep(REQUEST_DELAY)

        except RuntimeError as e:
            emsg = str(e).lower()
            if "out of memory" in emsg and device == "cuda":
                print("CUDA OOM — spróbuj mniejszej rozdzielczości (np. 448x448) lub mniej kroków.")
            else:
                print(f"Error during generation: {e}")
            time.sleep(0.5)
            continue


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  FORCE_FULL_FP32={FORCE_FULL_FP32}")

    ensure_dir(OUT_DIR)
    pipe = init_pipeline(device)

    generate_batch(
        pipe=pipe,
        count=NUM_ARTISTIC,
        build_fn=build_artistic_prompt,
        negative_prompt=NEGATIVE_ARTISTIC,
        out_subdir="artistic",
        prefix="art",
        device=device,
    )

    generate_batch(
        pipe=pipe,
        count=NUM_VECTOR,
        build_fn=build_vector_prompt,
        negative_prompt=NEGATIVE_VECTOR,
        out_subdir="vector",
        prefix="vector",
        device=device,
    )

    generate_batch(
        pipe=pipe,
        count=NUM_STRUCTURED3D,
        build_fn=build_structured3d_prompt,
        negative_prompt=NEGATIVE_STRUCTURED3D,
        out_subdir="structured3d_like",
        prefix="structured3d",
        device=device,
    )

    print(f"\nAll done! Images saved in folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
