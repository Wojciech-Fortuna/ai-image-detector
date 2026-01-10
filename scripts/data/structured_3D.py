import os
import shutil

SOURCE_ROOT = "Structured3D"
OUTPUT_DIR = "Structured3D_300_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0
MAX_IMAGES = 300

for scene in sorted(os.listdir(SOURCE_ROOT)):
    scene_path = os.path.join(SOURCE_ROOT, scene)
    if not os.path.isdir(scene_path) or not scene.startswith("scene_"):
        continue

    rendering_path = os.path.join(scene_path, "2D_rendering")
    if not os.path.isdir(rendering_path):
        continue

    for view_id in sorted(os.listdir(rendering_path)):
        view_path = os.path.join(rendering_path, view_id)
        if not os.path.isdir(view_path):
            continue

        persp_path = os.path.join(view_path, "perspective")
        if not os.path.isdir(persp_path):
            continue

        full_path = os.path.join(persp_path, "full")
        if not os.path.isdir(full_path):
            continue

        for config in ["0", "1", "2"]:
            config_path = os.path.join(full_path, config)
            if not os.path.isdir(config_path):
                continue

            img_path = os.path.join(config_path, "rgb_rawlight.png")
            if os.path.isfile(img_path):
                # Copy to output directory
                target_name = f"{count:04d}.png"
                shutil.copy(img_path, os.path.join(OUTPUT_DIR, target_name))

                count += 1
                print(f"Copied: {img_path} → {target_name}")

                if count >= MAX_IMAGES:
                    print(f"Finished! {MAX_IMAGES} images extracted.")
                    quit()

print(f"Finished scanning — found only {count} images.")
