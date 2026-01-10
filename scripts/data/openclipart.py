from datasets import load_dataset
import os
import subprocess
import tempfile
from tqdm import tqdm

N = 350

OUT_DIR = "SVG"
os.makedirs(OUT_DIR, exist_ok=True)

INKSCAPE = "inkscape"

ds = load_dataset(
    "nyuuzyou/openclipart",
    split="train",
    streaming=True,
)

with tempfile.TemporaryDirectory() as tmpdir:
    for i, example in tqdm(
        enumerate(ds),
        total=N,
        desc="Converting"
    ):
        if i >= N:
            break

        svg_content = example["svg_content"]

        base_name = f"{i:06d}"
        tmp_svg = os.path.join(tmpdir, base_name + ".svg")
        out_png = os.path.join(OUT_DIR, base_name + ".png")

        with open(tmp_svg, "w", encoding="utf-8") as f:
            f.write(svg_content)

        try:
            subprocess.run(
                [
                    INKSCAPE,
                    tmp_svg,
                    "--export-type=png",
                    f"--export-filename={out_png}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except Exception as e:
            print(f"[WARN] Problem with {base_name}: {e}")
            continue

print(f"Done! Files saved to folder: {OUT_DIR}")
