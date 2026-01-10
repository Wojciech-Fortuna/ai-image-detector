import base64
import io
import json
from pathlib import Path
from typing import Dict, Any
from PIL import Image, ExifTags


def load_image(file_obj) -> Image.Image:
    img = Image.open(file_obj)
    img.load()
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def extract_exif(img: Image.Image) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        raw = img.getexif()
        if raw:
            for tag_id, value in raw.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag in {"Make", "Model", "Software", "DateTime", "Artist", "Copyright"}:
                    info[str(tag)] = str(value)
    except Exception:
        pass
    return info


def b64png_to_image(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data))


def save_report_json(report: Dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
