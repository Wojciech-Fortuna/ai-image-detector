import pytest
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def tiny_img() -> Image.Image:
    return Image.new("RGB", (2, 2), color=(123, 45, 67))
