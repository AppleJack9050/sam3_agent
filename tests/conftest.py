import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def ice_image() -> np.ndarray:
    """256x256 RGB image with a central blue-white 'ice' patch."""
    img = np.full((256, 256, 3), 40, dtype=np.uint8)
    img[80:176, 80:176] = (210, 225, 240)
    return img


@pytest.fixture
def ice_image_path(ice_image, tmp_path: Path) -> Path:
    p = tmp_path / "ice.png"
    Image.fromarray(ice_image, "RGB").save(p)
    return p


@pytest.fixture
def sky_and_ground_image() -> np.ndarray:
    """256x256 RGB: top half is blue sky, bottom half is dark gray ground
    with a bright blue-white 'ice' highlight in the lower-middle."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:128] = (120, 170, 230)   # sky (bluish)
    img[128:] = (55, 50, 48)      # dark ground
    img[160:210, 90:170] = (210, 225, 240)  # ice patch
    return img


@pytest.fixture
def sky_and_ground_image_path(sky_and_ground_image, tmp_path: Path) -> Path:
    p = tmp_path / "sky_ground.png"
    Image.fromarray(sky_and_ground_image, "RGB").save(p)
    return p
