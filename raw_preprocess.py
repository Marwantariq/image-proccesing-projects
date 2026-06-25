"""RAW Bayer preprocessing utilities, extracted from raw-hvi/final.ipynb cell 10."""
from pathlib import Path
import numpy as np

try:
    import rawpy
except ImportError:
    rawpy = None


def pack_raw_bayer(raw_path):
    """Read RAW, black/white-level normalise, pack 2x2 Bayer into 4 channels.

    Returns float32 array shaped (H/2, W/2, 4) in [0, 1].
    """
    if rawpy is None:
        raise ImportError("rawpy is required to read RAW files. `pip install rawpy`.")
    with rawpy.imread(str(raw_path)) as raw:
        raw_image = raw.raw_image_visible.astype(np.float32)
        black_levels = np.array(raw.black_level_per_channel, dtype=np.float32)
        black_level = float(np.mean(black_levels))
        white_level = float(raw.white_level)
        raw_image = np.maximum(raw_image - black_level, 0.0)
        raw_image = raw_image / max((white_level - black_level), 1.0)
        raw_image = np.clip(raw_image, 0.0, 1.0)
    H, W = raw_image.shape
    H -= H % 2; W -= W % 2
    raw_image = raw_image[:H, :W]
    packed = np.stack([
        raw_image[0:H:2, 0:W:2],
        raw_image[0:H:2, 1:W:2],
        raw_image[1:H:2, 1:W:2],
        raw_image[1:H:2, 0:W:2],
    ], axis=-1)
    return packed.astype(np.float32)


def raw_to_rgb_float(raw_path):
    """RAW -> RGB float in [0, 1], using camera white balance and no auto-bright."""
    if rawpy is None:
        raise ImportError("rawpy is required to read RAW files. `pip install rawpy`.")
    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16, user_flip=0)
    return np.clip(rgb.astype(np.float32) / 65535.0, 0.0, 1.0)


def estimate_exposure_ratio_from_name(filename: str) -> float:
    """Parse the exposure seconds from an SID filename like 10003_00_0.04s.ARW."""
    import re
    m = re.search(r"_(\d+\.?\d*)s\.[A-Z]{3,4}$", str(filename), re.IGNORECASE)
    if not m:
        return 1.0
    return float(m.group(1))
