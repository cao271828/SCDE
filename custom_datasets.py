"""
custom_datasets.py — Custom dataset registry for LDGnet project.
Auto-imported by datasets.py: DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)

HyRANK: Loukia (14 classes) / Dioni (12 classes, missing 6,8)
- Remove label 6,8 from Loukia to align with Dioni → 12 effective classes
- Remap labels {1,2,3,4,5,7,9,10,11,12,13,14} → {0,1,2,3,4,5,6,7,8,9,10,11}
"""
import numpy as np
import tifffile

CUSTOM_DATASETS_CONFIG = {}

# After removing label 6,8: effective 12 classes
# Original labels: 1,2,3,4,5,7,9,10,11,12,13,14
# Remapped to:     0,1,2,3,4,5,6, 7, 8, 9,10,11
HYRANK_LABEL_MAP = {1:0, 2:1, 3:2, 4:3, 5:4, 7:5, 9:6, 10:7, 11:8, 12:9, 13:10, 14:11}
HYRANK_LABEL_VALUES = [
    "class_1", "class_2", "class_3", "class_4", "class_5",
    "class_7", "class_9", "class_10", "class_11",
    "class_12", "class_13", "class_14",
]
HYRANK_IGNORED_LABELS = [0]  # After remap, 0 = background
HYRANK_RGB_BANDS = [29, 19, 9]

def _remap_gt(gt, label_map):
    """Remap GT labels: {1,2,...,14} → {0,1,...,11}, bg stays 0."""
    out = np.zeros_like(gt)
    for orig, new in label_map.items():
        out[gt == orig] = new
    return out

def _hyrank_loader_loukia(folder):
    """Load Loukia GeoTIFF."""
    img = tifffile.imread(folder + "Loukia.tif").astype(np.float32)
    gt = tifffile.imread(folder + "Loukia_GT.tif").astype(np.int64)
    gt = _remap_gt(gt, HYRANK_LABEL_MAP)
    return img, gt, HYRANK_RGB_BANDS, HYRANK_IGNORED_LABELS, HYRANK_LABEL_VALUES, None

def _hyrank_loader_dioni(folder):
    """Load Dioni GeoTIFF."""
    img = tifffile.imread(folder + "Dioni.tif").astype(np.float32)
    gt = tifffile.imread(folder + "Dioni_GT.tif").astype(np.int64)
    gt = _remap_gt(gt, HYRANK_LABEL_MAP)
    return img, gt, HYRANK_RGB_BANDS, HYRANK_IGNORED_LABELS, HYRANK_LABEL_VALUES, None

CUSTOM_DATASETS_CONFIG = {
    'Loukia': {
        'img': 'Loukia.tif',
        'gt': 'Loukia_GT.tif',
        'loader': _hyrank_loader_loukia,
    },
    'Dioni': {
        'img': 'Dioni.tif',
        'gt': 'Dioni_GT.tif',
        'loader': _hyrank_loader_dioni,
    },
}
