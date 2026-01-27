# __init__.py (root của repo ComfyUI-OmarioNodes)
# Gom & đăng ký các node từ ./nodes/*

from .blend_scheduler import DualEndpointColorBlendScheduler
from .mask_clamped_crop import MaskClampedCrop

NODE_CLASS_MAPPINGS = {
    "DualEndpointColorBlendScheduler": DualEndpointColorBlendScheduler,
    "MaskClampedCrop": MaskClampedCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualEndpointColorBlendScheduler": "Dual Endpoint Color Blend (by Frames)",
    "MaskClampedCrop": "Mask Tracking Crop (Clamped)",
}
