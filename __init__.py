# __init__.py (root của repo ComfyUI-OmarioNodes)
# Gom & đăng ký các node từ ./nodes/*

from .blend_scheduler import DualEndpointColorBlendScheduler
from .mask_clamped_crop import MaskClampedCrop
from .mask_clamped_crop_sticky import MaskClampedCropSticky

NODE_CLASS_MAPPINGS = {
    "DualEndpointColorBlendScheduler": DualEndpointColorBlendScheduler,
    "MaskClampedCrop": MaskClampedCrop,
    "MaskClampedCropSticky": MaskClampedCropSticky,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualEndpointColorBlendScheduler": "Dual Endpoint Color Blend (by Frames)",
    "MaskClampedCrop": "Mask Tracking Crop (Clamped)",
    "MaskClampedCropSticky": "Mask Tracking Crop (Sticky)",
}
