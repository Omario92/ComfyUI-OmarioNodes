# __init__.py (root của repo ComfyUI-OmarioNodes)
# Gom & đăng ký các node từ ./nodes/*

from .blend_scheduler import DualEndpointColorBlendScheduler
from .gemma_api_text_encode import GemmaAPITextEncode
from .mask_clamped_crop import MaskClampedCrop
from .mask_clamped_crop_sticky import MaskClampedCropSticky
from .light_leaks_transition import LightLeaksTransition

NODE_CLASS_MAPPINGS = {
    "DualEndpointColorBlendScheduler": DualEndpointColorBlendScheduler,
    "GemmaAPITextEncode": GemmaAPITextEncode,
    "MaskClampedCrop": MaskClampedCrop,
    "MaskClampedCropSticky": MaskClampedCropSticky,
    "LightLeaksTransition": LightLeaksTransition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualEndpointColorBlendScheduler": "Dual Endpoint Color Blend (by Frames)",
    "GemmaAPITextEncode": "LTX-2 API Text Encode",
    "MaskClampedCrop": "Mask Tracking Crop (Clamped)",
    "MaskClampedCropSticky": "Mask Tracking Crop (Sticky)",
    "LightLeaksTransition": "🔦 Light Leaks Transition (like CrossFadeImages)",
}
