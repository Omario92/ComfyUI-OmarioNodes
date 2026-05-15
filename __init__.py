# __init__.py (root cua repo ComfyUI-OmarioNodes)
# Gom & dang ky cac node tu ./nodes/*

from .blend_scheduler import DualEndpointColorBlendScheduler
from .gemma_api_text_encode import GemmaAPITextEncode
from .mask_clamped_crop import MaskClampedCrop
from .mask_clamped_crop_sticky import MaskClampedCropSticky
from .light_leaks_transition import LightLeaksTransition
from .stitcher_cache import SaveInpaintCropCache, LoadInpaintCropCache
from .conditioning_utils import SaveConditioning, LoadConditioning

NODE_CLASS_MAPPINGS = {
    "DualEndpointColorBlendScheduler": DualEndpointColorBlendScheduler,
    "GemmaAPITextEncode": GemmaAPITextEncode,
    "MaskClampedCrop": MaskClampedCrop,
    "MaskClampedCropSticky": MaskClampedCropSticky,
    "LightLeaksTransition": LightLeaksTransition,
    "SaveInpaintCropCache": SaveInpaintCropCache,
    "LoadInpaintCropCache": LoadInpaintCropCache,
    "SaveConditioning": SaveConditioning,
    "LoadConditioning": LoadConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualEndpointColorBlendScheduler": "Dual Endpoint Color Blend (by Frames)",
    "GemmaAPITextEncode": "LTX-2 API Text Encode",
    "MaskClampedCrop": "Mask Tracking Crop (Clamped)",
    "MaskClampedCropSticky": "Mask Tracking Crop (Sticky)",
    "LightLeaksTransition": "Light Leaks Transition (like CrossFadeImages)",
    "SaveInpaintCropCache": "Save Inpaint Crop Cache",
    "LoadInpaintCropCache": "Load Inpaint Crop Cache",
    "SaveConditioning": "Save Conditioning",
    "LoadConditioning": "Load Conditioning",
}
