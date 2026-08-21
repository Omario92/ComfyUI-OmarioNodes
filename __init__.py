# __init__.py (root cua repo ComfyUI-OmarioNodes)
# Gom & dang ky cac node tu ./nodes/*

from .blend_scheduler import DualEndpointColorBlendScheduler
from .gemma_api_text_encode import GemmaAPITextEncode
from .mask_clamped_crop import MaskClampedCrop
from .mask_clamped_crop_sticky import MaskClampedCropSticky
from .light_leaks_transition import LightLeaksTransition
from .stitcher_cache import SaveInpaintCropCache, LoadInpaintCropCache
from .conditioning_utils import SaveConditioning, LoadConditioning
from .save_image_plus import SaveImagePlus
from .resolution_selector import VideoResolutionSelector
from .pad_batch_17n_plus_5 import PadBatchTo17nPlus5

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
    "SaveImagePlus": SaveImagePlus,
    "VideoResolutionSelector": VideoResolutionSelector,
    "PadBatchTo17nPlus5": PadBatchTo17nPlus5,
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
    "SaveImagePlus": "Save Image Plus",
    "VideoResolutionSelector": "Video Resolution Selector",
    "PadBatchTo17nPlus5": "Pad Batch to 17n+5",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
