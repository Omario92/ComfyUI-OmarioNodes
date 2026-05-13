import os
import re
from pathlib import Path

import torch


CACHE_VERSION = 1


def _safe_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = name.strip("._-")
    return name or "default_cache"


def _cache_file(cache_dir: str, cache_key: str) -> Path:
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{_safe_name(cache_key)}.pt"


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)

    return obj


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class SaveInpaintCropCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "cropped_image": ("IMAGE",),
                "cropped_mask": ("MASK",),
                "cache_key": ("STRING", {"default": "HALIDA_FACESWAP_LPBN_V2"}),
                "cache_dir": ("STRING", {"default": "/workspace/ComfyUI/user/default/inpaint_crop_cache"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask", "cache_path")
    FUNCTION = "save"
    CATEGORY = "inpaint/cache"

    def save(self, stitcher, cropped_image, cropped_mask, cache_key, cache_dir, overwrite):
        path = _cache_file(cache_dir, cache_key)

        if overwrite or not path.exists():
            payload = {
                "version": CACHE_VERSION,
                "cache_key": cache_key,
                "stitcher": _to_cpu(stitcher),
                "cropped_image": _to_cpu(cropped_image),
                "cropped_mask": _to_cpu(cropped_mask),
            }

            tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)

        return (stitcher, cropped_image, cropped_mask, str(path))


class LoadInpaintCropCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_key": ("STRING", {"default": "HALIDA_FACESWAP_LPBN_V2"}),
                "cache_dir": ("STRING", {"default": "/workspace/ComfyUI/user/default/inpaint_crop_cache"}),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask", "cache_path")
    FUNCTION = "load"
    CATEGORY = "inpaint/cache"

    @classmethod
    def IS_CHANGED(cls, cache_key, cache_dir):
        path = Path(cache_dir).expanduser() / f"{_safe_name(cache_key)}.pt"
        if not path.exists():
            return "missing"
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def load(self, cache_key, cache_dir):
        path = _cache_file(cache_dir, cache_key)

        if not path.exists():
            raise FileNotFoundError(
                f"Inpaint crop cache not found: {path}. "
                f"Run Save Inpaint Crop Cache once first."
            )

        payload = _torch_load(path)

        if payload.get("version") != CACHE_VERSION:
            raise ValueError(
                f"Unsupported cache version: {payload.get('version')}. "
                f"Expected: {CACHE_VERSION}"
            )

        return (
            payload["stitcher"],
            payload["cropped_image"],
            payload["cropped_mask"],
            str(path),
        )
