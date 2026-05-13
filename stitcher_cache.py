import os
import re
from pathlib import Path

import torch


CACHE_VERSION = 1
DEFAULT_CACHE_KEY = "HALIDA_FACESWAP_LPBN_V2"
ENV_CACHE_DIR = "OMARIO_INPAINT_CROP_CACHE_DIR"
FALLBACK_CACHE_DIR = "/workspace/ComfyUI/user/default/inpaint_crop_cache"


def _safe_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = name.strip("._-")
    return name or "default_cache"


def _comfy_user_dir():
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.get_user_directory())
    except Exception:
        return None


def _default_cache_dir() -> Path:
    env_dir = os.getenv(ENV_CACHE_DIR, "").strip()
    if env_dir:
        return Path(env_dir).expanduser()

    comfy_user = _comfy_user_dir()
    if comfy_user is not None:
        return comfy_user / "default" / "inpaint_crop_cache"

    return Path(FALLBACK_CACHE_DIR).expanduser()


def _resolve_cache_dir(cache_dir: str | None) -> Path:
    raw = str(cache_dir).strip() if cache_dir is not None else ""
    if raw:
        return Path(raw).expanduser()
    return _default_cache_dir()


def _cache_file(cache_dir: str, cache_key: str, ensure_dir: bool = True) -> Path:
    resolved_dir = _resolve_cache_dir(cache_dir)
    if ensure_dir:
        resolved_dir.mkdir(parents=True, exist_ok=True)
    return resolved_dir / f"{_safe_name(cache_key)}.pt"


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
                "cache_key": ("STRING", {"default": DEFAULT_CACHE_KEY}),
                "cache_dir": ("STRING", {"default": FALLBACK_CACHE_DIR}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask", "cache_path")
    FUNCTION = "save"
    CATEGORY = "inpaint/cache"

    def save(self, stitcher, cropped_image, cropped_mask, cache_key, cache_dir, overwrite):
        path = _cache_file(cache_dir, cache_key, ensure_dir=True)

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
                "cache_key": ("STRING", {"default": DEFAULT_CACHE_KEY}),
                "cache_dir": ("STRING", {"default": FALLBACK_CACHE_DIR}),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask", "cache_path")
    FUNCTION = "load"
    CATEGORY = "inpaint/cache"

    @classmethod
    def IS_CHANGED(cls, cache_key, cache_dir):
        path = _cache_file(cache_dir, cache_key, ensure_dir=False)
        if not path.exists():
            return "missing"
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def load(self, cache_key, cache_dir):
        path = _cache_file(cache_dir, cache_key, ensure_dir=False)

        if not path.exists():
            raise FileNotFoundError(
                f"Inpaint crop cache not found: {path}. "
                "Use a persistent volume on Runpod and keep cache_dir identical between pods. "
                f"You can also set {ENV_CACHE_DIR}."
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
