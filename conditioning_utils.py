import os
import re
from pathlib import Path

import folder_paths
import torch


CONDITION_FOLDER = "conditions"
CONDITION_EXTENSIONS = {".pt", ".ckpt", ".safetensors"}


if CONDITION_FOLDER not in folder_paths.folder_names_and_paths:
    condition_path = os.path.join(folder_paths.models_dir, CONDITION_FOLDER)
    folder_paths.folder_names_and_paths[CONDITION_FOLDER] = (
        [condition_path],
        CONDITION_EXTENSIONS,
    )


def _safe_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = name.strip("._-")
    return name or "condition"


def _resolve_condition_dir() -> Path:
    try:
        dirs = folder_paths.get_folder_paths(CONDITION_FOLDER)
        if dirs:
            return Path(dirs[0])
    except Exception:
        pass

    return Path(folder_paths.models_dir) / CONDITION_FOLDER


def _condition_file(filename: str, ensure_dir: bool = True) -> Path:
    resolved_dir = _resolve_condition_dir()
    if ensure_dir:
        resolved_dir.mkdir(parents=True, exist_ok=True)

    safe = _safe_name(filename)
    if Path(safe).suffix.lower() not in CONDITION_EXTENSIONS:
        safe += ".pt"

    return resolved_dir / safe


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


class SaveConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("CONDITIONING",),
                "filename": ("STRING", {"default": "my_condition"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "conditioning/utils"
    OUTPUT_NODE = True

    def save(self, condition, filename, overwrite):
        path = _condition_file(filename, ensure_dir=True)

        if path.exists() and not overwrite:
            print(f"[OmarioNodes] Condition already exists, skipped: {path}")
            return ()

        try:
            tensors_to_save = []
            for item in condition:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    cond = _to_cpu(item[0])
                    cond_info = _to_cpu(item[1]) if len(item) > 1 else {}
                    tensors_to_save.append((cond, cond_info))
                else:
                    tensors_to_save.append(_to_cpu(item))

            payload = {
                "version": 1,
                "condition": tensors_to_save,
                "original_filename": filename,
            }

            tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            print(f"[OmarioNodes] Condition saved: {path}")
        except Exception as error:
            print(f"[OmarioNodes] Error saving condition: {error}")

        return ()


class LoadConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            condition_dir = _resolve_condition_dir()
            condition_dir.mkdir(parents=True, exist_ok=True)

            files = folder_paths.get_filename_list(CONDITION_FOLDER)
            file_list = [
                filename
                for filename in files
                if Path(filename).suffix.lower() in CONDITION_EXTENSIONS
            ]
            if not file_list:
                file_list = ["(no condition files found)"]
        except Exception as error:
            print(f"[OmarioNodes] Error listing condition files: {error}")
            file_list = ["(error listing files)"]

        return {
            "required": {
                "filename": (
                    file_list,
                    {"default": file_list[0]},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("condition",)
    FUNCTION = "load"
    CATEGORY = "conditioning/utils"

    @classmethod
    def IS_CHANGED(cls, filename):
        path = _condition_file(filename, ensure_dir=False)
        if not path.exists():
            return "missing_file"

        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def load(self, filename):
        path = _condition_file(filename, ensure_dir=False)

        if not path.exists():
            print(f"[OmarioNodes] Condition file not found: {path}")
            return ([],)

        try:
            payload = _torch_load(path)
            if isinstance(payload, dict) and "condition" in payload:
                loaded_condition = payload["condition"]
            else:
                loaded_condition = payload

            print(f"[OmarioNodes] Condition loaded: {path}")
            return (loaded_condition,)
        except Exception as error:
            print(f"[OmarioNodes] Error loading condition: {error}")
            return ([],)
