import json
import os
from datetime import datetime

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths


class SaveImagePlus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "subfolder": ("STRING", {"default": ""}),
                "add_timestamp": ("BOOLEAN", {"default": True}),
                "extension": (["png", "jpg", "webp"], {"default": "png"}),
                "quality": (
                    "INT",
                    {"default": 95, "min": 1, "max": 100, "step": 1},
                ),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def _build_file_prefix(self, filename_prefix: str, add_timestamp: bool) -> str:
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            return f"{filename_prefix}_{timestamp}"
        return filename_prefix

    def save_images(
        self,
        images,
        filename_prefix,
        subfolder,
        add_timestamp,
        extension,
        quality,
        save_metadata,
        prompt=None,
        extra_pnginfo=None,
    ):
        output_dir = self.output_dir
        if subfolder.strip():
            output_dir = os.path.join(output_dir, subfolder.strip().replace("..", ""))
        os.makedirs(output_dir, exist_ok=True)

        prefix = self._build_file_prefix(filename_prefix, add_timestamp)
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            prefix,
            output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )

        results = []
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            file = f"{filename}_{counter:05}_.{extension}"
            file_path = os.path.join(full_output_folder, file)

            pnginfo = None
            if save_metadata and extension == "png":
                pnginfo = PngInfo()
                if prompt is not None:
                    pnginfo.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        pnginfo.add_text(key, json.dumps(value))

            save_kwargs = {}
            if extension in {"jpg", "webp"}:
                save_kwargs["quality"] = quality
                if extension == "jpg":
                    img = img.convert("RGB")
            if extension == "png" and pnginfo is not None:
                save_kwargs["pnginfo"] = pnginfo

            img.save(file_path, **save_kwargs)
            results.append(
                {
                    "filename": file,
                    "subfolder": os.path.relpath(full_output_folder, self.output_dir),
                    "type": self.type,
                }
            )
            counter += 1

        return {"ui": {"images": results}}
