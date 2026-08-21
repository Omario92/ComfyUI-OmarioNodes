"""Resolution presets for ComfyUI workflows."""

from math import sqrt


ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}

MEGAPIXEL_PRESETS = [
    "0.25",
    "0.5",
    "0.75",
    "1.0",
    "1.5",
    "2.0",
    "3.0",
    "4.0",
    "6.0",
    "8.0",
    "12.0",
    "16.0",
]

MULTIPLE_PRESETS = ["8", "16", "32", "64", "128"]


def _round_to_multiple(value, multiple):
    """Round to the nearest positive multiple without banker's rounding."""
    return max(multiple, int(value / multiple + 0.5) * multiple)


def _floor_to_multiple(value, multiple):
    """Round down so a configured maximum is never exceeded."""
    return max(multiple, int(value // multiple) * multiple)


def calculate_resolution(
    aspect_ratio,
    size_mode,
    megapixels,
    max_pixels,
    multiple,
    shorter_pixels=768,
):
    """Calculate a resolution while preserving the selected aspect ratio.

    ``Megapixels`` targets total pixel area. ``Max Pixels`` makes the longest
    edge as large as possible without exceeding ``max_pixels``. ``Shorter
    Size`` targets the shorter edge. All outputs are aligned to ``multiple``
    for compatibility with latent-based models.
    """
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"Unknown aspect ratio: {aspect_ratio}")

    ratio_width, ratio_height = ASPECT_RATIOS[aspect_ratio]
    multiple = max(1, int(multiple))

    if size_mode == "Megapixels":
        target_area = max(0.01, float(megapixels)) * 1_000_000
        raw_width = sqrt(target_area * ratio_width / ratio_height)
        raw_height = sqrt(target_area * ratio_height / ratio_width)
        width = _round_to_multiple(raw_width, multiple)
        height = _round_to_multiple(raw_height, multiple)
    elif size_mode == "Max Pixels":
        longest_edge = _floor_to_multiple(max(multiple, int(max_pixels)), multiple)

        if ratio_width >= ratio_height:
            width = longest_edge
            height = _round_to_multiple(
                longest_edge * ratio_height / ratio_width, multiple
            )
        else:
            height = longest_edge
            width = _round_to_multiple(
                longest_edge * ratio_width / ratio_height, multiple
            )
    elif size_mode in ("Shorter Size", "shorter_size"):
        shortest_edge = _round_to_multiple(
            max(multiple, int(shorter_pixels)), multiple
        )

        if ratio_width >= ratio_height:
            height = shortest_edge
            width = _round_to_multiple(
                shortest_edge * ratio_width / ratio_height, multiple
            )
        else:
            width = shortest_edge
            height = _round_to_multiple(
                shortest_edge * ratio_height / ratio_width, multiple
            )
    else:
        raise ValueError(f"Unknown size mode: {size_mode}")

    return width, height


class VideoResolutionSelector:
    """Return width and height from common aspect-ratio presets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (
                    list(ASPECT_RATIOS),
                    {"default": "9:16 (Portrait Widescreen)"},
                ),
                "size_mode": (
                    ["Megapixels", "Max Pixels", "Shorter Size"],
                    {"default": "Megapixels"},
                ),
                "megapixels": (MEGAPIXEL_PRESETS, {"default": "1.0"}),
                "max_pixels": (
                    "INT",
                    {"default": 1920, "min": 64, "max": 16384, "step": 64},
                ),
                "shorter_pixels": (
                    "INT",
                    {"default": 768, "min": 64, "max": 16384, "step": 64},
                ),
                "multiple": (MULTIPLE_PRESETS, {"default": "64"}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "select_resolution"
    CATEGORY = "Omario/Utilities"

    def select_resolution(
        self,
        aspect_ratio,
        size_mode,
        megapixels,
        max_pixels,
        shorter_pixels,
        multiple,
    ):
        width, height = calculate_resolution(
            aspect_ratio,
            size_mode,
            megapixels,
            max_pixels,
            multiple,
            shorter_pixels,
        )
        return (width, height)


NODE_CLASS_MAPPINGS = {"VideoResolutionSelector": VideoResolutionSelector}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoResolutionSelector": "Video Resolution Selector"
}
