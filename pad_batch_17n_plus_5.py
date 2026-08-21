import math

import torch


class PadBatchTo17nPlus5:
    """Pad an image batch by repeating its last frame until its size is 17n+5."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("padded_images", "original_count", "total_count")
    FUNCTION = "pad_batch"
    CATEGORY = "Omario/video"

    def pad_batch(self, images):
        original_count = len(images)
        if original_count == 0:
            raise ValueError("Pad Batch to 17n+5 requires at least one input frame.")

        # Find the smallest non-negative n whose 17n+5 frame count can contain
        # the input batch. For counts below 5, n=0 gives the first valid size.
        n = max(0, math.ceil((original_count - 5) / 17))
        total_count = 17 * n + 5
        padding_count = total_count - original_count

        if padding_count == 0:
            return (images, original_count, total_count)

        repeat_shape = (padding_count,) + (1,) * (images.ndim - 1)
        padding = images[-1:].repeat(repeat_shape)
        padded_images = torch.cat((images, padding), dim=0)

        return (padded_images, original_count, total_count)
