# __init__.py (root của repo ComfyUI-OmarioNodes)
# Gom & đăng ký các node từ ./nodes/*

import torch

from .blend_scheduler import DualEndpointColorBlendScheduler
from .gemma_api_text_encode import GemmaAPITextEncode
from .mask_clamped_crop import MaskClampedCrop
from .mask_clamped_crop_sticky import MaskClampedCropSticky

class MaskTrackingCropAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "crop_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "crop_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "mode_control": (["tracking_raw", "tracking_smooth", "stationary_center"],),
                "smooth_strength": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 0.99, "step": 0.05}),
            },
            "optional": {
                "optional_image_to_crop": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("BOX", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("crop_box", "cropped_image", "cropped_mask", "bbox_mask_full")
    FUNCTION = "calculate_and_crop_advanced"
    CATEGORY = "BT Studio/Video"

    def calculate_and_crop_advanced(self, mask, crop_width, crop_height, mode_control, smooth_strength, optional_image_to_crop=None):
        if len(mask.shape) == 3:
            B, H, W = mask.shape
        else:
            B, H, W, C = mask.shape

        target_w = min(crop_width, W)
        target_h = min(crop_height, H)

        boxes = []
        cropped_masks = []
        cropped_images = []
        full_bbox_masks = []

        # --- BƯỚC 1: QUÉT TRƯỚC VỊ TRÍ MASK TOÀN BỘ VIDEO ---
        all_centers = []

        # Mặc định center nếu không tìm thấy mask
        default_x, default_y = W // 2, H // 2

        for i in range(B):
            current_mask = mask[i]
            coords = torch.nonzero(current_mask > 0.1)
            if len(coords) > 0:
                min_y = torch.min(coords[:, 0]).item()
                max_y = torch.max(coords[:, 0]).item()
                min_x = torch.min(coords[:, 1]).item()
                max_x = torch.max(coords[:, 1]).item()
                cy = int((min_y + max_y) / 2)
                cx = int((min_x + max_x) / 2)
                all_centers.append((cx, cy))
            else:
                # Nếu frame này mất mask, tạm thời đánh dấu là None
                all_centers.append(None)

        # --- BƯỚC 2: XỬ LÝ LOGIC THEO MODE ---
        final_coords_per_frame = [] # Danh sách (center_x, center_y) cho từng frame

        if mode_control == "stationary_center":
            # Tính trung bình cộng của TẤT CẢ các frame có mask
            valid_cx = [c[0] for c in all_centers if c is not None]
            valid_cy = [c[1] for c in all_centers if c is not None]

            if valid_cx:
                avg_x = int(sum(valid_cx) / len(valid_cx))
                avg_y = int(sum(valid_cy) / len(valid_cy))
            else:
                avg_x, avg_y = default_x, default_y

            # Gán tọa độ cố định này cho toàn bộ frame
            final_coords_per_frame = [(avg_x, avg_y)] * B

        elif mode_control == "tracking_smooth":
            # Tracking có làm mượt (Interpolation)
            current_x, current_y = default_x, default_y

            # Tìm điểm khởi đầu hợp lệ đầu tiên
            for c in all_centers:
                if c is not None:
                    current_x, current_y = c
                    break

            for i in range(B):
                target = all_centers[i]
                if target is not None:
                    # Công thức làm mượt: Pos_Mới = (1-alpha)*Target + alpha*Pos_Cũ
                    # smooth_strength càng cao (0.9) thì càng ì, ít rung
                    alpha = smooth_strength
                    current_x = (1 - alpha) * target[0] + alpha * current_x
                    current_y = (1 - alpha) * target[1] + alpha * current_y

                final_coords_per_frame.append((int(current_x), int(current_y)))

        else: # "tracking_raw"
            # Giống logic cũ: Frame nào tính frame đó, mất mask thì giữ lại vị trí cũ
            last_x, last_y = default_x, default_y
            for i in range(B):
                if all_centers[i] is not None:
                    last_x, last_y = all_centers[i]
                final_coords_per_frame.append((last_x, last_y))

        # --- BƯỚC 3: CROP VÀ RENDER OUTPUT ---
        for i in range(B):
            center_x, center_y = final_coords_per_frame[i]

            # Tính toán Clamped Box
            top = center_y - (target_h // 2)
            left = center_x - (target_w // 2)

            if left + target_w > W: left = W - target_w
            if top + target_h > H: top = H - target_h
            left = max(0, left)
            top = max(0, top)

            # Output 1: Box
            box = (left, top, target_w, target_h)
            boxes.append(box)

            # Output 2: Cropped Mask
            crop_m = mask[i][top:top+target_h, left:left+target_w]
            cropped_masks.append(crop_m)

            # Output 3: Cropped Image
            if optional_image_to_crop is not None:
                crop_i = optional_image_to_crop[i, top:top+target_h, left:left+target_w, :]
                cropped_images.append(crop_i)

            # Output 4: Full BBOX Mask
            bbox_mask_frame = torch.zeros((H, W), dtype=torch.float32, device=mask.device)
            bbox_mask_frame[top:top+target_h, left:left+target_w] = 1.0
            full_bbox_masks.append(bbox_mask_frame)

        out_masks = torch.stack(cropped_masks, dim=0)
        out_bbox_full = torch.stack(full_bbox_masks, dim=0)

        if optional_image_to_crop is not None:
            out_images = torch.stack(cropped_images, dim=0)
        else:
            out_images = torch.zeros((B, target_h, target_w, 3))

        return (boxes, out_images, out_masks, out_bbox_full)

NODE_CLASS_MAPPINGS = {
    "DualEndpointColorBlendScheduler": DualEndpointColorBlendScheduler,
    "GemmaAPITextEncode": GemmaAPITextEncode,
    "MaskClampedCrop": MaskClampedCrop,
    "MaskClampedCropSticky": MaskClampedCropSticky,
    "MaskTrackingCropAdvanced": MaskTrackingCropAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualEndpointColorBlendScheduler": "Dual Endpoint Color Blend (by Frames)",
    "GemmaAPITextEncode": "LTX-2 API Text Encode",
    "MaskClampedCrop": "Mask Tracking Crop (Clamped)",
    "MaskClampedCropSticky": "Mask Tracking Crop (Sticky)",
    "MaskTrackingCropAdvanced": "Mask Tracking Crop (Advanced)",
}
