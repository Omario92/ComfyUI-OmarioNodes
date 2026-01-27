import torch


class MaskClampedCropSticky:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "crop_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "crop_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            },
            "optional": {
                "optional_image_to_crop": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("BOX", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("crop_box", "cropped_image", "cropped_mask", "bbox_mask_full")
    FUNCTION = "calculate_and_crop_sticky"
    CATEGORY = "BT Studio/Video"

    def calculate_and_crop_sticky(self, mask, crop_width, crop_height, optional_image_to_crop=None):
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

        # --- LOGIC GIỮ VỊ TRÍ CŨ (STICKY) ---
        last_center_x = W // 2
        last_center_y = H // 2

        for i in range(B):
            current_mask = mask[i]
            coords = torch.nonzero(current_mask > 0.1)

            if len(coords) == 0:
                # Không có mask -> Dùng lại vị trí cũ
                center_y = last_center_y
                center_x = last_center_x
            else:
                # Có mask -> Tính toán và cập nhật vị trí mới
                min_y = torch.min(coords[:, 0]).item()
                max_y = torch.max(coords[:, 0]).item()
                min_x = torch.min(coords[:, 1]).item()
                max_x = torch.max(coords[:, 1]).item()

                center_y = int((min_y + max_y) / 2)
                center_x = int((min_x + max_x) / 2)

                last_center_y = center_y
                last_center_x = center_x

            # --- TÍNH TOÁN CROP (CLAMPED) ---
            top = center_y - (target_h // 2)
            left = center_x - (target_w // 2)

            if left + target_w > W:
                left = W - target_w
            if top + target_h > H:
                top = H - target_h
            left = max(0, left)
            top = max(0, top)

            box = (left, top, target_w, target_h)
            boxes.append(box)

            crop_m = current_mask[top:top + target_h, left:left + target_w]
            cropped_masks.append(crop_m)

            if optional_image_to_crop is not None:
                crop_i = optional_image_to_crop[i, top:top + target_h, left:left + target_w, :]
                cropped_images.append(crop_i)

            bbox_mask_frame = torch.zeros((H, W), dtype=torch.float32, device=mask.device)
            bbox_mask_frame[top:top + target_h, left:left + target_w] = 1.0
            full_bbox_masks.append(bbox_mask_frame)

        out_masks = torch.stack(cropped_masks, dim=0)
        out_bbox_full = torch.stack(full_bbox_masks, dim=0)

        if optional_image_to_crop is not None:
            out_images = torch.stack(cropped_images, dim=0)
        else:
            out_images = torch.zeros((B, target_h, target_w, 3))

        return (boxes, out_images, out_masks, out_bbox_full)
