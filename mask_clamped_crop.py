import torch

class MaskClampedCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),  # Input là Mask (đen trắng)
                "crop_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "crop_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            },
            "optional": {
                "optional_image_to_crop": ("IMAGE",), # Nếu nối Image vào, nó sẽ crop luôn Image
            }
        }

    # Thêm output thứ 4: bbox_mask_full
    RETURN_TYPES = ("BOX", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("crop_box", "cropped_image", "cropped_mask", "bbox_mask_full")
    FUNCTION = "calculate_and_crop"
    CATEGORY = "BT Studio/Video"

    def calculate_and_crop(self, mask, crop_width, crop_height, optional_image_to_crop=None):
        # Lấy kích thước gốc từ mask (Batch, Height, Width)
        if len(mask.shape) == 3:
            B, H, W = mask.shape
        else:
            B, H, W, C = mask.shape
        
        # Đảm bảo crop size không lớn hơn ảnh gốc
        target_w = min(crop_width, W)
        target_h = min(crop_height, H)
        
        boxes = []
        cropped_masks = []
        cropped_images = []
        full_bbox_masks = [] # List chứa các mask BBOX full size

        # Xử lý từng frame
        for i in range(B):
            current_mask = mask[i] # (H, W)
            
            # --- TÌM TRỌNG TÂM ---
            coords = torch.nonzero(current_mask > 0.1)
            
            if len(coords) == 0:
                # Nếu không có mask, lấy tâm giữa
                center_y, center_x = H // 2, W // 2
            else:
                min_y = torch.min(coords[:, 0]).item()
                max_y = torch.max(coords[:, 0]).item()
                min_x = torch.min(coords[:, 1]).item()
                max_x = torch.max(coords[:, 1]).item()
                
                center_y = int((min_y + max_y) / 2)
                center_x = int((min_x + max_x) / 2)

            # --- TÍNH TOÁN VỊ TRÍ ---
            top = center_y - (target_h // 2)
            left = center_x - (target_w // 2)

            # --- LOGIC CLAMPING (GIỮ KHUNG HÌNH) ---
            if left + target_w > W:
                left = W - target_w
            
            if top + target_h > H:
                top = H - target_h
                
            left = max(0, left)
            top = max(0, top)
            
            # 1. Output BOX
            box = (left, top, target_w, target_h)
            boxes.append(box)

            # 2. Output Cropped Mask
            crop_m = current_mask[top:top+target_h, left:left+target_w]
            cropped_masks.append(crop_m)

            # 3. Output Cropped Image
            if optional_image_to_crop is not None:
                crop_i = optional_image_to_crop[i, top:top+target_h, left:left+target_w, :]
                cropped_images.append(crop_i)

            # 4. Output Full Size BBOX Mask (MỚI)
            # Tạo một mask đen (H, W) cùng kích thước video gốc
            # device=mask.device để đảm bảo chạy đúng trên GPU/CPU giống input
            bbox_mask_frame = torch.zeros((H, W), dtype=torch.float32, device=mask.device)
            
            # Vẽ hình chữ nhật trắng tại vị trí đã tính toán
            bbox_mask_frame[top:top+target_h, left:left+target_w] = 1.0
            full_bbox_masks.append(bbox_mask_frame)

        # Stack lại thành batch
        out_masks = torch.stack(cropped_masks, dim=0)
        out_bbox_full = torch.stack(full_bbox_masks, dim=0)
        
        if optional_image_to_crop is not None:
            out_images = torch.stack(cropped_images, dim=0)
        else:
            out_images = torch.zeros((B, target_h, target_w, 3))

        return (boxes, out_images, out_masks, out_bbox_full)

# Đăng ký node
NODE_CLASS_MAPPINGS = {
    "MaskClampedCrop": MaskClampedCrop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskClampedCrop": "Mask Tracking Crop (Clamped)"
}
