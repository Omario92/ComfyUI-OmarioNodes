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

    RETURN_TYPES = ("BOX", "IMAGE", "MASK")
    RETURN_NAMES = ("crop_box", "cropped_image", "cropped_mask")
    FUNCTION = "calculate_and_crop"
    CATEGORY = "BT Studio/Video"

    def calculate_and_crop(self, mask, crop_width, crop_height, optional_image_to_crop=None):
        # Lấy kích thước gốc từ mask (Batch, Height, Width)
        # Lưu ý: Mask trong ComfyUI thường là (B, H, W) hoặc (B, H, W, 1)
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

        # Xử lý từng frame
        for i in range(B):
            current_mask = mask[i] # (H, W)
            
            # Tìm tọa độ các điểm màu trắng (giá trị > 0)
            # torch.nonzero trả về (y, x)
            coords = torch.nonzero(current_mask > 0.1)
            
            if len(coords) == 0:
                # Nếu frame đen thui (không có mask), giữ vị trí ở giữa hoặc vị trí cũ
                # Ở đây mình set mặc định là giữa khung hình
                center_y, center_x = H // 2, W // 2
            else:
                # Tính trọng tâm của vùng mask (bounding box bao quanh mask)
                min_y = torch.min(coords[:, 0]).item()
                max_y = torch.max(coords[:, 0]).item()
                min_x = torch.min(coords[:, 1]).item()
                max_x = torch.max(coords[:, 1]).item()
                
                # Tìm điểm giữa của mask hiện tại
                center_y = int((min_y + max_y) / 2)
                center_x = int((min_x + max_x) / 2)

            # Tính toán góc trên-trái (Top-Left) dự kiến
            top = center_y - (target_h // 2)
            left = center_x - (target_w // 2)

            # --- LOGIC CLAMPING (GIỮ KHUNG HÌNH KHÔNG OUT VIỀN) ---
            
            # 1. Xử lý cạnh phải và dưới:
            # Nếu (left + target_w) > W thì đẩy left lùi lại sao cho mép phải vừa chạm biên
            if left + target_w > W:
                left = W - target_w
            
            if top + target_h > H:
                top = H - target_h
                
            # 2. Xử lý cạnh trái và trên (ưu tiên cao hơn để tránh số âm):
            # Nếu left < 0 thì set về 0
            left = max(0, left)
            top = max(0, top)
            
            # Tạo Box tuple (x, y, w, h) theo chuẩn ComfyUI
            box = (left, top, target_w, target_h)
            boxes.append(box)

            # Crop Mask để output
            # Cắt tensor: [top:bottom, left:right]
            crop_m = current_mask[top:top+target_h, left:left+target_w]
            cropped_masks.append(crop_m)

            # Crop Image nếu có input
            if optional_image_to_crop is not None:
                # Image shape là (B, H, W, C)
                crop_i = optional_image_to_crop[i, top:top+target_h, left:left+target_w, :]
                cropped_images.append(crop_i)

        # Stack lại thành batch
        out_masks = torch.stack(cropped_masks, dim=0)
        
        if optional_image_to_crop is not None:
            out_images = torch.stack(cropped_images, dim=0)
        else:
            # Nếu không có ảnh đầu vào, trả về tensor rỗng hoặc bản sao mask (để tránh lỗi node)
            out_images = torch.zeros((B, target_h, target_w, 3))

        return (boxes, out_images, out_masks)

# Đăng ký node với ComfyUI
NODE_CLASS_MAPPINGS = {
    "MaskClampedCrop": MaskClampedCrop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskClampedCrop": "Mask Tracking Crop (Clamped)"
}
