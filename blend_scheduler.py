# nodes/blend_scheduler.py
# DualEndpointColorBlendScheduler:
# - KHÔNG chỉnh màu. Chỉ blend giữa: base_images, start_corrected, end_corrected.
# - Lịch blend dựa trên tổng số frames (timeline) và tham số do user nhập.
# - Hỗ trợ "durations" (thời lượng) hoặc "indices" (chỉ số frame trực tiếp).
# - Giữ blend_curve: linear | smoothstep.
# - Tự xử lý B != total_frames (map theo tỉ lệ), và broadcast nếu corrected có B=1.

import torch

def _smoothstep(t):
    # 3t^2 - 2t^3
    return t * t * (3.0 - 2.0 * t)

def _clamp_int(x, lo, hi):
    return max(lo, min(hi, x))

def _weights_from_indices(i, N, s0, s1, e0, e1, curve="linear"):
    """
    Tính trọng số blend cho frame index i (0..N-1) với mốc INCLUSIVE:
      Start:
        - full khi i < s0
        - fade khi s0 <= i <= s1 (giảm 1 -> 0)
        - zero khi i > s1
      End:
        - zero khi i < e0
        - fade khi e0 <= i <= e1 (tăng 0 -> 1)
        - full khi i > e1
    """
    # Clamp & sort biên
    s0, s1 = sorted([_clamp_int(s0, 0, N-1), _clamp_int(s1, 0, N-1)])
    e0, e1 = sorted([_clamp_int(e0, 0, N-1), _clamp_int(e1, 0, N-1)])

    # Start weight
    if i < s0:
        ws = 1.0
    elif i > s1:
        ws = 0.0
    else:
        # i in [s0..s1]
        denom = max(1.0, float(s1 - s0))
        t = (s1 - float(i)) / denom  # 1 -> 0
        ws = float(_smoothstep(t) if curve == "smoothstep" else t)

    # End weight
    if i < e0:
        we = 0.0
    elif i > e1:
        we = 1.0
    else:
        # i in [e0..e1]
        denom = max(1.0, float(e1 - e0))
        t = (float(i) - e0) / denom  # 0 -> 1
        we = float(_smoothstep(t) if curve == "smoothstep" else t)

    return ws, we

def _map_i_from_B_to_N(i, B, N):
    """
    Map frame index i (0..B-1) -> j (0..N-1) theo tỉ lệ (round).
    """
    if B <= 1 or N <= 1:
        return 0
    return int(round(i * (N - 1) / (B - 1)))

def _get_frame(tensor, idx):
    """
    Lấy frame idx từ tensor [B,H,W,3]. Nếu B==1 thì broadcast.
    Nếu idx vượt biên, clamp về frame cuối.
    """
    if tensor.dim() != 4 or tensor.shape[-1] != 3:
        raise ValueError("IMAGE tensor phải có dạng [B,H,W,3] với kênh cuối = 3 (RGB).")
    B = tensor.shape[0]
    if B == 1:
        return tensor[0]
    idx = _clamp_int(idx, 0, B - 1)
    return tensor[idx]

class DualEndpointColorBlendScheduler:
    """
    Blend giữa base_images, start_corrected, end_corrected theo lịch dựa trên 'tổng số frames' & cấu hình người dùng.
    - Không chỉnh màu; chỉ trộn 3 nhánh theo timeline.
    - Khi ws and we > 0: corrected_mix = (ws * start + we * end) / (ws + we);
      sau đó out = lerp(base, corrected_mix, clamp(ws+we, 0..1)).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_images": ("IMAGE",),
                "start_corrected": ("IMAGE",),
                "end_corrected": ("IMAGE",),

                # Lịch blend
                "total_frames": ("INT", {"default": 81, "min": 1, "max": 100000, "step": 1}),
                "range_mode": (["durations", "indices"], {"default": "durations"}),

                # durations mode (đơn vị: frames)
                "start_full_frames":   ("INT", {"default": 10, "min": 0, "max": 100000, "step": 1}),
                "start_fade_duration": ("INT", {"default": 21, "min": 0, "max": 100000, "step": 1}),
                "end_fade_duration":   ("INT", {"default": 21, "min": 0, "max": 100000, "step": 1}),
                "end_full_frames":     ("INT", {"default": 10, "min": 0, "max": 100000, "step": 1}),

                # indices mode (inclusive)
                "start_fade_from_idx": ("INT", {"default": 10, "min": 0, "max": 100000, "step": 1}),
                "start_fade_to_idx":   ("INT", {"default": 30, "min": 0, "max": 100000, "step": 1}),
                "end_fade_from_idx":   ("INT", {"default": 50, "min": 0, "max": 100000, "step": 1}),
                "end_fade_to_idx":     ("INT", {"default": 70, "min": 0, "max": 100000, "step": 1}),

                "blend_curve": (["linear", "smoothstep"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Video/Color"

    # ----- helper -----
    def _resolve_indices(self, N, mode,
                         start_full_frames, start_fade_duration,
                         end_fade_duration, end_full_frames,
                         s_from, s_to, e_from, e_to):
        """
        Trả về (s0, s1, e0, e1) dùng cho _weights_from_indices(), theo quy ước INCLUSIVE.

        durations mode:
          - Start:
              full: i < s0, với s0 = start_full_frames
              fade: [s0 .. s1], với s1 = s0 + start_fade_duration - 1
          - End:
              full: i > e1, vùng full cuối có độ dài end_full_frames
                    => mốc cuối fade: e1 = N - end_full_frames - 1
              fade: [e0 .. e1], với e0 = e1 - (end_fade_duration - 1)

        ví dụ N=81, end_fade_duration=21, end_full_frames=10:
          e1 = 81 - 10 - 1 = 70
          e0 = 70 - (21 - 1) = 50
          => fade 50..70, full cuối 71..80 (đúng mong muốn)
        """
        if mode == "indices":
            s0, s1 = sorted([_clamp_int(s_from, 0, N-1), _clamp_int(s_to, 0, N-1)])
            e0, e1 = sorted([_clamp_int(e_from, 0, N-1), _clamp_int(e_to, 0, N-1)])
            return s0, s1, e0, e1

        # mode == "durations"
        # --- Start side ---
        s0 = _clamp_int(int(start_full_frames), 0, N-1)
        if start_fade_duration <= 0:
            s1 = s0 - 1  # không có fade
        else:
            s1 = s0 + int(start_fade_duration) - 1

        # --- End side ---
        if end_full_frames <= 0:
            # không có vùng full cuối; fade có thể chạm tới frame cuối
            e1 = N - 1
        else:
            e1 = N - int(end_full_frames) - 1
        if end_fade_duration <= 0:
            e0 = e1 + 1  # không có fade
        else:
            e0 = e1 - (int(end_fade_duration) - 1)

        # Clamp & sort
        s0 = _clamp_int(s0, 0, N-1)
        s1 = _clamp_int(s1, 0, N-1)
        e0 = _clamp_int(e0, 0, N-1)
        e1 = _clamp_int(e1, 0, N-1)
        s0, s1 = sorted([s0, s1])
        e0, e1 = sorted([e0, e1])
        return s0, s1, e0, e1

    # ----- main -----
    def run(self,
            base_images, start_corrected, end_corrected,
            total_frames, range_mode,
            start_full_frames, start_fade_duration, end_fade_duration, end_full_frames,
            start_fade_from_idx, start_fade_to_idx, end_fade_from_idx, end_fade_to_idx,
            blend_curve="linear"):

        # Kiểm tra tensor & kích thước
        for name, t in (("base_images", base_images),
                        ("start_corrected", start_corrected),
                        ("end_corrected", end_corrected)):
            if t.dim() != 4 or t.shape[-1] != 3:
                raise ValueError(f"{name} phải có dạng [B,H,W,3] (RGB trong miền 0..1).")

        if base_images.shape[1:3] != start_corrected.shape[1:3] or \
           base_images.shape[1:3] != end_corrected.shape[1:3]:
            raise ValueError("H và W của 3 luồng ảnh phải giống nhau (không resize trong node).")

        B, H, W, C = base_images.shape
        device = base_images.device
        dtype = base_images.dtype

        # Resolve indices theo mode
        N = int(max(1, total_frames))
        s0, s1, e0, e1 = self._resolve_indices(
            N, range_mode,
            int(max(0, start_full_frames)),
            int(max(0, start_fade_duration)),
            int(max(0, end_fade_duration)),
            int(max(0, end_full_frames)),
            int(max(0, start_fade_from_idx)),
            int(max(0, start_fade_to_idx)),
            int(max(0, end_fade_from_idx)),
            int(max(0, end_fade_to_idx)),
        )

        out_frames = []
        for i in range(B):
            # Map i (0..B-1) -> j (0..N-1) để lấy trọng số theo timeline N
            j = _map_i_from_B_to_N(i, B, N)

            ws, we = _weights_from_indices(j, N, s0, s1, e0, e1, curve=blend_curve)
            ws_t = torch.tensor(ws, device=device, dtype=dtype)
            we_t = torch.tensor(we, device=device, dtype=dtype)

            base = base_images[i]
            sfrm = _get_frame(start_corrected, i)  # broadcast nếu cần
            efrm = _get_frame(end_corrected, i)

            if (ws == 0.0) and (we == 0.0):
                out_frames.append(base)
                continue

            # Trộn giữa start & end corrected theo tỉ lệ tương đối (nếu đều > 0)
            sum_w = ws_t + we_t
            if float(sum_w) <= 0.0:
                corrected_mix = base
                w_total = torch.tensor(0.0, device=device, dtype=dtype)
            else:
                s_share = ws_t / sum_w
                e_share = we_t / sum_w
                corrected_mix = s_share * sfrm + e_share * efrm
                # Tổng trọng số corrected với base:
                w_total = torch.clamp(sum_w, 0.0, 1.0)

            out = (1.0 - w_total) * base + w_total * corrected_mix
            out_frames.append(torch.clamp(out, 0.0, 1.0))

        result = torch.stack(out_frames, dim=0)
        return (result,)
