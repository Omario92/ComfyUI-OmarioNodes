import math

import torch


# ================== EASING FUNCTIONS (copy từ KJNodes) ==================
def ease_in(t):
    return t * t


def ease_out(t):
    return 1 - (1 - t) * (1 - t)


def ease_in_out(t):
    return 3 * t * t - 2 * t * t * t


def bounce(t):
    if t < 0.5:
        return ease_out(t * 2) * 0.5
    return ease_in((t - 0.5) * 2) * 0.5 + 0.5


easing_functions = {
    "linear": lambda t: t,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "bounce": bounce,
}


# ================== BLEND MODES ==================
def screen_blend(base, overlay):
    """Screen blend - chuẩn cho light leaks."""
    return 1 - (1 - base) * (1 - overlay)


def add_blend(base, overlay, intensity=1.0):
    return torch.clamp(base + overlay * intensity, 0.0, 1.0)


# ================== MAIN NODE ==================
class LightLeaksTransition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_1": ("IMAGE",),
                "images_2": ("IMAGE",),
                "light_leaks": ("IMAGE",),  # Có thể là 1 ảnh hoặc nhiều ảnh
                "transition_start_index": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "transitioning_frames": ("INT", {"default": 30, "min": 1, "max": 500, "step": 1}),
                "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce"], {"default": "ease_in_out"}),
                "start_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "light_intensity": ("FLOAT", {"default": 1.8, "min": 0.0, "max": 5.0, "step": 0.05}),
                "light_curve": (["peak_middle", "constant", "fade_in_out"], {"default": "peak_middle"}),
                "blend_mode": (["screen", "add"], {"default": "screen"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_transition"
    CATEGORY = "KJNodes/image"

    def do_transition(
        self,
        images_1,
        images_2,
        light_leaks,
        transition_start_index,
        transitioning_frames,
        interpolation,
        start_level,
        end_level,
        light_intensity,
        light_curve,
        blend_mode,
    ):
        # Xử lý index âm (giống KJNodes)
        if transition_start_index < 0:
            transition_start_index = len(images_1) + transition_start_index

        transition_start_index = max(0, transition_start_index)
        transitioning_frames = min(
            transitioning_frames,
            len(images_1) - transition_start_index,
            len(images_2),
        )

        if transitioning_frames <= 0:
            # Không có transition → trả về images_1 + images_2 còn lại
            return (torch.cat([images_1, images_2], dim=0),)

        # Tạo alpha cho crossfade
        alphas = torch.linspace(start_level, end_level, transitioning_frames, device=images_1.device)
        easing = easing_functions.get(interpolation, lambda t: t)

        transition_images = []

        # Chuẩn bị light leak (lặp lại nếu cần)
        leak_count = light_leaks.shape[0]

        for i in range(transitioning_frames):
            # Crossfade giữa 2 ảnh
            alpha = easing(alphas[i])
            img1 = images_1[transition_start_index + i]
            img2 = images_2[i]
            cross = (1 - alpha) * img1 + alpha * img2

            # Áp dụng Light Leak (cycle theo batch light_leaks)
            leak = light_leaks[i % leak_count]

            # Tính opacity theo curve
            if transitioning_frames <= 1:
                curve_pos = 1.0
            else:
                curve_pos = i / (transitioning_frames - 1)

            if light_curve == "peak_middle":
                opacity = math.sin(math.pi * curve_pos) * light_intensity
            elif light_curve == "fade_in_out":
                opacity = math.sin(math.pi * curve_pos) * 0.8 * light_intensity
            else:  # constant
                opacity = light_intensity

            # Blend
            if blend_mode == "screen":
                screened = screen_blend(cross, leak)
                leaked = torch.lerp(cross, screened, torch.tensor(opacity, device=cross.device, dtype=cross.dtype).clamp(0.0, 1.0))
            else:
                leaked = add_blend(cross, leak, opacity)

            transition_images.append(leaked)

        transition_tensor = torch.stack(transition_images, dim=0)

        # Ghép lại giống CrossFadeImages
        beginning = images_1[:transition_start_index]
        remaining = images_2[transitioning_frames:]

        result = torch.cat([beginning, transition_tensor], dim=0)
        if len(remaining) > 0:
            result = torch.cat([result, remaining], dim=0)

        return (result,)
