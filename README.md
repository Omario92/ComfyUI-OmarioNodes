# ComfyUI-OmarioNodes

Custom nodes for ComfyUI by [Omario92](https://github.com/Omario92).

## Nodes

### Image & Transition

| Node | Display name | Description |
| --- | --- | --- |
| `LightLeaksTransition` | Light Leaks Transition (like CrossFadeImages) | Light leak transition effect similar to CrossFadeImages. |
| `SaveImagePlus` | Save Image Plus | Save images with subfolder, timestamp, metadata, and extension controls. |

### Mask & Crop

| Node | Display name | Description |
| --- | --- | --- |
| `MaskClampedCrop` | Mask Tracking Crop (Clamped) | Smart mask crop with clamping and motion tracking. |
| `MaskClampedCropSticky` | Mask Tracking Crop (Sticky) | Sticky variant of Mask Tracking Crop. |

### Scheduler & Blend

| Node | Display name | Description |
| --- | --- | --- |
| `DualEndpointColorBlendScheduler` | Dual Endpoint Color Blend (by Frames) | Three-point color blending with a flexible frame timeline. |

### API & Text

| Node | Display name | Description |
| --- | --- | --- |
| `GemmaAPITextEncode` | LTX-2 API Text Encode | Text encode through Gemma API. |

### Cache & Inpaint

| Node | Display name | Description |
| --- | --- | --- |
| `SaveInpaintCropCache` | Save Inpaint Crop Cache | Save inpaint crop cache safely. |
| `LoadInpaintCropCache` | Load Inpaint Crop Cache | Load versioned inpaint crop cache. |

### Conditioning

| Node | Display name | Description |
| --- | --- | --- |
| `SaveConditioning` | Save Conditioning | Save `CONDITIONING` to disk. |
| `LoadConditioning` | Load Conditioning | Load saved `CONDITIONING` from a dropdown. |

## Conditioning Usage

### Save Conditioning

- Input: `CONDITIONING`
- Input: `filename`, for example `character_pose1`
- Input: `overwrite`
- Output: none

The file is saved under `ComfyUI/models/conditions/`. The folder is created automatically.

### Load Conditioning

- Input: dropdown listing `.pt`, `.ckpt`, and `.safetensors` files from `ComfyUI/models/conditions/`
- Output: `CONDITIONING`

Restart ComfyUI after updating the custom node folder.
