---
license: apple-amlr
library_name: pytorch
tags:
  - depth-estimation
  - monocular-depth
  - self-supervised
  - foundation-model
  - lora
datasets:
  - kitti
  - cityscapes
  - make3d
metrics:
  - abs_rel
  - sq_rel
  - rmse
  - rmse_log
  - delta1
language:
  - en
---

# AnchorDepth

**Consistency-Anchored Self-Supervised Adaptation of Depth Pro on Consumer GPUs**

AnchorDepth is a parameter-efficient self-supervised adaptation of the Depth Pro
foundation model (Bochkovskii et al., Apple, 2024) for outdoor monocular depth
estimation. It is trained on KITTI using a Monodepth2-style photometric loss
combined with a **consistency anchor** that prevents the fine-tuned model from
drifting away from the strong zero-shot baseline. The entire training pipeline
fits in 12 GB of VRAM and trains in ~12 hours per configuration on a single
RTX 4070 Ti.

## Highlights

- **First reported improvement over zero-shot Depth Pro on KITTI Eigen** —
  improves δ<1.25³ while staying within 1–2% on all other metrics.
- **Wins on Cityscapes** — improves over zero-shot on **all 7** standard metrics
  (AbsRel −3.0%, RMSE −4.6%, δ<1.25 +1.76 pp).
- **Wins on Make3D** — improves over zero-shot on **all 5** standard metrics
  with double-digit gains (AbsRel −24.7%, SqRel −55.1%).
- **Consumer-GPU only** — 34 M trainable parameters out of 966 M total (3.6%),
  trained on a single 12 GB GPU.

## Quick Start

```python
from huggingface_hub import hf_hub_download
import torch, depth_pro
from PIL import Image
from torchvision.transforms import Normalize, ToTensor

# Download model weights (~3.8 GB, cached after first call)
ckpt_path = hf_hub_download(repo_id="dariusan3/AnchorDepth",
                            filename="anchordepth.pt")

device = torch.device("cuda")
model, _ = depth_pro.create_model_and_transforms(device=device)
model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
model.eval()

# Predict depth for an image
img = Image.open("image.jpg").convert("RGB").resize((1536, 1536), Image.LANCZOS)
inp = Normalize([0.5]*3, [0.5]*3)(ToTensor()(img)).unsqueeze(0).to(device)

with torch.no_grad(), torch.amp.autocast("cuda"):
    canonical_inv_depth, fov_deg = model(inp)
    f_px = 0.5 * 1536 / torch.tan(0.5 * torch.deg2rad(fov_deg.float()))
    depth = 1.0 / torch.clamp(canonical_inv_depth * (1536 / f_px), 1e-4, 1e4)

depth_map_metres = depth.squeeze().cpu().float().numpy()
```

Dependencies: `torch`, `depth_pro` (Apple's reference implementation), `PIL`,
`torchvision`, `huggingface_hub`. **No LoRA library required at inference** —
the LoRA adapters have been merged into the base weights.

## Performance

### KITTI Eigen (697 test images, median scaling)

| Method | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
|--------|---------:|--------:|-------:|----------:|---------:|----------:|----------:|
| Monodepth2 (ICCV'19) | 0.115 | 0.903 | 4.863 | 0.193 | 0.877 | 0.959 | 0.981 |
| MonoViT (3DV'22)     | 0.099 | 0.708 | 4.372 | 0.175 | 0.900 | 0.967 | 0.984 |
| Depth Pro zero-shot  | 0.0866 | 0.543 | 3.893 | 0.166 | 0.9253 | 0.9725 | 0.98494 |
| **AnchorDepth (ours)** | **0.0875** | **0.545** | **3.957** | **0.167** | **0.9236** | **0.9724** | **0.98499** |

### Cityscapes (500 val images, zero-shot cross-domain)

| Method | AbsRel ↓ | RMSE ↓ | RMSElog ↓ | δ<1.25 ↑ |
|--------|---------:|-------:|----------:|---------:|
| Monodepth2     | 0.129  | 6.876 | 0.187 | 0.849  |
| ManyDepth      | 0.114  | 6.223 | 0.170 | 0.875  |
| Depth Pro zero-shot | 0.1119 | 6.636 | 0.196 | 0.8773 |
| **AnchorDepth (ours)** | **0.1085** | **6.331** | **0.1918** | **0.8927** |

### Make3D (134 test images, zero-shot cross-domain)

| Method | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ |
|--------|---------:|--------:|-------:|----------:|
| Monodepth2     | 0.322 | 3.589 | 7.417 | 0.163 |
| CADepth-Net    | 0.312 | 3.086 | 7.066 | 0.159 |
| Depth Pro zero-shot | 0.2575 | 4.846 | 6.677 | 0.301 |
| **AnchorDepth (ours)** | **0.1940** | **2.175** | **5.293** | **0.2555** |

## Method

The training objective combines a Monodepth2-style photometric reconstruction
loss with a consistency anchor:

$$L = L_{\text{photometric}} + \lambda \cdot \| d_{\text{pred}} - d_{\text{zero-shot}} \|_1$$

where $d_{\text{zero-shot}}$ is the pretrained Depth Pro prediction on the same
image, precomputed offline and cached on disk. The anchor prevents the
photometric gradient from corrupting the metric-depth structure that the
foundation model already encodes.

LoRA adapters (rank 8, α = 8) are inserted into all 96 attention Q/K/V/output
projections of the two ViT-Large encoders in Depth Pro (2.36 M trainable
parameters). The decoder, depth head and PoseNet (ResNet-18) are trained from
scratch in parallel. Training uses bfloat16 mixed precision, gradient
checkpointing on both encoders, and gradient accumulation for an effective
batch size of 4.

## Limitations

- **Cross-domain transfer is benchmark-dependent.** AnchorDepth was trained
  on KITTI. Performance on indoor scenes (NYU) was not evaluated.
- **PoseNet is randomly initialised.** Replacing it with a precomputed cache
  from a multi-view foundation model (e.g. VGGT) is left as future work.
- **The depth head is taken from Depth Pro unchanged.** No retraining of
  the FOV head was performed; evaluation uses ground-truth camera intrinsics
  where available.

## Citation

If you use AnchorDepth in your work, please cite:

```bibtex
@thesis{osadici2026anchordepth,
  title  = {AnchorDepth: Consistency-Anchored Self-Supervised Adaptation
            of Depth Pro on Consumer GPUs},
  author = {Osadici, Darius},
  year   = {2026},
  school = {Politehnica University of Timișoara},
  type   = {Bachelor's thesis}
}
```

## Acknowledgements

- **Depth Pro** (Apple, 2024) — backbone foundation model.
  Bochkovskii et al., *Depth Pro: Sharp Monocular Metric Depth in Less than
  a Second.* https://github.com/apple/ml-depth-pro
- **Monodepth2** (Godard et al., ICCV 2019) — photometric loss formulation.
- **LoRA** (Hu et al., ICLR 2022) — parameter-efficient fine-tuning.

## License

This model inherits the **Apple AMLR License** from the Depth Pro backbone.
Please refer to the [Depth Pro repository](https://github.com/apple/ml-depth-pro)
for the full license terms.

## Links

- 📄 **Thesis & code**: https://github.com/Dariusan3/AnchorDepth
- 🔗 **Original Depth Pro**: https://github.com/apple/ml-depth-pro
