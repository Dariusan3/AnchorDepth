# Thesis — Related Work Chapter
## Chapter 2: Related Work

---

### 2.1 Supervised Monocular Depth Estimation

Early work on monocular depth estimation used hand-crafted features and probabilistic graphical models (Saxena et al., 2008). The advent of deep learning transformed the field: Eigen et al. (2014) were the first to demonstrate that a convolutional neural network could directly predict per-pixel depth from a single image, establishing the encoder-decoder architecture that has remained standard ever since.

Subsequent work improved accuracy through better architectures, loss functions, and multi-scale prediction. Laina et al. (2016) introduced fully convolutional residual networks for depth, while Fu et al. (2018) reframed depth estimation as an ordinal regression problem. Bhat et al. (2021) proposed AdaBins, decomposing depth into adaptive bins to better handle the full depth range. Lee et al. (2019) introduced BTS (Big-to-Small), which uses local planar guidance layers to produce sharp depth boundaries.

A key weakness of purely supervised approaches is their dependence on large-scale annotated data. Ground truth depth from LiDAR is sparse (the velodyne HDL-64 sensor covers less than 5% of image pixels), requires careful temporal and spatial calibration with the camera, and is expensive to acquire outside of purpose-built data collection campaigns.

---

### 2.2 Self-Supervised Monocular Depth Estimation

Self-supervised monocular depth estimation emerged as a response to the labeling bottleneck. The foundational insight is that a video sequence implicitly encodes depth information: if the scene is static and the camera moves, consecutive frames are related by a projective transformation parameterized by the scene depth and the camera's ego-motion.

**Zhou et al. (2017)** formalized this as a joint learning problem. A depth network and a pose network are trained simultaneously: the depth network predicts per-pixel depth for a target frame, the pose network estimates the relative transformation between the target and a source frame, and a differentiable warping module reconstructs the target view from the source. The reconstruction quality, measured by photometric error, provides the training signal. This framework — often called the "SfMLearner" approach — became the foundation for all subsequent self-supervised depth methods.

**Godard et al. (2019)** — Monodepth2 — introduced three key improvements that remain standard today:

1. *Per-pixel minimum reprojection:* Instead of averaging the photometric error across all source frames, taking the minimum per pixel handles occlusions — a pixel visible in the target but occluded in one source frame will have lower error from the other source frame.

2. *Auto-masking:* Pixels where the photometric error from warping is higher than the identity error (copying the source frame directly) indicate static cameras or objects moving at the same velocity as the camera. These pixels provide no useful training signal and are masked out.

3. *Full-resolution multi-scale prediction:* Computing the loss at multiple decoder scales and upsampling predictions to full resolution before computing the loss avoids texture-copy artifacts and improves boundary sharpness.

Monodepth2 achieved AbsRel of 0.115 on KITTI Eigen, establishing a widely-used baseline.

**Later improvements** to self-supervised depth fall into several categories:

*Better encoders:* Johnston and Carneiro (2020) showed that replacing ResNet encoders with EfficientNet improved results. Guizilini et al. (2020) introduced PackNet, a 3D convolution-based architecture that better preserves spatial information. Zhou et al. (2021) proposed DIFFNet, using HRNet for dense feature extraction (AbsRel 0.102).

*Attention-based encoders:* Zhao et al. (2022) introduced MonoViT, replacing the CNN encoder with a small Vision Transformer (ViT-Small). The improved global context modeling of ViT raised the bar to AbsRel 0.099 — the closest prior work to our approach in terms of using a transformer encoder for self-supervised depth.

*Additional supervision:* Shu et al. (2020) proposed DepthHints, using stereo-derived pseudo-labels to guide monocular training. Watson et al. (2021) showed that using depth completion with sparse LiDAR hints during training (not evaluation) significantly boosts performance. These semi-supervised approaches require additional sensor data and are orthogonal to our contribution.

*Improved loss functions:* Shu et al. (2020) introduced feature-metric loss, comparing encoder features rather than raw pixels. Rares et al. (2020) proposed edge-guided loss functions. Lyu et al. (2021) combined self-supervised training with sparse depth completion.

Our work differs from all of the above in a fundamental way: we do not train a depth network from scratch. Instead, we adapt a large pretrained foundation model using parameter-efficient fine-tuning — a direction unexplored by prior self-supervised depth methods.

---

### 2.3 Depth Estimation Foundation Models

The recent trend toward foundation models has reached depth estimation. These models are trained on large and diverse collections of images, often using a mix of labeled and pseudo-labeled data, and achieve strong zero-shot performance across diverse scenes and camera types.

**MiDaS (Ranftl et al., 2020, 2022)** pioneered the mixing of multiple depth datasets with different annotations into a unified affine-invariant depth representation. By training on a diverse mixture and using scale-and-shift invariant loss functions, MiDaS achieves generalization across indoor, outdoor, and aerial scenes without domain-specific fine-tuning.

**DPT (Ranftl et al., 2021)** replaced the CNN backbone in MiDaS with a Vision Transformer, exploiting the global context modeling of ViT for dense prediction tasks. DPT demonstrated that ViTs, originally developed for image classification, transfer effectively to depth estimation when combined with appropriate decoder designs.

**ZoeDepth (Bhat et al., 2023)** builds on MiDaS/DPT with metric-scale heads for specific domains, achieving zero-shot metric depth estimation on novel scenes.

**Depth Anything (Yang et al., 2024)** scaled up the training data significantly, using 62 million labeled images alongside unlabeled data and a teacher-student framework for semi-supervised pre-training. The resulting model achieves state-of-the-art zero-shot performance across benchmarks. Depth Anything V2 further improved results with higher-quality synthetic data.

**Depth Pro (Bochkovskii et al., 2025)** — the model we build upon — takes a different approach. Rather than maximizing training data volume, Depth Pro focuses on architectural design for sharp depth boundaries and metric scale estimation without camera calibration. Its multi-resolution patch processing strategy, shared ViT encoders, and separate FOV head for focal length prediction produce metric depth maps with sharp object boundaries and strong zero-shot generalization. Depth Pro is the most capable publicly available metric depth foundation model at the time of this writing.

None of these foundation models have been adapted to self-supervised training. They all require ground truth labels. Our work is the first to adapt Depth Pro to self-supervised photometric training.

---

### 2.4 Parameter-Efficient Fine-Tuning

As pretrained models grew larger, the cost of full fine-tuning became prohibitive. Parameter-efficient fine-tuning (PEFT) methods adapt large pretrained models by updating only a small subset of parameters, leaving the pretrained weights largely intact.

**Adapter layers (Houlsby et al., 2019)** insert small bottleneck MLP modules between transformer blocks. Only the adapter parameters are updated during fine-tuning. Applied to BERT, adapters achieved near full fine-tuning performance with less than 4% of the parameters.

**Prefix tuning (Li and Liang, 2021)** and **prompt tuning (Lester et al., 2021)** prepend trainable "soft prompts" to the input sequence, keeping all model weights frozen. These approaches work well for NLP tasks but have seen limited application in vision.

**Visual Prompt Tuning (VPT, Jia et al., 2022)** adapts these ideas to vision transformers, prepending trainable prompt tokens to the image patch sequence. VPT outperforms linear probing and full fine-tuning on several downstream vision tasks with significantly fewer trainable parameters.

**LoRA (Low-Rank Adaptation, Hu et al., 2022)** is arguably the most impactful PEFT method. For each attention weight matrix W ∈ R^{d×k}, LoRA injects a trainable low-rank decomposition ΔW = BA where B ∈ R^{d×r} and A ∈ R^{r×k} with r ≪ min(d, k). The pretrained weights remain frozen; only A and B are updated. LoRA was originally demonstrated on GPT-3 for NLP tasks but has since been widely adopted for vision transformers, image generation models (LoRA for Stable Diffusion), and multimodal models.

Our work applies LoRA to Depth Pro's dual DINOv2-Large ViT encoders (rank=8, 96 attention layers), reducing the encoder adaptation parameter count from 608M to 2.36M. To our knowledge, this is the first application of LoRA to a depth estimation foundation model.

---

### 2.5 Self-Supervised Learning with ViT Encoders

Self-supervised depth methods using Vision Transformers are a recent development. MonoViT (Zhao et al., 2022) was the first to demonstrate that ViTs trained on ImageNet can serve as effective encoders for self-supervised depth, outperforming CNN-based approaches. However, MonoViT uses a ViT-Small (~22M parameters), far smaller than the DINOv2-Large (304M) used in our backbone.

**DINOv2 (Oquab et al., 2024)** produced ViT representations pretrained with self-supervised learning on 142M images. The resulting features are remarkably strong for dense prediction tasks including depth, surface normals, and semantic segmentation — without any task-specific fine-tuning. Depth Pro exploits these representations by using DINOv2-Large as its encoder, then adding a task-specific decoder trained with metric depth supervision.

Our approach inherits these powerful representations and adapts them to self-supervised photometric training, combining the best of both worlds: the rich visual features of DINOv2 (via Depth Pro) and the label-free training paradigm of Monodepth2.

---

### 2.6 Multi-View Foundation Models for Camera Pose

Classical self-supervised depth methods rely on a small **PoseNet** — typically a ResNet-18 with a six-channel input (the concatenated target and source frames) — trained jointly with the depth network. Because PoseNet is initialized randomly, the early training epochs produce noisy pose estimates that inject error into the photometric loss. The community has long suspected (Wang et al., CVPR 2018) that better pose estimates would meaningfully improve self-supervised depth, but until recently no large pretrained pose estimator existed.

The arrival of **multi-view geometric foundation models** in 2024–2025 changes this. These models are trained on heterogeneous multi-view datasets to jointly predict camera intrinsics, extrinsics, depth maps, and 3D point coordinates.

**DUSt3R (Wang et al., CVPR 2024)** was the first foundation model in this category. It takes a pair of images and predicts dense pixel-to-pixel correspondence along with per-image point maps in a shared coordinate system, from which camera extrinsics can be recovered by least-squares alignment. DUSt3R is fast (one forward pass) and demonstrated strong zero-shot performance on stereo and structure-from-motion benchmarks. **MASt3R (Leroy et al., 2024)** extended DUSt3R with metric depth and improved scalability to dozens of views.

**VGGT — Visual Geometry Grounded Transformer (Wang et al., CVPR 2025 Best Paper)** is the most recent and most capable model in this family. With 1.2 billion parameters, VGGT takes from one to several hundred input images and jointly predicts:
- Camera extrinsics (3D translation + quaternion rotation) per view
- Camera intrinsics (field-of-view angles, principal point assumed at image center)
- Per-image depth maps in metric scale
- Dense 3D point clouds in a unified coordinate system
- 2D point tracks across all input views

The model outputs a 9-dimensional pose encoding per camera that can be converted to standard 3×4 extrinsic matrices via the utility function `pose_encoding_to_extri_intri()`. Operating in OpenCV convention (x-right, y-down, z-forward), VGGT produces world-to-camera transformations from which relative pose between any two frames is recovered by a single matrix product.

VGGT's relevance to this thesis is direct: it can **substitute entirely for the trained PoseNet**, providing high-quality relative poses for every pair of KITTI frames at no training cost. The challenge is memory — VGGT requires roughly 5 GB of VRAM for a three-frame inference, which combined with Depth Pro (10–11 GB during training) exceeds our 12 GB budget. We resolve this with an **offline pose precomputation pipeline** (Section 3.6, run v16): execute VGGT once over all 6,635 KITTI training triplets (about 1.5 hours on the same RTX 4070 Ti), serialize the resulting 4×4 relative transformations to disk (approximately 5 MB total), and load them as a cached dataset during Depth Pro fine-tuning. The trainable PoseNet is bypassed entirely.

To our knowledge, this thesis is the first work to combine two foundation models from different domains — depth estimation (Depth Pro, 952M parameters) and multi-view geometry (VGGT, 1.2B parameters) — at training time on a single consumer GPU. The combination is enabled by the asymmetry of inference: VGGT poses are needed only as inputs and require no gradients, so they can be precomputed once and reused, while the trainable surface (LoRA in Depth Pro) remains small enough to fit alongside Depth Pro inference.

---

### 2.7 Summary and Positioning

The table below positions our work relative to the most relevant prior methods:

| Method | Encoder | Trainable params | Pose source | Labels needed | Consumer GPU |
|--------|---------|------------------|-------------|---------------|--------------|
| Monodepth2 (ICCV'19) | ResNet-18 (11M) | 14.8M | Trained PoseNet | None | Yes |
| PackNet-SfM (CVPR'20) | PackNet (128M) | 128M | Trained PoseNet | None | Yes |
| DIFFNet (ECCV'22) | HRNet-18 (65M) | 65M | Trained PoseNet | None | Yes |
| MonoViT (3DV'22) | ViT-Small (22M) | 22M | Trained PoseNet | None | Yes |
| Depth Anything (CVPR'24) | ViT-Large (307M) | 307M | — | 62M labeled | No |
| Depth Pro (2024) | ViT-Large ×2 (952M) | 952M | — | Labeled | No |
| **Ours — v15** | **ViT-L ×2 + LoRA (952M+2.36M)** | **34M (3.6%)** | Trained PoseNet | **None** | **Yes (12 GB)** |
| **Ours — v16** | **ViT-L ×2 + LoRA (952M+2.36M)** | **22.4M (2.4%)** | **VGGT (1.2B, offline)** | **None** | **Yes (12 GB)** |

Our method occupies a unique position in this landscape:

1. **First to adapt a depth foundation model via self-supervision.** All prior self-supervised depth methods (Monodepth2 through MonoViT) train from scratch or from ImageNet initialization on small networks. All prior depth foundation models (Depth Anything, Depth Pro) require labeled ground truth. We are the first to combine the two paradigms.

2. **First to combine two foundation models from different domains at training time on consumer hardware.** Run v16 simultaneously leverages Depth Pro (depth foundation model, 952M parameters) for the depth network and VGGT (multi-view foundation model, 1.2 B parameters) for ego-motion via offline precomputation — a combination that exceeds the 12 GB VRAM budget when both run jointly, but fits when VGGT is amortized once over the training set.

3. **First to demonstrate that consistency-anchored adaptation extracts a measurable improvement over zero-shot Depth Pro on KITTI.** Our run v15 improves δ<1.25³ from 0.98494 (zero-shot) to 0.98499 with self-supervised signal only; run v16 improves it further to 0.98500 using VGGT-supplied poses. Chapter 4 reports the full ablation table including the negative results that establish why naïve photometric supervision alone is insufficient.

The key technical enablers — LoRA for compact parameter updates, bfloat16 mixed precision with NaN-safe checkpointing for numerical stability, offline VGGT pose precomputation for memory feasibility, and the consistency loss for objective–metric alignment — are detailed in the next chapter.
