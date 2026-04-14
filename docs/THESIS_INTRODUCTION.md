# Thesis — Introduction Chapter
## Chapter 1: Introduction

---

### 1.1 Motivation

Depth perception is a fundamental capability for machines operating in the physical world. Autonomous vehicles must know how far obstacles are before they can avoid them. Robots need spatial understanding to manipulate objects. Augmented reality systems must estimate scene geometry to blend virtual and physical elements convincingly. At the core of all these applications lies a single question: given an image, how far away is each pixel?

The most straightforward solution is to use dedicated depth sensors — LiDAR, structured light cameras, or stereo rigs. These sensors are accurate but expensive, fragile, and impractical to deploy at scale. A more appealing alternative is **monocular depth estimation**: inferring depth from a single RGB image, using only the visual cues that humans rely on every day — perspective, texture gradients, object size, and occlusion.

The challenge is that monocular depth estimation is geometrically ill-posed. An infinite number of 3D scenes can project to the same 2D image. A small car close-up and a large car far away look identical in a single frame. Recovering depth from a single image therefore requires learning strong scene priors from data.

---

### 1.2 From Supervised to Self-Supervised Learning

The dominant approach to monocular depth estimation has historically been supervised learning — training a neural network to predict depth by showing it pairs of images with corresponding ground truth depth maps acquired from LiDAR or structured light sensors. This paradigm has produced impressive results, but it has a fundamental scalability problem: annotating depth is expensive. LiDAR scans are sparse, require careful calibration, and are limited to specific sensor setups. Creating large-scale labeled datasets is a bottleneck that constrains the amount and diversity of training data.

**Self-supervised monocular depth estimation** offers a compelling alternative. Instead of requiring ground truth depth labels, these methods learn from the temporal consistency of monocular video sequences. The key insight, formalized by Zhou et al. (2017) and refined by Godard et al. (2019) in Monodepth2, is:

> *If we correctly predict the depth of a frame and the camera's motion between frames, we can reconstruct one frame from another. The quality of this reconstruction — measured by photometric error — serves as a training signal without any ground truth depth.*

This approach is attractive for several reasons. First, monocular video is abundant and free — dashcams, smartphones, and surveillance cameras continuously produce unlabeled sequences at scale. Second, self-supervised training generalizes naturally to new environments, since no labeling effort is required for new domains. Third, the method is inherently tied to the actual geometry of the scene, unlike supervised methods that can overfit to dataset-specific depth distributions.

The main limitation of self-supervised methods is that they recover depth only up to an unknown scale factor. Without a reference (such as a known object size or a stereo baseline), the network cannot tell the difference between a scene that is 5m deep and one that is 50m deep. In practice, this is addressed at evaluation time by median scaling — aligning the predicted depth scale to the ground truth using the median ratio. While this limitation is real, self-supervised depth is still highly useful for relative depth, scene understanding, and applications where metric scale can be recovered from other sources (e.g., known camera height in autonomous driving).

---

### 1.3 Foundation Models and the Efficiency Challenge

In parallel with advances in self-supervised learning, the deep learning community has witnessed the rise of **foundation models** — large neural networks pretrained on internet-scale data that achieve remarkable zero-shot performance across diverse tasks. In the vision domain, models like DINOv2 (Oquab et al., 2024) and SAM (Kirillov et al., 2023) have demonstrated that scale and pretraining data quality can produce representations that transfer across domains with little or no fine-tuning.

Depth Pro (Bochkovskii et al., 2025) is a state-of-the-art depth estimation foundation model that combines a DINOv2-Large encoder (304M parameters) with a multi-resolution patch processing strategy to produce sharp metric depth maps in a single forward pass, without requiring camera calibration. Evaluated on diverse benchmarks including KITTI, NYU Depth V2, and ETH3D, Depth Pro demonstrates strong zero-shot generalization.

However, foundation models present a new challenge for adaptation: their size. Depth Pro contains 952 million parameters. Full fine-tuning on a new dataset or task requires gradient computation through the entire model, demanding GPU memory that far exceeds what consumer hardware provides. An NVIDIA RTX 4070 Ti — a high-end consumer GPU with 12GB VRAM — cannot fit the optimizer states for 952M parameters at once.

**Parameter-efficient fine-tuning (PEFT)** methods address this challenge. LoRA (Low-Rank Adaptation, Hu et al., 2022) injects small trainable low-rank matrices into the frozen pretrained model, achieving adaptation comparable to full fine-tuning while updating only a fraction of the parameters. Originally developed for large language models, LoRA has since been applied to vision transformers with strong results.

---

### 1.4 Research Gap and Thesis Contribution

Despite independent progress in both self-supervised depth estimation and foundation model adaptation, these two directions have not been combined. Prior self-supervised methods (Monodepth2, MonoViT, DIFFNet) train small custom architectures from scratch. Depth estimation foundation models (Depth Pro, Depth Anything) are trained with full supervision on massive labeled datasets. No prior work has asked:

> *Can a large pretrained depth foundation model be adapted to self-supervised training on unlabeled monocular video, using only a fraction of its parameters, on a consumer GPU?*

This thesis answers that question. Our contributions are:

**Contribution 1 — LoRA adaptation of Depth Pro for self-supervised training.**
We inject LoRA adapters (rank=8) into all 96 attention layers of Depth Pro's dual ViT encoders, enabling encoder adaptation with only 2.36M parameters — a 257× reduction compared to full fine-tuning. Combined with the fully trainable decoder (19.67M) and depth head (0.40M), the total trainable parameter count is 22.43M from the depth backbone.

**Contribution 2 — PoseNet integration for ego-motion estimation.**
We design and train a lightweight ResNet-18 based PoseNet (11.91M parameters) jointly with the depth backbone. PoseNet estimates the 6-DoF camera pose between consecutive frames, enabling the differentiable warping needed for photometric self-supervision. Total trainable parameters: 34.33M out of 966M (3.56%).

**Contribution 3 — Canonical depth scaling for geometrically consistent warping.**
Depth Pro outputs canonical inverse depth — a representation designed for zero-shot generalization that is independent of camera focal length. We identify that this representation must be scaled using the camera's actual focal length for geometrically consistent photometric training, and show that using the camera's ground truth intrinsics (rather than the model's FOV head prediction) is critical for training stability on KITTI.

Together, these contributions enable self-supervised monocular depth training of a 952M-parameter foundation model on a single 12GB consumer GPU, completing 20 epochs on KITTI in approximately 4.6 days.

---

### 1.5 Thesis Structure

The remainder of this thesis is organized as follows:

- **Chapter 2 — Related Work:** Reviews the literature on supervised and self-supervised monocular depth estimation, vision foundation models, and parameter-efficient fine-tuning methods that this work builds upon.

- **Chapter 3 — Method:** Describes the four modules introduced in this work: LoRA adapters for the depth backbone, PoseNet for ego-motion estimation, the differentiable warper, and the self-supervised loss functions.

- **Chapter 4 — Experiments:** Presents the experimental setup, evaluation protocol, and quantitative results on the KITTI Eigen benchmark. Includes comparison with self-supervised baselines (Monodepth2, MonoViT) and analysis of training dynamics.

- **Chapter 5 — Conclusion:** Summarizes findings, discusses limitations, and outlines directions for future work.
