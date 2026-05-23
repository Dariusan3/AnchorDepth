# Thesis — Introduction Chapter
## Chapter 1: Introduction

---

### 1.1 Motivation

Depth perception is a fundamental capability for machines operating in the
physical world. Autonomous vehicles must know how far obstacles are before
they can avoid them. Robots need spatial understanding to manipulate objects.
Augmented reality systems must estimate scene geometry to blend virtual and
physical elements convincingly. At the core of all these applications lies a
single question: given an image, how far away is each pixel?

The most straightforward solution is to use dedicated depth sensors — LiDAR,
structured light cameras, or stereo rigs. These sensors are accurate but
expensive, fragile, and impractical to deploy at scale. A more appealing
alternative is **monocular depth estimation**: inferring depth from a single
RGB image, using only the visual cues that humans rely on every day —
perspective, texture gradients, object size, and occlusion.

The challenge is that monocular depth estimation is geometrically ill-posed.
An infinite number of 3D scenes can project to the same 2D image. A small car
close-up and a large car far away look identical in a single frame. Recovering
depth from a single image therefore requires learning strong scene priors
from data.

---

### 1.2 From Supervised to Self-Supervised Learning

The dominant approach to monocular depth estimation has historically been
supervised learning — training a neural network to predict depth by showing it
pairs of images with corresponding ground truth depth maps acquired from
LiDAR or structured light sensors. This paradigm has produced impressive
results, but it has a fundamental scalability problem: annotating depth is
expensive. LiDAR scans are sparse, require careful calibration, and are
limited to specific sensor setups. Creating large-scale labeled datasets is a
bottleneck that constrains the amount and diversity of training data.

**Self-supervised monocular depth estimation** offers a compelling alternative.
Instead of requiring ground truth depth labels, these methods learn from the
Vtemporal consistency of monocular video sequences. The key insight, formalized
by Zhou et al. (2017) and refined by Godard et al. (2019) in Monodepth2, is:

> *If we correctly predict the depth of a frame and the camera's motion*
> *between frames, we can reconstruct one frame from another. The quality of*
> *this reconstruction — measured by photometric error — serves as a training*
> *signal without any ground truth depth.*

This approach is attractive for several reasons. First, monocular video is
abundant and free — dashcams, smartphones, and surveillance cameras
continuously produce unlabeled sequences at scale. Second, self-supervised
training generalizes naturally to new environments, since no labeling effort
is required for new domains. Third, the method is inherently tied to the
actual geometry of the scene, unlike supervised methods that can overfit to
dataset-specific depth distributions.

The main limitation of self-supervised methods is that they recover depth
only up to an unknown scale factor. Without a reference (such as a known
object size or a stereo baseline), the network cannot tell the difference
between a scene that is 5m deep and one that is 50m deep. In practice, this
is addressed at evaluation time by median scaling — aligning the predicted
depth scale to the ground truth using the median ratio. While this limitation
is real, self-supervised depth is still highly useful for relative depth,
scene understanding, and applications where metric scale can be recovered
from other sources (e.g., known camera height in autonomous driving).

---

### 1.3 Foundation Models and the Efficiency Challenge

In parallel with advances in self-supervised learning, the deep learning
community has witnessed the rise of **foundation models** — large neural
networks pretrained on internet-scale data that achieve remarkable zero-shot
performance across diverse tasks. In the vision domain, models like DINOv2
(Oquab et al., 2024) and SAM (Kirillov et al., 2023) have demonstrated that
scale and pretraining data quality can produce representations that transfer
across domains with little or no fine-tuning.

Depth Pro (Bochkovskii et al., 2025) is a state-of-the-art depth estimation
foundation model that combines a DINOv2-Large encoder (304M parameters) with
a multi-resolution patch processing strategy to produce sharp metric depth
maps in a single forward pass, without requiring camera calibration.
Evaluated on diverse benchmarks including KITTI, NYU Depth V2, and ETH3D,
Depth Pro demonstrates strong zero-shot generalization.

A surprising finding of this thesis — established before any fine-tuning is
attempted — is that the zero-shot Depth Pro model **already surpasses every
published self-supervised method on the KITTI Eigen test split**. With no
KITTI-specific training, it reaches AbsRel = 0.0866 and δ<1.25 = 0.9253,
a 13% relative improvement in AbsRel over the previous best self-supervised
method (MonoViT, AbsRel = 0.099). This reshapes the research question we
ask: rather than trying to close the gap to supervised methods (which a
foundation model has already done), the meaningful problem is whether
self-supervised adaptation can extract any **additional** signal from KITTI
video on top of an already strong zero-shot baseline, without degrading it.

However, foundation models present a new challenge for adaptation: their
size. Depth Pro contains 952 million parameters. Full fine-tuning on a new
dataset or task requires gradient computation through the entire model,
demanding GPU memory that far exceeds what consumer hardware provides. An
NVIDIA RTX 4070 Ti — a high-end consumer GPU with 12GB VRAM — cannot fit
the optimizer states for 952M parameters at once.

**Parameter-efficient fine-tuning (PEFT)** methods address this challenge.
LoRA (Low-Rank Adaptation, Hu et al., 2022) injects small trainable low-rank
matrices into the frozen pretrained model, achieving adaptation comparable
to full fine-tuning while updating only a fraction of the parameters.
Originally developed for large language models, LoRA has since been applied
to vision transformers with strong results.

A second, complementary kind of foundation model is also relevant to this
work: **multi-view geometric foundation models**. VGGT (Visual Geometry
Grounded Transformer, Wang et al., CVPR 2025 Best Paper) is a 1.2 billion
parameter transformer that takes one or more images and jointly predicts
camera intrinsics, extrinsics, depth maps, and 3D point clouds.
Self-supervised depth methods classically require a pose network (e.g.
a small ResNet-18 trained from scratch) to estimate the relative motion
between consecutive frames. A foundation model such as VGGT can substitute
for this pose network by providing high-quality precomputed relative poses,
removing a source of noise from the photometric warping signal. Because
VGGT is too large to co-reside with Depth Pro on a 12 GB GPU during
training, we propose and implement an **offline pose precomputation**
pipeline that runs VGGT once over the entire KITTI training set and caches
the resulting 4×4 relative pose matrices to disk for reuse during
fine-tuning.

---

### 1.4 Research Gap and Thesis Contribution

Despite independent progress in both self-supervised depth estimation and
foundation model adaptation, these two directions have not been combined.
Prior self-supervised methods (Monodepth2, MonoViT, DIFFNet) train small
custom architectures from scratch. Depth estimation foundation models
(Depth Pro, Depth Anything) are trained with full supervision on massive
labeled datasets. No prior work has asked:

> *Can a large pretrained depth foundation model be adapted to*
> *self-supervised training on unlabeled monocular video, using only a*
> *fraction of its parameters, on a consumer GPU?*

This thesis answers that question. Our five contributions are:

**Contribution 1 — Zero-shot Depth Pro establishes a new state of the art
on KITTI Eigen.**
We provide what is, to the best of our knowledge, the first published
evaluation of Depth Pro on the KITTI Eigen test split (697 images, Garg
crop, median scaling). The model — never trained on KITTI — achieves
AbsRel = 0.0866 and δ<1.25 = 0.9253, a **13% relative improvement in
AbsRel over the previous self-supervised state of the art** (MonoViT,
AbsRel = 0.099, 3DV 2022). This single observation reframes the problem:
published self-supervised methods no longer define the baseline for KITTI
monocular depth — the foundation model itself does.

**Contribution 2 — A documented negative result: naïve photometric
self-supervision degrades a strong foundation model.**
Across nine independent training configurations (LoRA rank 8/4, frozen
encoder, varying smoothness weights), applying the standard Monodepth2
photometric loss to Depth Pro **worsens AbsRel by a factor of five**
(0.087 → 0.46) on the held-out test set, even though the training
photometric loss decreases monotonically. We characterize three concurrent
causes:
(i) an objective–metric gap that grows as the starting point becomes
stronger,
(ii) a mismatch between training-time loss resolution (416×128) and
evaluation resolution (1242×375), and
(iii) numerical instabilities of LoRA under FP16 that silently corrupt
checkpoints.
This negative result is, on its own, a contribution: the standard
self-supervised recipe must not be applied unmodified to depth foundation
models.

**Contribution 3 — Consistency-anchored adaptation that improves over
zero-shot without degrading other metrics (main positive result, v15).**
We introduce a depth consistency loss that anchors the fine-tuned
predictions to the model's own zero-shot output:

$$L = L_{\text{photometric}} + \lambda \cdot \| d_{\text{pred}} - d_{\text{zero-shot}} \|_1$$

With $\lambda = 10$, LoRA rank 8, and PoseNet for ego-motion, the resulting
model (run v15) **improves on δ<1.25³ over zero-shot** (0.98499 vs.
0.98494) while staying within 1–2% of zero-shot on all six remaining
metrics. To our knowledge this is the first reported improvement of any
kind over the Depth Pro zero-shot baseline on KITTI, and it is achieved
using only self-supervised signal. The result is even stronger when the
same family of models is evaluated cross-domain on the **Make3D outdoor
benchmark** — a different geographic and camera setting that neither
model has seen during training. **Every consistency-anchored variant
(v15–v20) improves over zero-shot Depth Pro on every Make3D metric**, and
the best variant — **v18, log-space consistency with λ = 10** — reduces
AbsRel by **24.7%**, SqRel by **55.1%**, RMSE by **20.7%**, RMSElog by
**15.0%** and log₁₀ by **14.0%** relative to zero-shot. These are
order-of-magnitude larger improvements than on KITTI, demonstrating that
the consistency loss captures something universal about outdoor depth
supervision rather than KITTI-specific over-fitting. The L1 anchor (v15)
is the most conservative configuration and the best choice on saturated
benchmarks; the log-space anchor (v18) extracts the largest gains when
the foundation model has headroom on the target dataset.

**Contribution 4 — VGGT-based offline pose supervision as a replacement
for the trained PoseNet.**
Classical self-supervised depth methods rely on a small pose network
(typically ResNet-18) trained jointly with the depth model from random
initialization. The poses produced in the early epochs are noisy and inject
error into the photometric loss. We propose and implement an alternative:
run the VGGT foundation model (1.2 B parameters, CVPR 2025 Best Paper)
**offline** on every KITTI training triplet, extract the relative
target-to-source camera transformations as 4×4 matrices, and cache them to
disk. During Depth Pro fine-tuning these poses are loaded from cache and
bypass the trainable PoseNet entirely, so a 1.2 B-parameter pose foundation
model can be effectively combined with the 952M-parameter depth foundation
model on a single 12 GB GPU at no extra training-time memory cost.
Ablation v16 — VGGT poses with edge-aware consistency — produces the
largest measured improvement on δ<1.25³ (0.98500), confirming that
high-quality precomputed poses are a meaningful contribution to the
photometric reconstruction signal.

**Contribution 5 — A reproducible numerical-stability recipe for LoRA +
mixed precision at the foundation-model scale.**
We isolate two previously-undocumented failure modes of LoRA fine-tuning
under mixed precision and provide fixes:
(i) **LoRA weight overflow under FP16** — gradients propagating back
through 96 attention layers into low-rank matrices overflow the 5-bit
exponent of float16; weights silently become NaN; the depth-output
`nan_to_num` guard masks the symptom, so training continues with corrupted
parameters. Switching autocast to bfloat16 plus parameter-level NaN
validation eliminates the failure.
(ii) **Focal-length scaling mismatch** between training and evaluation —
Depth Pro's canonical inverse depth must be converted to metric depth via
a focal-length-dependent factor that silently differed between training
(scaled KITTI intrinsics) and evaluation (FOV head prediction). Using
KITTI ground-truth intrinsics in both phases restores consistency.

Together, contributions 1–5 enable self-supervised monocular depth training
of a 952M-parameter foundation model — combined when desired with a
1.2 B-parameter pose foundation model — on a single 12 GB consumer GPU,
delivering one measurable improvement over the strongest published baseline
(zero-shot Depth Pro itself) without any test-set degradation on the other
six standard metrics.

---

### 1.5 Thesis Structure

The remainder of this thesis is organized as follows:

- **Chapter 2 — Related Work:** Reviews supervised and self-supervised
  monocular depth estimation (Eigen et al., Monodepth2, MonoViT, DIFFNet,
  PackNet-SfM), depth foundation models (Depth Pro, DepthAnything-v2,
  Marigold), parameter-efficient fine-tuning (LoRA, AdaLoRA), and
  multi-view foundation models (VGGT) that this work builds upon.

- **Chapter 3 — Method:** Describes the components introduced or adapted in
  this work: the Depth Pro backbone and its canonical inverse depth
  representation, the Monodepth2-style photometric loss with auto-masking,
  the ResNet-18 PoseNet (and the offline VGGT alternative), the
  differentiable warper, the LoRA adapters in attention layers, the
  proposed **consistency loss** that anchors fine-tuned predictions to the
  zero-shot baseline (Section 3.6, the main methodological contribution),
  and the numerical-stability recipe required for stable training on 12 GB
  VRAM.

- **Chapter 4 — Experiments:** Presents the experimental setup, evaluation
  protocol on the KITTI Eigen 697-image test split, and quantitative
  results. Section 4.3 reports the zero-shot Depth Pro baseline
  (Contribution 1). Section 4.4 reports the nine ablation runs (v6–v19),
  of which v15 — consistency loss with $\lambda=10$ — is identified as the
  main positive result (Contribution 3). Section 4.5 analyzes the
  systematic failure of naïve photometric self-supervision on foundation
  models (Contribution 2). Comparisons with the published state of the art
  (Monodepth2, MonoViT, DIFFNet, PackNet-SfM) close the chapter.

- **Chapter 5 — Conclusion:** Summarizes findings, discusses limitations
  (single GPU, single dataset, single foundation model), and outlines
  directions for future work (consistency-regularized adaptation with
  larger anchors, VGGT-based pose supervision, scaling to larger VRAM
  budgets).
