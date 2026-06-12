# Chapter 5 — Conclusions

In our study, we re-examine the problem of adapting large pretrained depth
foundation models to specific outdoor distributions under a strict
consumer-GPU constraint, and present a novel and efficient method which
we have named **AnchorDepth**. This method is designed to overcome the
catastrophic-forgetting failure mode that pure photometric
self-supervision exhibits when applied to a strong pretrained model such
as Depth Pro — a limitation that, if left uncorrected, degrades the
test-set AbsRel by a factor of five. AnchorDepth is a self-supervised
adaptation pipeline built on top of Depth Pro: LoRA adapters at rank 8
inject 2.36 M trainable parameters into the frozen attention layers, a
trainable Monodepth2-style decoder and a ResNet-18 PoseNet jointly
support the photometric reconstruction objective, and a **consistency
anchor** against the model's own zero-shot prediction prevents the
adaptation from drifting away from the strong prior. The entire system
fits in 12 GB of VRAM and trains in approximately 12 hours per
configuration on a single RTX 4070 Ti. On KITTI Eigen, AnchorDepth
improves over the state-of-the-art zero-shot Depth Pro baseline on
δ<1.25³ and stays within 1–2% on the remaining metrics; on Cityscapes,
it improves over zero-shot on **all seven** standard metrics; and on
Make3D it improves on **all five** standard metrics with double-digit
percentage gains (AbsRel −24.7%, SqRel −55.1%). Additionally, we
showcase the enhanced cross-domain generalisability of our model and the
existence of a controllable variant–saturation pairing whereby the
optimal consistency-loss configuration tracks the saturation of the
zero-shot baseline on the target benchmark. Our findings position
AnchorDepth as a leading contender for future work on parameter-efficient
self-supervised adaptation of depth foundation models on consumer
hardware.

Regarding future development and research, due to the recent advancement
and expansion of depth foundation models, there are four major
directions worth investigating: applying the consistency-anchored
adaptation recipe to other depth foundation models — in particular
DepthAnything-v2, Marigold and Metric3D-v2 — to verify that the
variant–saturation pairing observed in this work generalises across
model families; extending the evaluation to indoor distributions such as
NYU Depth V2 to probe whether the consistency anchor remains beneficial
when the zero-shot baseline is less saturated than on outdoor driving
imagery; replacing the depth-space L1 anchor with richer distillation
signals — for instance feature-map agreement at intermediate decoder
layers, or learned per-pixel uncertainty re-weighting — to reduce the
gap between the L1, log-space and edge-aware variants studied here; and
last but not least, optimising the PoseNet network by replacing the
trainable ResNet-18 with a precomputed cache from a multi-view
geometric foundation model such as VGGT in the production setting, while
re-deriving the consistency-loss recipe under that supervision.
