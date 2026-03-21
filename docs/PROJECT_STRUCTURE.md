# Project Structure

```
ml-depth-pro/
├── src/depth_pro/                  # Core model code (from Apple)
│   ├── depth_pro.py                # Main model class
│   ├── utils.py                    # Image loading utilities
│   ├── network/
│   │   ├── encoder.py              # Multi-resolution ViT encoder
│   │   ├── decoder.py              # CNN feature fusion decoder
│   │   ├── fov.py                  # Field-of-view estimation
│   │   ├── vit.py                  # Vision Transformer backbone
│   │   └── vit_factory.py          # ViT config and factory
│   ├── improvements/               # YOUR MODIFICATIONS GO HERE
│   │   ├── __init__.py
│   │   ├── tta.py                  # Test-time augmentation
│   │   ├── losses.py               # Advanced loss functions
│   │   ├── augmentations.py        # Training augmentations
│   │   ├── attention_decoder.py    # Transformer decoder
│   │   ├── edge_refine.py          # Edge-aware refinement
│   │   └── ...
│   ├── eval/
│   │   └── boundary_metrics.py     # Boundary F1/Recall metrics
│   └── cli/
│       └── run.py                  # Apple's CLI inference
│
├── scripts/                        # Training & evaluation scripts
│   ├── train_nyu.py                # Fine-tuning script
│   ├── evaluate_nyu.py             # Evaluation on NYU Depth V2
│   ├── save_results.py             # Save visual comparison results
│   └── run_experiment.py           # Full experiment pipeline
│
├── experiments/                    # One folder per experiment
│   ├── v0_baseline/
│   │   ├── README.md               # Experiment documentation
│   │   └── eval_results_nyu_full.json
│   ├── v1_finetuned_decoder/
│   │   ├── README.md
│   │   ├── eval_results.json
│   │   ├── training_log.json
│   │   ├── depth_pro_finetuned_best.pt
│   │   └── results/                # Visual outputs
│   │       ├── rgb/
│   │       ├── depth_pretrained/
│   │       ├── depth_finetuned/
│   │       ├── depth_gt/
│   │       ├── comparison/
│   │       ├── error_maps/
│   │       └── npz/
│   └── v2_tta/                     # Next experiment...
│
├── docs/                           # Documentation
│   ├── EXPERIMENTS.md              # Master experiment log table
│   ├── IMPROVEMENT_ROADMAP.md      # Ideas and plan
│   └── PROJECT_STRUCTURE.md        # This file
│
├── checkpoints/                    # Model weights
│   ├── depth_pro.pt                # Apple pretrained (1.8GB)
│   └── depth_pro_finetuned_best.pt # Current best fine-tuned
│
├── datasets/                       # Evaluation datasets
│   └── nyu_depth_v2_labeled.mat    # NYU Depth V2 (2.8GB)
│
└── data/                           # Sample images
    └── example.jpg
```

## How to Run an Experiment

```bash
# Full pipeline: train → evaluate → save visuals → document
python scripts/run_experiment.py \
  --name v2_tta \
  --description "Test-time augmentation with multi-scale + flip" \
  --train-args "--epochs 25 --lr 1e-4"

# Evaluate only (no training)
python scripts/run_experiment.py \
  --name v2_tta \
  --checkpoint checkpoints/some_model.pt \
  --skip-training

# Manual steps
python scripts/train_nyu.py --epochs 25 --lr 1e-4
python scripts/evaluate_nyu.py --checkpoint checkpoints/depth_pro_finetuned_best.pt --scale-invariant
python scripts/save_results.py --num-samples 20
```

## Adding a New Improvement

1. Create your module in `src/depth_pro/improvements/`
2. Modify `scripts/train_nyu.py` or create a variant
3. Run via `scripts/run_experiment.py` — it handles everything
4. Results auto-logged to `docs/EXPERIMENTS.md`
5. Visual outputs saved to `experiments/<name>/results/`
