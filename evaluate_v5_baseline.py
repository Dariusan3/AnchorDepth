"""Evaluate v5 model (LoRA + 2 unfrozen blocks) WITHOUT TTA on 654 Eigen test images."""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
import depth_pro
from train_nyu_lora import LoRALinear, apply_lora_to_encoder


def compute_metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    thresh = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "log10": float(np.mean(np.abs(np.log10(p) - np.log10(g)))),
        "delta1": float(np.mean(thresh < 1.25)),
        "delta2": float(np.mean(thresh < 1.25 ** 2)),
        "delta3": float(np.mean(thresh < 1.25 ** 3)),
    }


def compute_si_metrics(pred, gt, mask):
    """Scale-invariant metrics (optimal per-image affine alignment)."""
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    A = np.vstack([p, np.ones_like(p)]).T
    result = np.linalg.lstsq(A, g, rcond=None)
    scale, shift = result[0]
    p_aligned = pred * scale + shift
    p_aligned = np.clip(p_aligned, 1e-3, 10.0)
    return compute_metrics(p_aligned, gt, mask)


def load_v5_model(device, checkpoint_path="checkpoints/depth_pro_lora_unfreeze2_best.pt"):
    """Load model with LoRA + 2 unfrozen blocks structure, then load checkpoint."""
    model, transform = depth_pro.create_model_and_transforms(device=device)
    apply_lora_to_encoder(model, rank=8, alpha=16.0)

    # Move LoRA params to device
    for enc_name in ["patch_encoder", "image_encoder"]:
        enc = getattr(model.encoder, enc_name)
        for block in enc.blocks:
            if isinstance(block.attn.qkv, LoRALinear):
                block.attn.qkv.lora_A = nn.Parameter(block.attn.qkv.lora_A.to(device))
                block.attn.qkv.lora_B = nn.Parameter(block.attn.qkv.lora_B.to(device))
            if isinstance(block.attn.proj, LoRALinear):
                block.attn.proj.lora_A = nn.Parameter(block.attn.proj.lora_A.to(device))
                block.attn.proj.lora_B = nn.Parameter(block.attn.proj.lora_B.to(device))

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model, transform


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading v5 model (LoRA + 2 unfrozen blocks)...")
    model, transform = load_v5_model(device)
    print(f"Model loaded on {device}")

    f = h5py.File("datasets/nyu_depth_v2_labeled.mat", "r")
    images = f["images"]
    depths = f["depths"]

    with open("eigen_test_indices.json") as ef:
        test_indices = json.load(ef)
    print(f"Evaluating on {len(test_indices)} test images")

    metrics_accum = {k: 0.0 for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "delta1", "delta2", "delta3"]}
    si_accum = {k: 0.0 for k in metrics_accum}
    count = 0
    times = []

    for idx in tqdm(test_indices, desc="v5 baseline"):
        rgb_raw = images[idx].transpose(2, 1, 0)
        gt_depth = depths[idx].T

        img_pil = Image.fromarray(rgb_raw.astype(np.uint8))
        img_tensor = transform(img_pil)

        t0 = time.time()
        prediction = model.infer(img_tensor, f_px=None)
        pred_depth = prediction["depth"].detach().cpu().numpy().squeeze()
        times.append(time.time() - t0)

        if pred_depth.shape != gt_depth.shape:
            pred_pil = Image.fromarray(pred_depth)
            pred_pil = pred_pil.resize((gt_depth.shape[1], gt_depth.shape[0]), Image.BILINEAR)
            pred_depth = np.array(pred_pil)

        mask = (gt_depth > 1e-3) & (pred_depth > 1e-3) & (gt_depth < 10.0)
        if mask.sum() < 100:
            continue

        pred_depth = np.clip(pred_depth, 1e-3, 10.0)

        m = compute_metrics(pred_depth, gt_depth, mask)
        for k in metrics_accum:
            metrics_accum[k] += m[k]

        si_m = compute_si_metrics(pred_depth, gt_depth, mask)
        for k in si_accum:
            si_accum[k] += si_m[k]

        count += 1

    for k in metrics_accum:
        metrics_accum[k] /= count
        si_accum[k] /= count

    metrics_accum["avg_inference_time"] = float(np.mean(times))
    metrics_accum["num_samples"] = count
    metrics_accum["scale_invariant"] = si_accum

    with open("eval_results_v5_baseline.json", "w") as out:
        json.dump(metrics_accum, out, indent=2)

    print(f"\nResults (v5 baseline - LoRA + 2 unfrozen blocks):")
    print(f"  AbsRel:     {metrics_accum['abs_rel']:.4f}")
    print(f"  RMSE:       {metrics_accum['rmse']:.4f}")
    print(f"  delta<1.25: {metrics_accum['delta1']:.4f}")
    print(f"  SI AbsRel:  {si_accum['abs_rel']:.4f}")
    print(f"  SI RMSE:    {si_accum['rmse']:.4f}")
    print(f"  Avg time:   {metrics_accum['avg_inference_time']:.3f}s")
    print(f"Saved to eval_results_v5_baseline.json")


if __name__ == "__main__":
    main()
