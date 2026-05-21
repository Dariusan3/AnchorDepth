"""Evaluate LoRA model with TTA flip on 654 Eigen test images."""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
import depth_pro
from depth_pro.improvements.tta import tta_infer
from depth_pro.network.vit_factory import VIT_CONFIG_DICT
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model with LoRA structure, then load weights
    print("Loading model with LoRA...")
    model, transform = depth_pro.create_model_and_transforms(device=device)
    apply_lora_to_encoder(model, rank=8, alpha=16.0)

    # Move LoRA params to device
    import torch.nn as nn
    for enc_name in ["patch_encoder", "image_encoder"]:
        enc = getattr(model.encoder, enc_name)
        for block in enc.blocks:
            if isinstance(block.attn.qkv, LoRALinear):
                block.attn.qkv.lora_A = nn.Parameter(block.attn.qkv.lora_A.to(device))
                block.attn.qkv.lora_B = nn.Parameter(block.attn.qkv.lora_B.to(device))
            if isinstance(block.attn.proj, LoRALinear):
                block.attn.proj.lora_A = nn.Parameter(block.attn.proj.lora_A.to(device))
                block.attn.proj.lora_B = nn.Parameter(block.attn.proj.lora_B.to(device))

    # Load trained weights
    ckpt = torch.load("checkpoints/depth_pro_lora_best.pt", map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print("LoRA model loaded")

    # Load dataset
    f = h5py.File("datasets/nyu_depth_v2_labeled.mat", "r")
    images = f["images"]
    depths = f["depths"]

    with open("eigen_test_indices.json") as ef:
        test_indices = json.load(ef)
    print(f"Evaluating on {len(test_indices)} test images")

    # Run evaluation with TTA flip
    metrics_accum = {k: 0.0 for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "delta1", "delta2", "delta3"]}
    count = 0
    times = []

    for idx in tqdm(test_indices, desc="LoRA + TTA flip"):
        rgb_raw = images[idx].transpose(2, 1, 0)
        gt_depth = depths[idx].T

        img_pil = Image.fromarray(rgb_raw.astype(np.uint8))
        img_tensor = transform(img_pil)

        t0 = time.time()
        prediction = tta_infer(model, img_tensor, f_px=None, mode="flip")
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
        count += 1

    for k in metrics_accum:
        metrics_accum[k] /= count

    metrics_accum["avg_inference_time"] = float(np.mean(times))
    metrics_accum["num_samples"] = count

    with open("eval_results_nyu_lora_tta.json", "w") as out:
        json.dump(metrics_accum, out, indent=2)

    print(f"\nResults (LoRA + TTA flip):")
    print(f"  AbsRel:     {metrics_accum['abs_rel']:.4f}")
    print(f"  RMSE:       {metrics_accum['rmse']:.4f}")
    print(f"  delta<1.25: {metrics_accum['delta1']:.4f}")
    print(f"  Avg time:   {metrics_accum['avg_inference_time']:.3f}s")
    print(f"Saved to eval_results_nyu_lora_tta.json")


if __name__ == "__main__":
    main()
