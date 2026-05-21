"""Generate comprehensive comparison chart of all experiments (v0-v5)."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Collect all results
results = {}

# v0: Pretrained baseline
with open("eval_results_nyu_full.json") as f:
    d = json.load(f)
    results["v0 Pretrained"] = d

# v1: Decoder fine-tuning
with open("eval_results_nyu_finetuned.json") as f:
    d = json.load(f)
    results["v1 Decoder FT"] = d

# v3: Phase 1 (complex losses)
with open("eval_results_nyu_v3.json") as f:
    d = json.load(f)
    results["v3 Phase1"] = d

# v4: LoRA baseline
with open("eval_results_nyu_lora_baseline.json") as f:
    d = json.load(f)
    results["v4 LoRA"] = d

# v4 + TTA
with open("eval_results_nyu_lora_tta.json") as f:
    d = json.load(f)
    results["v4 LoRA+TTA"] = d

# v5: LoRA + unfreeze 2
with open("eval_results_v5_baseline.json") as f:
    d = json.load(f)
    results["v5 LoRA+Unfreeze"] = d

# v5 + TTA
with open("eval_results_v5_tta.json") as f:
    d = json.load(f)
    results["v5 LoRA+Unf+TTA"] = d

names = list(results.keys())
colors = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71', '#27ae60', '#9b59b6', '#8e44ad']

# Extract metrics
abs_rel = [results[n]["abs_rel"] for n in names]
rmse = [results[n]["rmse"] for n in names]
delta1 = [results[n]["delta1"] for n in names]
sq_rel = [results[n]["sq_rel"] for n in names]
rmse_log = [results[n]["rmse_log"] for n in names]
log10_vals = [results[n]["log10"] for n in names]

# SI metrics (where available)
si_abs_rel = []
si_rmse = []
for n in names:
    si = results[n].get("scale_invariant", {})
    si_abs_rel.append(si.get("abs_rel", None))
    si_rmse.append(si.get("rmse", None))

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Depth Pro NYU Depth V2 — All Experiments Comparison\n(654 Eigen Test Images)",
             fontsize=16, fontweight='bold', y=0.98)

# 1. AbsRel (lower is better)
ax = axes[0, 0]
bars = ax.bar(range(len(names)), abs_rel, color=colors)
ax.set_title("AbsRel ↓ (lower is better)", fontsize=13, fontweight='bold')
ax.set_ylabel("AbsRel")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
for i, v in enumerate(abs_rel):
    ax.text(i, v + 0.001, f"{v:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
best_idx = np.argmin(abs_rel)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)
ax.set_ylim(0, max(abs_rel) * 1.15)

# 2. RMSE (lower is better)
ax = axes[0, 1]
bars = ax.bar(range(len(names)), rmse, color=colors)
ax.set_title("RMSE ↓ (lower is better)", fontsize=13, fontweight='bold')
ax.set_ylabel("RMSE")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
for i, v in enumerate(rmse):
    ax.text(i, v + 0.005, f"{v:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
best_idx = np.argmin(rmse)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)
ax.set_ylim(0, max(rmse) * 1.15)

# 3. delta < 1.25 (higher is better)
ax = axes[0, 2]
bars = ax.bar(range(len(names)), delta1, color=colors)
ax.set_title("δ < 1.25 ↑ (higher is better)", fontsize=13, fontweight='bold')
ax.set_ylabel("δ < 1.25")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
for i, v in enumerate(delta1):
    ax.text(i, v + 0.002, f"{v:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
best_idx = np.argmax(delta1)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)
ax.set_ylim(min(delta1) * 0.97, 1.005)

# 4. SI AbsRel (lower is better)
ax = axes[1, 0]
valid_si = [(n, v) for n, v in zip(names, si_abs_rel) if v is not None]
si_names = [x[0] for x in valid_si]
si_vals = [x[1] for x in valid_si]
si_colors = [colors[names.index(n)] for n in si_names]
bars = ax.bar(range(len(si_names)), si_vals, color=si_colors)
ax.set_title("SI AbsRel ↓ (scale-invariant)", fontsize=13, fontweight='bold')
ax.set_ylabel("SI AbsRel")
ax.set_xticks(range(len(si_names)))
ax.set_xticklabels(si_names, rotation=35, ha='right', fontsize=9)
for i, v in enumerate(si_vals):
    ax.text(i, v + 0.001, f"{v:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
best_idx = np.argmin(si_vals)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)
ax.set_ylim(0, max(si_vals) * 1.15)

# 5. Improvement over pretrained baseline
ax = axes[1, 1]
baseline_abs_rel = results["v0 Pretrained"]["abs_rel"]
improvements = [(baseline_abs_rel - v) / baseline_abs_rel * 100 for v in abs_rel]
bars = ax.bar(range(len(names)), improvements, color=colors)
ax.set_title("AbsRel Improvement vs Pretrained (%)", fontsize=13, fontweight='bold')
ax.set_ylabel("Improvement (%)")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
for i, v in enumerate(improvements):
    y_pos = v + 0.5 if v >= 0 else v - 1.5
    ax.text(i, y_pos, f"{v:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

# 6. Summary table
ax = axes[1, 2]
ax.axis('off')
table_data = [
    ["Experiment", "AbsRel", "RMSE", "δ<1.25", "Params"],
    ["v0 Pretrained", "0.1155", "0.5092", "0.8777", "0 (frozen)"],
    ["v1 Decoder FT", "0.0855", "0.3288", "0.9389", "20.1M"],
    ["v3 Phase1", "0.0878", "0.3273", "0.9362", "20.1M"],
    ["v4 LoRA", "0.0781", "0.3038", "0.9530", "2.36M"],
    ["v4 LoRA+TTA", "0.0765", "0.2982", "0.9549", "2.36M"],
    ["v5 LoRA+Unfreeze", "0.0787", "0.3069", "0.9520", "56M"],
    ["v5 LoRA+Unf+TTA", "0.0773", "0.3014", "0.9546", "56M"],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='center', colColours=['#ddd']*5)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.5)

# Highlight best row (v4 LoRA+TTA)
for j in range(5):
    table[5, j].set_facecolor('#d5f5e3')
    table[5, j].set_text_props(fontweight='bold')

ax.set_title("Summary Table", fontsize=13, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("results_comparison_all.png", dpi=150, bbox_inches='tight')
print("Chart saved to results_comparison_all.png")

# Also print a text comparison
print("\n" + "="*90)
print("FULL COMPARISON TABLE")
print("="*90)
print(f"{'Experiment':<22} {'AbsRel':>8} {'RMSE':>8} {'δ<1.25':>8} {'SqRel':>8} {'RMSElog':>8} {'SI_AR':>8}")
print("-"*90)
for n in names:
    r = results[n]
    si = r.get("scale_invariant", {}).get("abs_rel", None)
    si_str = f"{si:.4f}" if si else "  N/A"
    print(f"{n:<22} {r['abs_rel']:>8.4f} {r['rmse']:>8.4f} {r['delta1']:>8.4f} {r['sq_rel']:>8.4f} {r['rmse_log']:>8.4f} {si_str:>8}")
