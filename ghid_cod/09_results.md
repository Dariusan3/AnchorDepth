# 9. `results/`

## Ce este
Rezultatele finale care intră în teză: 29 JSON-uri cu metrici, 8 figuri PNG, tabele formatate
(Markdown + LaTeX). Output-urile produse de `evaluation/` și `tools/`.

## A. JSON-uri de evaluare (29) — decodorul variantelor
| Variantă | Ce testează |
|---|---|
| `zeroshot` / `pretrained_zeroshot` | Depth Pro nemodificat (baseline) |
| `v7_no_lora` / `v11_no_lora` | fine-tuning fără LoRA (ablație) |
| `v10_lora` / `v13_lora_only` | doar LoRA, fără consistency |
| `v15_consistency` | **AnchorDepth principal pe KITTI** (L1, λ=10) |
| `v18_log_l10` | varianta log-space → **câștigătoare Make3D** |
| `v20_consistency_lambda20` | λ=20 → **câștigătoare Cityscapes** |
| `v16_vggt_edge`, `v17_log_far`, `v19_vggt_log` | ablații (VGGT, ponderare distanță) |

Fiecare = 7 metrici + `num_samples` + `mean_scale`. Baza tuturor tabelelor.

## B. Figuri PNG (8)
- `figure_{kitti,cityscapes,make3d}_qualitative.png` — comparațiile vizuale (slide-uri de impact).
- `figure_lambda_ablation.png` — graficul ablației pe λ (λ=10 optim).
- `v7_training_curves.png`, `comparison_*.png`.

## C. Tabele formatate
- `thesis_figures/results_table.tex` — tabelul KITTI în LaTeX (Monodepth2/DIFFNet/MonoViT completate).
  ⚠️ Rândurile tale au încă **`TBD`** — completează-le înainte de predare.
- `thesis_figures/results_table.md`, `make3d_comparison.md`.

## ⚠️ Neconcordanță de cifre — verifică înainte de apărare
- `eval_pretrained_zeroshot.json` → KITTI **AbsRel = 0.0866**
- `eval_v15_consistency.json` → KITTI **AbsRel = 0.0875** (ușor mai SLAB decât zero-shot)
- README/teza revendică AnchorDepth = **0.0852** (mai bun)

Comentariile din `scripts/run_v20.sh` confirmă: `AbsRel < 0.0866 (v15: 0.0875, gap 0.0009)` și
`δ<1.25³ (v15: 0.98499 ✓ already won)`. Deci v15 bate zero-shot **doar pe δ<1.25³** pe KITTI.

➡️ Cifra 0.0852 vine probabil dintr-o rulare mai nouă/altă variantă. **Reconciliază README ↔ teză ↔
JSON ↔ ce spui la comisie.** Rulează din nou `evaluate_kitti.py` pe checkpoint-ul final.

## De spus la comisie
„`results/` centralizează output-urile: JSON-uri cu metrici per variantă/dataset, figuri calitative,
ablația pe λ, tabele LaTeX/Markdown. Denumirea codifică experimentul — v15 KITTI, v18 Make3D, v20 Cityscapes."
