# Ghid de cod AnchorDepth — explicații pe foldere

Citește fiecare fișier **lângă codul** corespunzător (deschide-le în split în VS Code).
Ordinea recomandată = ordinea numerelor. `src/` (12) e miezul tehnic.

| # | Fișier ghid | Acoperă |
|---|---|---|
| — | [GHID_PREZENTARE_LICENTA.md](../GHID_PREZENTARE_LICENTA.md) | rezumatul general + 16 întrebări de comisie |
| 1 | [01_checkpoints.md](01_checkpoints.md) | `checkpoints/` — greutăți + jurnale |
| 2 | [02_data.md](02_data.md) | `data/` — imagini exemplu + indici |
| 4 | [04_evaluation.md](04_evaluation.md) | `evaluation/` — metrici pe 3 dataset-uri |
| 7 | [07_legacy.md](07_legacy.md) | `legacy/` — cod vechi arhivat |
| 9 | [09_results.md](09_results.md) | `results/` — JSON-uri + figuri + tabele |
| 10 | [10_scripts.md](10_scripts.md) | `scripts/` — automatizarea experimentelor |
| 11 | [11_splits.md](11_splits.md) | `splits/` — împărțirea Eigen KITTI |
| 12A | [12A_depth_pro.md](12A_depth_pro.md) | `src/depth_pro/depth_pro.py` — orchestratorul |
| 12B | [12B_encoder.md](12B_encoder.md) | `network/encoder.py` — encoderul multi-scale |
| 12C | [12C_decoder.md](12C_decoder.md) | `network/decoder.py` — decoderul DPT/FPN |
| 12D | [12D_fov.md](12D_fov.md) | `network/fov.py` — capul de câmp vizual |
| 12E | [12E_vit.md](12E_vit.md) | `network/vit.py` + `vit_factory.py` |
| 12F | [12F_utils.md](12F_utils.md) | `utils.py` — I/O EXIF |
| 12G1 | [12G1_losses.md](12G1_losses.md) | `selfsup/losses.py` — pierderi fotometrice |
| 12G2 | [12G2_warping.md](12G2_warping.md) | `selfsup/warping.py` — geometrie diferențiabilă |
| 12G3 | [12G3_pose_net.md](12G3_pose_net.md) | `selfsup/pose_net.py` — estimarea mișcării |
| 12G4 | [12G4_kitti_dataset.md](12G4_kitti_dataset.md) | `selfsup/kitti_dataset.py` — încărcarea datelor |
| 13 | [13_tools.md](13_tools.md) | `tools/` — precompute + export |
| 14 | [14_training.md](14_training.md) | `training/` — bucla de antrenare |

## Lanțul complet (de desenat pe tablă)
```
splits/ + data/  →  kitti_dataset.py  →  [triplet, K, ancoră]
                                              │
        target_depth (1536²) → encoder.py → decoder.py → head → depth
        PoseNet (pose_net.py) → 6-DoF → warping.py (Rodrigues) → warped
                                              │
   losses.py (fotometric) + consistency (train_*.py) → L_total → backward (stabilitate)
                                              │
        export_anchordepth.py → anchordepth.pt → evaluation/ → results/ → teză
```

## Notă importantă de verificat înainte de apărare
Cifra KITTI AbsRel din `results/eval_v15_consistency.json` (**0.0875**) **diferă** de cea din
README/teză (**0.0852**). Comentariile din `scripts/run_v20.sh` confirmă 0.0875. Reconciliază
README ↔ teză ↔ JSON ↔ ce spui la comisie înainte de susținere. Vezi [09_results.md](09_results.md).
