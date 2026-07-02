# 4. `evaluation/`

## Ce este
Folderul care **măsoară cât de bun e modelul**, comparându-l cu ground-truth real, pe 3 dataset-uri.
Aici se produc **toate cifrele din tabelele tezei**. Patru scripturi rulabile din terminal.

```
evaluate_kitti.py       (468 linii) ← cel principal; ceilalți îl refolosesc
evaluate_cityscapes.py  (201)       ← cross-domain stereo
evaluate_make3d.py      (206)       ← cross-domain laser
evaluate_ensemble.py    (200)       ← combinarea zero-shot + fine-tuned
```

---

## 4.1 `evaluate_kitti.py` — scriptul-mamă (6 funcții, în ordinea fluxului)

- **`load_velodyne_points`** (l.34) — citește norul 3D LiDAR din `.bin` (x,y,z).
- **`load_calib`** (l.40) — extrage `P2` (proiecția camerei, conține focala `P2[0,0]`), `R_rect`,
  `Tr_velo` (LiDAR→cameră).
- **`project_velodyne_to_cam`** (l.69) — construiește GT-ul: `pixel = (P2·R_rect·Tr_velo·puncte)`,
  împarte la adâncime, păstrează cel mai apropiat punct per pixel (ocluzii). Hartă **rară** (~5%).
- **`garg_crop`** (l.114) — decupajul standard Garg/Eigen (vertical 40.8–99.2%, orizontal 3.6–96.4%);
  aplicat identic la GT și predicție → comparație corectă cu literatura.
- **`compute_metrics`** (l.124) — inima:
  1. mască `1e-3 < gt < 80` m;
  2. **median scaling** (l.136): `scale = median(gt)/median(pred)`, `pred *= scale`;
  3. cele 7 metrici + `scale`.
- **`load_model`** (l.156) — reconstruiește structura LoRA (`apply_lora_to_encoder`), apoi încarcă
  greutățile filtrate pe formă (`strict=False`). Exact problema gestionată și de notebook.
- **`evaluate`** (l.191) — bucla principală: imagine → GT LiDAR → Garg crop → inferență 1536² →
  conversie în metri cu **focala reală KITTI** (`use_gt_focal`, l.255, fiindcă FOV nu a fost
  reantrenat) → metrici → mediere.
- **`main`** (l.362) — CLI: încarcă model (sau zero-shot fără `--checkpoint`), rulează, afișează cu
  ținte explicite (AbsRel target < 0.0866), salvează JSON în `results/`, opțional WandB.

### Metricile (formule)
Cu `d` = predicție scalată, `d*` = GT, `N` pixeli valizi:
```
AbsRel  = mean(|d−d*| / d*)              (↓)  principala
SqRel   = mean((d−d*)² / d*)             (↓)
RMSE    = sqrt(mean((d−d*)²))            (↓)  metri
RMSElog = sqrt(mean((log d − log d*)²))  (↓)
δ_i = max(d/d*, d*/d);  δ<1.25 / 1.25² / 1.25³  (↑)
```

### Median scaling (întrebare sigură)
Self-supervised dă adâncime doar până la un factor de scară. `scale = median(GT)/median(pred)`
per imagine (mediana = robustă la outlieri). Standard în literatură. `mean_scale ≈ 1.0` arată că
modelul e deja aproape metric.

---

## 4.2 `evaluate_cityscapes.py` — cross-domain (stereo)
- `decode_disparity_to_depth` (l.69): `D=(raw−1)/256`, apoi `depth=f·B/D` cu **f=2262.5 px, B=0.209 m**.
- Fără Garg crop, plafon 80m, median scaling. 500 imagini (3 orașe). Varianta câștigătoare: **v20**.

## 4.3 `evaluate_make3d.py` — cross-domain (laser)
- GT din `.mat`, protocol **C1**: `0 < gt < 70 m`. Metrică extra **log10** = `|log10(pred)−log10(gt)|`.
- Imagini portrait, fără rotație. Fără focală KITTI → se bazează pe FOV + median scaling.
- Varianta câștigătoare: **v18** (cel mai mare câștig: AbsRel −24.7%).

## 4.4 `evaluate_ensemble.py` — combinarea a două modele
- `arithmetic_mean=(d_zs+d_v15)/2`, `geometric_mean=√(d_zs·d_v15)` — fără tuning, rezultate legitime.
- `predict_all` cache-uiește predicțiile ca să nu recalculeze la fiecare combinație.

---

## De spus la comisie
„Evaluarea implementează protocoalele standard ca cifrele să fie comparabile direct: KITTI = LiDAR
proiectat + Garg crop + median scaling; Cityscapes = disparitate stereo; Make3D = protocol C1 cu
log10. `evaluate_kitti.py` e nucleul; ceilalți refolosesc încărcarea modelului. Cheia tehnică: la
încărcare reconstruiesc LoRA înainte de greutăți, și folosesc focala reală pentru modelele
fine-tuned (FOV nu a fost reantrenat)."
