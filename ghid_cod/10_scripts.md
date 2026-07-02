# 10. `scripts/`

## Ce este
**Scripturile bash de automatizare** — rețetele care leagă tot pipeline-ul: descarcă date →
precalculează ancora → antrenează → evaluează. Fiecare `run_vXX.sh` = un experiment complet
reproductibil cu un singur apel.

```
setup/download_kitti_raw.sh     descarcă KITTI raw (~175 GB)
setup/get_pretrained_models.sh  descarcă depth_pro.pt (Apple) în checkpoints/
run_v16..v20.sh                 experimentele individuale
run_after_v11.sh / run_all_experiments.sh  lanțuri de experimente
eval_all_checkpoints.sh         evaluează toate checkpoint-urile
```

## Anatomia unui experiment (4 pași — tiparul comun)
1. Activează conda + `set -e`.
2. Verifică/regenerează ancora zero-shot (`precompute_zeroshot_depths.py`) dacă lipsește cache-ul.
3. Antrenează cu `train_kitti_selfsup_ms.py` și argumentele variantei:
   ```
   v18: --consistency-weight 10 --consistency-mode log --stride 6 --epochs 5
   v20: --consistency-weight 20 (L1, ancoră grea)       --stride 6 --epochs 5
   ```
4. Evaluează imediat, salvează JSON în `results/`.

**Detaliul de aur:** fiecare script are în antet un **comentariu-jurnal** care explică *de ce*
schimbi parametrul față de versiunea anterioară (ex. v18: „log-space optimizează direct RMSElog și
δ<1.25²; v15 a pierdut δ<1.25² cu doar 0.0002"). Raționamentul experimental scris în cod.

## ⚠️ Confirmarea neconcordanței de cifre
`run_v20.sh` spune negru pe alb:
```
AbsRel  < 0.0866  (v15: 0.0875, gap 0.0009)    ← v15 NU bate zero-shot pe AbsRel
δ<1.25³ > 0.98494 (v15: 0.98499 ✓ already won)  ← v15 bate DOAR pe δ<1.25³
```
v18/v20 = încercări de a „întoarce" metricele pierdute la limită. Vezi `09_results.md`.

## `setup/`
- `download_kitti_raw.sh` — KITTI raw (~175 GB).
- `get_pretrained_models.sh` — `depth_pro.pt` (modelul Apple) de pe CDN Apple în `checkpoints/`.
  **Acesta** e fișierul `checkpoints/depth_pro.pt` pentru comparația zero-shot din notebook.

## De spus la comisie
„`scripts/` automatizează pipeline-ul reproductibil: fiecare `run_vXX.sh` verifică ancora, antrenează
cu parametrii exacți și evaluează, cu un apel. Antetele documentează raționamentul fiecărui experiment
— de ce trec de la L1 la log-space, de ce ridic λ — ca deciziile să fie trasabile."
