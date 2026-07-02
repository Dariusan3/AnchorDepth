# 2. `data/`

## Ce este
Folder mic, de intrări auxiliare. Nu conține cod, ci câteva fișiere de care depinde codul:
două imagini exemplu și o listă de indici.

## Conținutul, fișier cu fișier

### `eigen_test_indices.json` — lista de selecție (singurul cu rol „științific")
- Listă de **654 de numere întregi**: `[0, 1, 8, 13, ...1448]`.
- Indicii imaginilor din split-ul de test Eigen KITTI **care au ground-truth LiDAR valid**.
  Din ~697 imagini Eigen, doar acest subset e folosit la validarea cu GT.
- **Unde:** `training/train_nyu_lora.py:417` îl încarcă pentru a ști pe ce imagini să calculeze
  `abs_rel`/`delta1` în timpul antrenării.
- **De ce fișier separat:** ca să nu rescanezi tot dataset-ul la fiecare epocă (precompute).

### `example.jpg` (2.2 MB) — imaginea demo implicită
Imaginea folosită de CLI-ul Depth Pro: `src/depth_pro/cli/run.py:129` o are ca
`default="./data/example.jpg"`. Test rapid „merge instalarea?".

### `depth-pro-teaser.jpg` (256 KB) — imaginea de prezentare
Teaser-ul original Depth Pro (Apple). Pur ilustrativ, nefolosit în cod.

## De spus la comisie
„`data/` ține intrările auxiliare: o imagine demo pentru testul CLI și `eigen_test_indices.json`
— lista precalculată a celor 654 de imagini KITTI cu GT LiDAR valid, folosită la validare, ca să
nu rescanez dataset-ul la fiecare epocă."

> Dataset-urile mari reale (KITTI raw ~175 GB, NYU) NU stau aici — sunt sub `datasets/` (gitignored).
