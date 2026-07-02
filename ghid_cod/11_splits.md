# 11. `splits/`

## Ce este
**Listele care definesc ce imagini intră în antrenare, validare și test** — împărțirea standard
**Eigen** a KITTI. Fișiere text, dar fundamentale științific: garantează că nu antrenezi și testezi
pe aceleași scene și că folosești exact aceeași împărțire ca lucrările publicate.

## Formatul liniilor
```
2011_09_26/2011_09_26_drive_0002_sync   0000000069   l
└─ secvența (data/drive)                └─ nr. cadru  └─ camera (l=02/stânga, r=03/dreapta)
```
Exact formatul parsat de `evaluate_kitti.py:200-203` și de dataset.

## Cele 5 fișiere
| Fișier | Linii | Rol |
|---|---|---|
| `eigen_train_files.txt` | 39.810 | antrenare self-supervised |
| `eigen_val_files.txt` | 4.424 | validare (monitorizare) |
| `eigen_test_files.txt` | 697 | test (cifrele finale) |
| `kitti_all_sequences.txt` | 60 | toate secvențele |
| `kitti_train_sequences.txt` | 32 | secvențele de antrenare |

## De ce exact aceste numere
- **697** = setul de test Eigen canonic; toată literatura raportează pe exact acestea → comparabil.
  (Din ele, ~654 au GT LiDAR valid — vezi `data/eigen_test_indices.json`.)
- **Split pe secvențe, nu pe cadre:** train și test din **drive-uri diferite** → fără scurgere de
  informație (cadre aproape identice în ambele seturi).
- **Două camere (l/r):** KITTI e stereo; ambele camere ca imagini monoculare separate **dublează**
  datele (de aici 39.810 linii).

## De spus la comisie
„`splits/` conține împărțirea standard Eigen — 39.810 antrenare, 4.424 validare, 697 test. Folosesc
split-ul canonic ca rezultatele să fie comparabile direct cu Monodepth2 și MonoViT, iar împărțirea e
pe secvențe distincte, nu pe cadre, ca să nu existe scurgere între train și test."
