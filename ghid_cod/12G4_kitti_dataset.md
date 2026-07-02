# 12G4. `src/depth_pro/selfsup/kitti_dataset.py` (325 linii) — încărcarea datelor

Transformă fișierele KITTI brute în tensorii de care au nevoie modelul, PoseNet și pierderile.
Două clase: `KITTIRawDataset` (antrenare) și `KITTIEigenTestDataset` (evaluare).

## G4a. `KITTIRawDataset`

### Constructorul (l.30-92)
- **l.34-35 cele două rezoluții (conceptul-cheie):**
  ```python
  depth_size = (1536, 1536)   # intrarea Depth Pro (cerută de piramidă)
  pose_size  = (640, 192)     # PoseNet + pierderea fotometrică (warping la 1536² ar exploda VRAM)
  ```
- **l.37** `frame_ids = (-1, 0, 1)` — tripletul (anterior, curent, următor).
- **l.62-74** parsare split + `stride` (subeșantionează video 10Hz → mai puțină redundanță, paralaxă mai mare).
- **l.77-79** dicționare opționale: `vggt_poses` (înlocuiește PoseNet), `zeroshot_depths` (ancora).
- **l.87-89** color jitter; **l.91-92** cache de calibrare.

### `_get_intrinsics` (l.107-136)
Citește `P_rect` din `calib_cam_to_cam.txt`, extrage K 3×3. **Focala reală** folosită în loc de FOV.

### `__getitem__` (l.142-242)
- **l.159-168** încarcă tripletul cu fallback la cadrul curent dacă vecinul lipsește (marginile secvenței).
- **l.170-189 color augmentation IDENTICĂ pe toate 3 cadrele** — critic: pierderea fotometrică
  presupune **brightness consistency**; culori diferite între cadre ar falsifica semnalul.
- **l.191-193** horizontal flip (dezactivat cu poze VGGT).
- **l.197-216** redimensionare la 2 rezoluții: imaginile de pose în [0,1], `target_depth` normalizat [−1,1].
- **l.218-227 scalarea intrinsics-urilor:**
  ```python
  K_scaled[0,:] *= pose_w/orig_w   # scalează fx, cx (imaginea a fost redimensionată)
  K_scaled[1,:] *= pose_h/orig_h
  if do_flip: K_scaled[0,2] = pose_w - K_scaled[0,2]   # oglindește cx
  inv_K = inv(K_scaled)
  ```
- **l.229-240** atașează poze VGGT / ancora zero-shot; **la flip oglindește și ancora** (l.238-239)
  ca să rămână aliniată geometric.

## G4b. `KITTIEigenTestDataset` (l.245-325) — evaluare
Mult mai simplu: o singură imagine (nu triplet) + GT LiDAR.
- **l.298-316** redimensionează 1536², normalizează, returnează `image` + `orig_size` + `index`.
- **l.319-323** GT: `gt_depth = png / 256.0` (KITTI stochează adâncimea ×256 în PNG 16-biți).

## De spus la comisie
„Dataset-ul produce un triplet la două rezoluții — 1536² pentru encoder și o rezoluție mică pentru
PoseNet și pierdere, din buget de memorie. Aplic augmentarea de culoare identic pe toate 3 cadrele
(ipoteza de brightness consistency) și scalez intrinsics-urile la noua rezoluție, inclusiv oglindirea
principal point-ului la flip. La flip oglindesc și ancora zero-shot ca să rămână aliniată."
