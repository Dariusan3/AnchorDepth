# 12D. `src/depth_pro/network/fov.py` (82 linii) — capul de câmp vizual

**Ce face:** estimează unghiul **câmpului vizual (FOV)** în grade, dintr-o singură imagine. FOV-ul →
focala `f_px` → factorul care transformă adâncimea canonică (relativă) în **metri**.

## Constructorul (l.14-54) — CNN care reduce la 1 număr
```
Conv 3×3 stride 2: 256→128, 24×24   (fov_head0)
ReLU
Conv 3×3 stride 2: 128→64,  12×12
ReLU
Conv 3×3 stride 2: 64→32,   6×6
ReLU
Conv 6×6 stride 1: 32→1,    1×1     ← un singur scalar = FOV
```
Fiecare `stride 2` înjumătățește (24→12→6); ultimul `Conv 6×6` colapsează grila într-un număr.
**l.47-53:** cu `fov_encoder` (cazul tău, default-ul are `dinov2l16_384`) un al treilea ViT dă
capacitate; fără el, lucrează doar pe feature-ul decoderului.

## `forward(x, lowres_feature)` (l.56-82)
```python
if hasattr(self, "encoder"):                       # cu ViT dedicat
    x = interpolate(x, 0.25)                        # imaginea la 384
    x = self.encoder(x)[:, 1:].permute(0,2,1)       # ViT FOV, aruncă CLS
    lowres_feature = self.downsample(lowres_feature)
    x = x.reshape_as(lowres_feature) + lowres_feature  # FUZIONEAZĂ imagine + feature decoder
else:
    x = lowres_feature
return self.head(x)                                 # → FOV scalar
```
`lowres_feature` vine **detașat** din `DepthPro.forward` → gradientul FOV nu strică decoderul.

## De ce e CRUCIAL pentru teză
**Pe KITTI/Cityscapes capul e ÎNGHEȚAT și NU îl folosești.** În loc de FOV-ul prezis, folosești
**focala reală din calibrare** (`P2[0,0]`):
- `evaluate_kitti.py:255-256`: `if use_gt_focal: f_px = P2[0,0]`.
- În training: `model.fov.eval()` (capul rămâne înghețat).

**Motiv (răspuns comisie):** pe scene de condus KITTI estimarea FOV e nesigură; KITTI vine cu
calibrare exactă → focala reală e mai precisă. Doar pe **Make3D** (fără calibrare) te bazezi pe
FOV + median scaling.

## De spus la comisie
„Capul de FOV estimează câmpul vizual din care derivă focala și deci scara metrică. Îl îngheț pe
KITTI și Cityscapes și folosesc focala reală din calibrare, fiindcă pe scene de condus FOV-ul prezis
e nesigur; doar pe Make3D, fără calibrare, mă bazez pe el plus median scaling."
