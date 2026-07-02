# 12G3. `src/depth_pro/selfsup/pose_net.py` (91 linii) — estimarea mișcării camerei

**Ce face:** primește 2 cadre RGB și prezice cum s-a mișcat camera între ele (6 grade de libertate:
3 rotație + 3 translație). Necesar fiindcă în self-supervised monocular poza nu e cunoscută.

## Constructorul (l.23-58)
- **l.28** backbone **ResNet-18** preantrenat ImageNet.
- **l.30-38 trucul cu 6 canale:**
  ```python
  self.conv1 = Conv2d(6, 64, 7, stride=2, padding=3)   # 6 = 2 cadre RGB concatenate
  self.conv1.weight[:, :3] = original.weight / 2        # copiază ponderile ÷2
  self.conv1.weight[:, 3:6] = original.weight / 2
  ```
  ÷2 ca suma activării să rămână în aceeași magnitudine. Refolosește ImageNet, nu pornește de la zero.
- **l.40-46** corpul ResNet (layer1→layer4, 512 canale).
- **l.48-55** capul de pose: `Conv1×1(512→256) → ReLU → Conv3×3 → ReLU → Conv1×1(→6)`.

## `forward(target, source)` (l.60-91)
```python
x = cat([target, source], dim=1)   # (B, 6, H, W)
x = encoder ResNet → (B, 512, H/32, W/32)
x = pose_head(x) → (B, 6, ...)
x = global_pool(x) → (B, 6, 1, 1)  # media spațială = o poză globală (camera se mișcă rigid)
x = x.view(B, -1)                   # (B, 6)
x = 0.01 * x                        # ← scalare crucială
return x                            # [axis_angle(3), translation(3)]
```
**Scalarea cu 0.01 (l.89):** la inițializare poza ≈ identitate (rotație/translație ~0). De ce: la
primii pași adâncimea e proastă; o poză mare ar produce warp-uri haotice. Pornind de la „camera
aproape nemișcată", antrenarea e stabilă.

Cele 6 numere → `pose_vec_to_matrix` (warping.py) → matricea `T`.

## Legătura cu VGGT
PoseNet antrenat de la zero dă poze proaste la început. Alternativă: `--vggt-poses` îl dezactivează și
folosește poze precomputate de VGGT. Pe KITTI, PoseNet câștigă — dar VGGT e o contribuție metodologică.

## De spus la comisie
„PoseNet e un ResNet-18 modificat să primească 2 cadre concatenate pe canale și să prezică poza 6-DoF.
Inițializez primul conv din ImageNet împărțit la numărul de cadre, și scalez ieșirea cu 0.01 ca poza
inițială să fie aproape identitate, ceea ce stabilizează antrenarea când adâncimea e încă imprecisă."
