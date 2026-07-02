# 12G2. `src/depth_pro/selfsup/warping.py` (184 linii) — geometria diferențiabilă

Răspunde: „dacă știu adâncimea cadrului curent și mișcarea camerei, cum reconstruiesc cadrul curent
din vecin?". Totul **diferențiabil** (gradientul curge spre adâncime și poză). 4 componente.

**Intuiția fizică:** ridic pixelul în 3D cu adâncimea (backproject), îl mut în sistemul camerei vecine
cu poza (transform), îl proiectez în imaginea vecină (project), culeg culoarea de acolo (sample).

## G2a. `BackprojectDepth` (l.15-55) — pixel 2D → punct 3D
Constructor (l.18-34): precalculează grila omogenă `[u,v,1]` a tuturor pixelilor (`register_buffer` =
constantă geometrică, nu parametru).
forward (l.36-55): **P₃D = D(u,v) · K⁻¹ · [u,v,1]ᵀ**
```python
cam_points = inv_K[:3,:3] @ pix_coords    # K⁻¹·[u,v,1] → direcția razei
cam_points = depth * cam_points           # × adâncime → poziția pe rază (3D)
cam_points = cat([cam_points, ones], 1)   # → omogen 4D
```

## G2b. `Project3D` (l.58-94) — punct 3D → pixel în vecin
```python
P = K @ T @ points_3d                     # [81] mută în cadrul sursă (T) + proiectează (K)
pix_coords = P[:2] / (P[2] + 1e-7)        # [84] împarte la z = proiecție perspectivă
norm_u = pix/(W-1)*2 - 1; norm_v = ...    # [90-91] normalizare în [-1,1] pentru grid_sample
```

## G2c. `axis_angle_to_matrix` (l.97-124) — formula Rodrigues
PoseNet dă rotația ca vector axis-angle (axa × unghiul); o convertim în matrice 3×3:
```python
angle = axis_angle.norm()                 # θ
axis  = axis_angle / θ                     # axa unitară
K = [[0,-az,ay],[az,0,-ax],[-ay,ax,0]]    # matricea skew-simetrică
R = I + sin(θ)·K + (1-cos(θ))·K²          # RODRIGUES
```
Reprezentare diferențiabilă, fără singularități (mai bună decât unghiurile Euler).

## G2d. `pose_vec_to_matrix` (l.127-144) — asamblarea 4×4
```python
T[:3,:3] = R; T[:3,3] = translation; T[3,3] = 1.0   # [R|t; 0 0 0 1]
```

## G2e. `Warper` (l.147-184) — totul împreună
```python
cam_points = backproject(depth, inv_K)            # 1. pixel → 3D
pix_coords = project(cam_points, K, T)            # 2. 3D → pixel în vecin
warped = F.grid_sample(source_img, pix_coords,    # 3. culege culoarea (DIFERENȚIABIL)
                       mode="bilinear", padding_mode="border", align_corners=True)
```
`grid_sample` cu interpolare bilineară = diferențiabilă → gradientul curge spre adâncime și poză.
`padding_mode="border"` = pixelii din afara imaginii iau valoarea de margine.

## Lanțul complet
```
PoseNet → (axisangle, translation) → pose_vec_to_matrix → T
depth + inv_K + K + T → Warper → warped → photometric_loss → semnal de antrenare
```

## De spus la comisie
„`warping.py` face reconstrucția geometrică diferențiabilă: ridic fiecare pixel în 3D cu adâncimea și
inv_K, îl transform în cadrul vecin cu poza de la PoseNet (axis-angle → matrice prin Rodrigues), îl
reproiectez și eșantionez cu grid_sample bilinear. Totul diferențiabil, deci eroarea fotometrică
propagă gradient în adâncime și poză. Operații out-of-place ca să nu rup autograd."
