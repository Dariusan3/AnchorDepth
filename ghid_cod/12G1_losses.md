# 12G1. `src/depth_pro/selfsup/losses.py` (156 linii) — pierderile fotometrice

Inima semnalului self-supervised. Implementează pierderile Monodepth2 (Godard 2019). 4 funcții.

## `ssim` (l.16-39) — similaritatea structurală
Mai robust decât diferența brută (tolerează iluminarea):
```python
mu_x = avg_pool2d(x, 3,1,1)              # media locală (luminozitate)
sigma_x = avg_pool2d(x²) - mu_x²         # varianța (contrast)
sigma_xy = avg_pool2d(x·y) - mu_x·mu_y   # covarianța (structură)
ssim_map = ((2μxμy+C1)(2σxy+C2)) / ((μx²+μy²+C1)(σx+σy+C2))
return clamp((1-ssim_map)/2, 0, 1).mean(canale)   # PIERDERE: 1−SSIM, mic = bun
```

## `photometric_loss` (l.42-57) — SSIM + L1
```python
return 0.85·ssim_loss + 0.15·l1_loss     # α=0.85 din Monodepth2
```
`pred` = imaginea reconstruită (warped), `target` = imaginea reală curentă.

## `smooth_loss` (l.60-84) — netezimea edge-aware
```python
grad_disp_x *= exp(-grad_img_x)          # pondere MICĂ unde imaginea are margine
grad_disp_y *= exp(-grad_img_y)
return grad_disp_x.mean() + grad_disp_y.mean()
```
Adâncime netedă în zone plate, dar **salturi permise la marginile obiectelor** (unde `grad_img` mare
→ `exp(-mare)≈0` → penalizarea dispare). Operează pe disparitate (1/depth).

## `compute_selfsup_loss` (l.87-156) — orchestratorul
1. **l.112-114** pierdere fotometrică per sursă (t−1, t+1).
2. **l.116-118 minimum reprojection:** `min` per pixel peste cele 2 surse → tratează **ocluziile**
   (alege sursa unde reconstrucția reușește). NU media.
3. **l.120-134 auto-masking:** compară cu „identity loss" (sursa ne-warpată); maschează pixelii unde
   warping-ul NU îmbunătățește (obiecte fără paralaxă). + zgomot 1e-5 ca să rupă egalitățile.
   **În train_kitti `auto_mask=False`** → practic `photo_loss = min_reproj.mean()` (l.136).
4. **l.139-147 smoothness anti-NaN:** FP32 forțat + `nan_to_num`; `mean_disp` = normalizare la medie
   (trucul Monodepth2 contra „depth contraction"); plasă de siguranță dacă `s_loss` non-finit.
5. **l.149 total:** `photo_loss + smoothness_weight·s_loss` (λ_smooth = 1e-3).
   Acesta e `losses["total"]` la care training-ul adaugă `+ λ·cons_loss`.

## De spus la comisie
„`losses.py` implementează pierderile Monodepth2: SSIM+L1 pentru reconstrucție, minimum reprojection
per pixel pentru ocluzii, auto-masking pentru pixelii fără paralaxă, smoothness edge-aware pe
disparitate normalizată la medie. Partea sensibilă rulează în FP32 cu nan_to_num și plase de
siguranță — parte din rețeta de stabilitate numerică."
