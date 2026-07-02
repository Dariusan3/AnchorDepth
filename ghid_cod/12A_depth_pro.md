# 12A. `src/depth_pro/depth_pro.py` (298 linii) — orchestratorul

Dirijorul: definește configurarea, asamblează componentele, conține **forward pass-ul**.

## A1. `DepthProConfig` (l.26-46)
Dataclass de configurare. `DEFAULT_MONODEPTH_CONFIG_DICT` (l.39): ambii encoderi = **`dinov2l16_384`**
(ViT-Large DINOv2, patch 16, rezoluție 384). Config-ul pe care îl reconstruiește notebook-ul (cu
`checkpoint_uri=None`).

## A2. `create_model_and_transforms` (l.72-151) — fabrica
Funcția apelată de tot codul (notebook, evaluare, training):
- **l.90-99** construiește 3 ViT-uri: patch (detaliu local), image (context global), fov (opțional).
- **l.101-109** `DepthProEncoder` cu `hook_block_ids` = `[5,11,17,23]` (de unde trage feature-uri).
- **l.110-113** decoderul fuzionează cele 5 nivele la `dim_decoder=256`.
- **l.114-120** modelul cu `last_dims=(32,1)` (ultim canal = adâncimea inversă).
- **l.125-132** `Normalize([0.5]*3,[0.5]*3)` → range **[−1,1]** (ce face notebook-ul manual).
- **l.134-149 încărcarea cu `fc_norm`:** `strict=True`, dar se permite lipsa lui `fc_norm` (capul de
  clasificare DINOv2 nefolosit). De aceea notebook-ul afișează „missing keys: typically fc_norm".

## A3. Clasa `DepthPro` — capul (l.154-211)
`self.head` (l.182-204):
```
Conv2d(256→128, 3×3) → ConvTranspose2d(×2 upsample) → Conv2d(128→32) → ReLU → Conv2d(32→1) → ReLU
```
Ultimul **ReLU** garantează adâncime inversă ≥ 0 (fizic non-negativă).

## A4. `forward()` (l.218-241) — fluxul principal
```python
assert H == img_size and W == img_size          # OBLIGATORIU 1536×1536 (l.231)
encodings = self.encoder(x)                      # 1. imagine → 5 hărți multi-scale
features, features_0 = self.decoder(encodings)   # 2. fuziune
canonical_inverse_depth = self.head(features)    # 3. → adâncime inversă canonică
fov_deg = self.fov.forward(x, features_0.detach())  # 4. FOV din feature DETAȘAT (l.239)
return canonical_inverse_depth, fov_deg
```
**`features_0.detach()`** (l.239): gradientul de la FOV NU intră în decoder → cele două sarcini
decuplate. Output = tuplul `(canonical_inverse_depth, fov_deg)` (ce primește notebook-ul).

## A5. `infer()` (l.243-298) — wrapper de inferență
`@torch.no_grad()`. Redimensionează automat la 1536², apoi:
```python
if f_px is None: f_px = 0.5*W / tan(0.5*deg2rad(fov_deg))   # focala din FOV
inverse_depth = canonical_inverse_depth * (W / f_px)         # de-normalizare la geometria reală
depth = 1.0 / clamp(inverse_depth, 1e-4, 1e4)
```
> Notebook-ul NU folosește `infer()` — refaci manual exact aceste formule în `predict_depth`.

## De spus la comisie
„`depth_pro.py` e orchestratorul: `create_model_and_transforms` asamblează cei doi ViT-Large +
decoder + cap, definește normalizarea [−1,1] și încarcă greutățile cu `strict=True` (excepție
legitimă `fc_norm`). `forward` rulează encoder→decoder→head producând adâncimea inversă canonică,
plus FOV-ul dintr-un feature detașat. `infer` adaugă redimensionarea și conversia în metri."
