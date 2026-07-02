# 12B. `src/depth_pro/network/encoder.py` (332 linii) — encoderul multi-scale

**Problema:** un ViT lucrează la 384×384, dar adâncimea precisă cere 1536×1536. Soluția: spargi
imaginea mare în patch-uri de 384, le treci pe toate prin ViT, le lipești la loc. Plus o ramură globală.

4 pași: (1) piramidă → (2) spargere în patch-uri suprapuse → (3) encodare batch → (4) reasamblare.

## B1. Constructorul (l.46-138)
- **l.56-58 `out_size`** = 384/16 = **24** (fiecare patch → grilă 24×24 token-uri).
- **l.60-93 `_create_project_upsample_block`** — mini-rețele: proiecție `Conv 1×1` (schimbă canale) +
  `N × ConvTranspose2d(2×2)` (dublează rezoluția).
- **l.95-113** 5 ramuri de upsampling (latent0 ×8, latent1 ×4, upsample0/1/2 ×2).
- **l.115-130** ramura globală: `upsample_lowres` + `fuse_lowres` (combină global+local).
- **l.132-144 hook-urile** — `register_forward_hook` „spionează" ieșirea blocurilor timpurii (5, 11)
  ale ViT și o salvează. Blocurile timpurii păstrează detaliu spațial fin (înainte ca atenția să
  amestece global).

## B2. Metodele-ajutor
- **`img_size`** (l.146-149) = 384×4 = **1536** (de aici `assert`-ul din `DepthPro.forward`).
- **`_create_pyramid`** (l.151-168): x0=1536, x1=768 (×0.5), x2=384 (×0.25).
- **`split`** (l.170-188): fereastră 384 cu suprapunere → 5×5=25 patch-uri (la 1536, overlap 0.25).
  Suprapunere = evită cusături la lipire. Patch-urile se concatenează pe dimensiunea batch.
- **`merge`** (l.190-217): operația inversă; **l.202-209** taie `padding`(3px) de pe marginile
  interioare suprapuse (anti-cusătură).
- **`reshape_feature`** (l.219-231): aruncă token-ul CLS, reface secvența 1D în hartă 2D (B,C,H,W).

## B3. `forward` (l.233-332)
```python
x0, x1, x2 = pyramid(x)                              # 1536, 768, 384
x0_patches = split(x0, 0.25)  # 5×5 = 25
x1_patches = split(x1, 0.5)   # 3×3 = 9
x2_patches = x2               # 1×1 = 1
x_pyramid = cat(...)          # BATCH = 35 (=25+9+1) ← numărul magic

x_pyramid_encodings = self.patch_encoder(x_pyramid)  # UN singur forward batch (eficiență)
# hook-urile au capturat automat feature-urile intermediare în timpul ăsta
x_latent0/1 = merge(hook0/1[:batch*25], padding=3)   # feature-uri high-res
x0,x1,x2_enc = split(x_pyramid_encodings, [25,9,1])
x0_features = merge(x0_enc, padding=3)   # → 96×96
x1_features = merge(x1_enc, padding=6)   # → 48×48
x2_features = x2_enc                      # → 24×24

x_global = self.image_encoder(x2_patches)            # al DOILEA ViT (context global)
x_global = fuse_lowres(cat(x2_features, x_global))   # fuzionează local+global

return [x_latent0, x_latent1, x0_features, x1_features, x_global]  # 5 hărți (fin→grosier)
```
**Exact aceste 5 hărți** intră în decoder.

## De spus la comisie
„Encoderul rezolvă tensiunea dintre rezoluția fixă mică a ViT (384) și nevoia de detaliu (1536):
piramidă pe 3 nivele, spargere în 25+9+1=35 patch-uri trecute printr-un singur forward batch,
plus feature-uri high-res din blocuri timpurii prin forward hooks. Reasamblează tăind marginile
suprapuse anti-cusătură și fuzionează cu o ramură globală de la al doilea ViT → 5 hărți multi-scale."
