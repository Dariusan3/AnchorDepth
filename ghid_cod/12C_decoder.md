# 12C. `src/depth_pro/network/decoder.py` (206 linii) — decoderul DPT/FPN

Primește cele 5 hărți multi-rezoluție și le **fuzionează progresiv** (grosier→fin) într-o hartă
densă. Variantă de **DPT** (arXiv 2103.13413), practic un **FPN**. 3 clase.

## C1. `ResidualBlock` (l.96-118) — cărămida de bază
```python
delta_x = self.residual(x)
return x + delta_x          # conexiune reziduală (skip)
```
Învață doar **corecția** `delta_x`, nu transformarea completă. Gradientul curge nestingherit prin
`+x` (rezolvă vanishing gradient) → rețele adânci antrenabile.

## C2. `FeatureFusionBlock2d` (l.121-206) — blocul de fuziune
Constructor (l.124-164): `resnet1` (procesează feature-ul nou), `resnet2` (procesează fuzionatul),
`deconv` (upsample ×2, doar dacă deconv=True), `out_conv` (1×1).

`forward(x0, x1)` (l.166-180):
```python
x = x0                          # feature de la nivelul mai grosier (deja upsamplat)
if x1 is not None:
    x = x + resnet1(x1)         # FUZIUNE prin ADUNARE (nu concatenare) cu feature-ul de la encoder
x = resnet2(x)                  # rafinează
if use_deconv: x = deconv(x)    # ×2 pentru nivelul următor (mai fin)
x = out_conv(x)
```
`_residual_block` (l.182-206): fiecare „resnet" = 2 sub-blocuri `ReLU → Conv3×3`, fără BatchNorm.

## C3. `MultiresConvDecoder` (l.16-93) — orchestratorul
Constructor (l.19-72) creează 2 liste paralele:
- **`self.convs`** — proiecție per nivel la `dim_decoder=256` (nivelul 0: `Conv1×1` sau `Identity`;
  restul: `Conv3×3`).
- **`self.fusions`** — `FeatureFusionBlock2d` per nivel. **`deconv=(i != 0)`** (l.68): toate fac
  upsample ×2, **mai puțin** nivelul 0 (cel mai fin).

`forward(encodings)` (l.74-93) — fuziune progresivă grosier→fin:
```python
features = self.convs[-1](encodings[-1])   # PORNEȘTE de la cel mai GROSIER (global)
lowres_features = features                 # ← salvat pentru capul de FOV!
features = self.fusions[-1](features)
for i in range(num_levels-2, -1, -1):      # urcă spre FIN
    features = self.fusions[i](features, self.convs[i](encodings[i]))
return features, lowres_features
```
Două output-uri: `features` (→ `self.head` → adâncime) și `lowres_features` (→ `self.fov`, detașat).

## Cum se leagă
```
encoder → [5 hărți fin...grosier] → decoder (fuziune grosier→fin) → (features, lowres_features)
                                          │                                  │
                            head(features) → adâncime        fov(lowres.detach()) → FOV
```

## De spus la comisie
„Decoderul e DPT/FPN: aduce cele 5 hărți la 256 de canale și le fuzionează progresiv de la grosier
(cu context global) spre fin, prin blocuri care adună feature-urile (nu concatenare) și fac upsample
×2 la fiecare nivel. Întoarce feature-ul dens pentru adâncime și feature-ul grosier pentru FOV, care
primește o copie detașată ca să nu perturbe învățarea adâncimii."
