# 14. `training/` — bucla care leagă tot

Două fișiere: `train_nyu_lora.py` (LoRA supervizat pe NYU, dependența notebook-ului) și
**`train_kitti_selfsup_ms.py`** (metoda principală, 906 linii).

---

## `train_nyu_lora.py` (549 linii) — varianta supervizată indoor + sursa LoRA
Folosit de notebook (importă `LoRALinear`, `apply_lora_to_encoder`).
- **`LoRALinear`** (l.51-83): `y = W·x + (α/r)·(B·A·x)`. Inițializare: `A` Kaiming, `B` **zerouri** →
  la pasul 0 ΔW=0 → modelul = exact zero-shot. `W` și bias înghețate.
- **`apply_lora_to_encoder`** (l.86-118): înlocuiește `attn.qkv` și `attn.proj` în fiecare bloc al
  **ambilor** encoderi → 24×2×2 = **96 de layere**.
- Loss supervizat (are GT NYU): `ScaleInvariantLogLoss` (SI-log) + `GradientMatchingLoss`.
- Salvează `depth_pro_lora{suffix}_best.pt` cu structura LoRA → notebook-ul trebuie să reconstruiască
  LoRA înainte de a încărca.

---

## `train_kitti_selfsup_ms.py` → `train_one_epoch` (l.64-246)

### Setup (l.72-85)
`model.train()`, `pose_net.train()`, **`model.fov.eval()` (l.77-78) — FOV ÎNGHEȚAT**.

### Încărcarea batch-ului (l.89-97)
`target_depth` (B,3,1536,1536) pentru encoder; `target`/`source_±1` (B,3,128,416) pentru loss; `K`, `inv_K`.

### Pas 1 — adâncimea (l.99-113)
```python
with autocast("cuda", dtype=torch.bfloat16):     # l.99 PRECIZIE MIXTĂ bf16
    encodings = model.encoder(target_depth); del encodings   # del = eliberează VRAM
    features, _ = model.decoder(...); raw = model.head(features)
    inv_depth = interpolate(raw, pose_size) * (pose_w/f_px)   # de-normalizare cu focala reală
    inv_depth = nan_to_num(...); relu(...)+1e-6
    depth = 1/clamp(inv_depth, 1e-4, 1e4)
```

### Pas 2 — poza (l.115-128)
```python
if "T_prev" in batch: T_prev, T_next = batch[...]      # poze VGGT precomputate
else:
    pose_vec = pose_net(normalize_imagenet(target), normalize_imagenet(source))  # PoseNet vrea norm. ImageNet!
    T_prev = pose_vec_to_matrix(pose_vec[:,:3], pose_vec[:,3:])
```

### Pas 3 — warping + pierderi (l.130-181)
```python
warped_prev = warper(source_prev, depth, T_prev, K, inv_K)   # warping.py
losses = compute_selfsup_loss(...)                           # losses.py (fotometric + smoothness)
# + consistency loss (l.144-181): L_cons = |d_pred - d_zs| sau |log d_pred - log d_zs|
losses["total"] += consistency_weight * cons_loss
```

### Backward + optimizare (l.183-207) — stabilitatea numerică
```python
loss = losses["total"] / grad_accum_steps      # gradient accumulation
if not isfinite(loss): zero_grad(); continue    # l.186 skip pas non-finit
scaler.scale(loss).backward()
if (step+1) % grad_accum_steps == 0:
    scaler.unscale_(optimizer)
    for param: if not isfinite(grad): grad[~isfinite] = 0   # l.196-198 sanitizare gradienți
    clip_grad_norm_(all_params, max_norm=1.0)               # l.204 clipping
    scaler.step(optimizer); scaler.update(); zero_grad()
```
**4 plase de siguranță** (a 5-a contribuție): skip loss non-finit, zero-uire gradienți NaN, clipping,
gradient accumulation.

### `normalize_imagenet` (l.254-258)
PoseNet (backbone ImageNet) cere normalizarea ImageNet, diferită de cea Depth Pro [−1,1].

---

## `main` — orchestrarea (locațiile)
- Înghețare selectivă (~l.548-590): tot înghețat → LoRA → dezgheață decoder+head → FOV înghețat.
- Gradient checkpointing (~l.593-598) pe ambii encoderi.
- LR discriminative (~l.726-731): LoRA 1e-5, decoder/head 1e-4, PoseNet 1e-4.
- Scheduler (~l.735-737): CosineAnnealingWarmRestarts + warmup liniar 2 epoci.
- Salvare (~l.822-835, 877-887): `selfsup_best.pt` când scade `val_photometric`, cu verificare NaN
  înainte de salvare.

## De spus la comisie
„`train_one_epoch` e locul unde converg toate componentele: adâncime în bfloat16, poza cu PoseNet,
warping, pierdere fotometrică + consistency anchor, backward cu gradient accumulation. Înconjor pasul
cu patru garanții de stabilitate, fiindcă antrenarea unui model fundație de 950M parametri în precizie
mixtă cu LoRA e predispusă la divergență. Capul de FOV rămâne în eval — folosesc focala reală din calibrare."
