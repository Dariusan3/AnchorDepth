# 13. `tools/` — instrumentele offline

Filosofia: **precalculezi o singură dată ce altfel s-ar repeta**, sau **transformi** modelul pentru livrare.

## 13.1 `precompute_zeroshot_depths.py` (118 linii) — fabricarea ancorei
**Problema:** `d_zs` (zero-shot) e constant (modelul de bază e înghețat); a-l recalcula în buclă =
un al doilea Depth Pro la fiecare pas. **Soluția:** rulezi o dată, salvezi pe disc.

`precompute_split` (l.29-83):
```python
dataset = KITTIRawDataset(..., is_train=False)   # ACELAȘI dataset → indici identici cu antrenarea
for idx in range(len(dataset)):
    img = resize(1536²); inp = normalize(img)     # ACEEAȘI preprocesare ca target_depth
    f_px = K[0,0]                                  # focala REALĂ KITTI
    canonical_inv_depth = model(inp)
    inv_depth = canonical_inv_depth * (orig_w/f_px)  # de-normalizare (la fel ca antrenarea)
    inv_depth = interpolate(pose_size); nan_to_num; relu+1e-6
    depth = 1/clamp(inv_depth)
    depths[idx] = depth.cpu().half()              # FP16 (jumătate de spațiu)
torch.save → zeroshot_depths_train_s6_416x128.pt
```
**Trei detalii de apărat:** (1) același dataset+stride → indicii se potrivesc 1:1 (de aia stride-ul e
în numele fișierului); (2) aceeași focală+conversie → `d_zs` în aceleași unități ca `d_pred`; (3) FP16.

## 13.2 `export_anchordepth.py` (148 linii) — îmbinarea LoRA → modelul de livrare
**Ideea (LoRA e liniar):** `W_nou = W + (α/r)·B·A`, apoi înlocuiești stratul LoRA cu `nn.Linear` simplu.

`merge_lora_into_base` (l.44-67):
```python
for child in model: if isinstance(child, LoRALinear):
    delta = (alpha/rank) * (B @ A)        # corecția LoRA dezvoltată complet
    W_merged = W + delta
    new_linear = nn.Linear(in, out); new_linear.weight = W_merged
    setattr(parent, name, new_linear)     # ÎNLOCUIEȘTE fizic LoRALinear-ul
```
`main` (l.70-144): schelet → `apply_lora_to_encoder` → încarcă checkpoint → **merge** → 2 verificări:
(1) niciun LoRALinear rămas (l.122), (2) niciun NaN/Inf, altfel refuză salvarea (l.129-133). Rezultat:
`anchordepth.pt` (3.6 GB) care se încarcă cu **`strict=True`** fără cod LoRA. Fișierul de pe HuggingFace.
Maparea: v15→KITTI, v18→Make3D, v20→Cityscapes (rank=8, α=8).

## 13.3 + 13.4 (scurt)
- **`precompute_vggt_poses.py`** (199): VGGT offline → poze de calitate ca alternativă la PoseNet.
  `relative_pose`: `T = E_source · inv(E_target)`, inversa rigidă `inv([R|t]) = [Rᵀ | −Rᵀt]`.
- **`generate_results.py`** (436): produce figurile și tabelele din JSON-urile de evaluare. Pur prezentare.

## De spus la comisie
„`precompute_zeroshot_depths` rulează modelul de bază o singură dată și cache-uiește ancora în FP16, cu
exact același dataset, stride și focală ca antrenarea. `export_anchordepth` îmbină LoRA în ponderi —
`W + (α/r)·B·A` — și înlocuiește straturile LoRA cu unele standard, verificând că nu rămâne dependență
LoRA și niciun NaN, producând checkpoint-ul care se încarcă ca un Depth Pro normal."
