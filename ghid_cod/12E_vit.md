# 12E. `network/vit.py` + `vit_factory.py` (123 + 124 linii) — construcția ViT

Construiesc encoderul ViT-Large pornind de la un model preantrenat din `timm` și îl **adaptează**.
`vit_factory.py` = rețeta, `vit.py` = uneltele.

## E1. `vit_factory.py` — config + asamblare
`VIT_CONFIG_DICT["dinov2l16_384"]` (l.53-65):
```python
embed_dim=1024                          # ViT-Large
encoder_feature_layer_ids=[5,11,17,23]  # = hook_block_ids din encoder.py
encoder_feature_dims=[256,512,1024,1024]
img_size=384, patch_size=16             # ce VREM (Depth Pro)
timm_preset="vit_large_patch14_dinov2"
timm_img_size=518, timm_patch_size=14   # ce ARE modelul original
```
**Observația cheie:** original DINOv2 = patch 14, imagine 518. Depth Pro vrea patch 16, imagine 384.
De aici funcțiile de „resize".

`create_vit` (l.68-124):
```python
model = timm.create_model("vit_large_patch14_dinov2", pretrained=False, dynamic_img_size=True)
model = make_vit_b16_backbone(model, ...)
if patch_size != timm_patch_size: resize_patch_embed(model, (16,16))   # 14→16
if img_size != timm_img_size: resize_vit(model, (384,384))             # 518→384
```
> `use_pretrained=False` în pipeline-ul tău — ViT creat doar ca **schelet**; suprascrii tot cu
> checkpoint-ul Depth Pro complet.

## E2. `vit.py` — uneltele
- **`make_vit_b16_backbone`** (l.13-35): înfășoară modelul și, crucial, **l.33**
  `model.forward = model.forward_features` → returnează feature-uri spațiale, nu logit-uri de clasificare.
- **`resize_patch_embed`** (l.70-123): repară 14→16 prin **interpolare bicubică** a ponderilor
  conv-ului de patch embedding (l.85-90), cu rescalare de magnitudine (l.91-93). Reciclează ponderile
  preantrenate la altă mărime.
- **`resize_vit`** (l.51-67): re-eșantionează **position embeddings** (vectorii de poziție) la noua
  grilă prin `resample_abs_pos_embed` (l.58-64).
- **`forward_features_eva_fixed`** (l.38-48): variantă de forward pentru EVA02 (NU folosită la tine —
  ești pe DINOv2). Conține bucla cu `checkpoint(blk, ...)` = **gradient checkpointing** (recalculează
  activările în backward → economie VRAM).

## De spus la comisie
„Aceste fișiere construiesc ViT-Large DINOv2 din timm și-l adaptează la Depth Pro: redirectez
forward-ul să dea feature-uri spațiale, și re-eșantionez prin interpolare bicubică kernelul de patch
embedding (14→16) și position embeddings (518→384), reutilizând ponderile preantrenate la rezoluția
cerută. Tot aici e suportul pentru gradient checkpointing, esențial pentru cei 12GB."
