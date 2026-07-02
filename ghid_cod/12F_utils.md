# 12F. `src/depth_pro/utils.py` (112 linii) — utilitare I/O (scurt)

Fișier auxiliar (cod Apple), folosit doar de CLI — **nu** de pipeline-ul tău de antrenare.

## Trei funcții
- **`extract_exif`** (l.16-39): citește metadatele EXIF dintr-o imagine.
- **`fpx_from_f35`** (l.42-44): convertește focala din „mm echivalent 35mm" în pixeli:
  `f_px = f_mm · √(W²+H²) / √(36²+24²)` (36×24 = cadru film 35mm).
- **`load_rgb`** (l.47-112): încarcă o imagine (inclusiv HEIC de pe iPhone), o rotește după
  orientarea EXIF, scoate alpha, extrage focala din EXIF dacă există.

## Relevanță pentru tine: minimă
Notebook-ul și training-ul tău încarcă imaginile direct cu `PIL.Image.open`, nu prin asta. E moștenit
de la Apple pentru demo-ul CLI cu poze de telefon. Îl menționezi doar dacă te întreabă „de unde ia
focala dintr-o poză normală" → din EXIF, prin `fpx_from_f35`.
