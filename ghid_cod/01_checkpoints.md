# 1. `checkpoints/`

## Ce este
Folderul unde antrenarea **salvează rezultatele** — greutățile modelului (`*.pt`) și jurnalele
de antrenare (`*.json`). Nu conține cod, ci **artefacte produse de cod**.

## Greutățile modelului (`*.pt`) — lipsesc intenționat (gitignored, ~3.6 GB fiecare)
Convenția de denumire produsă de scripturile de antrenare:

| Tipar nume | Cine îl produce | Ce e |
|---|---|---|
| `depth_pro_lora{suffix}_best.pt` | `training/train_nyu_lora.py` | LoRA pe NYU, epoca cu cel mai bun `abs_rel` |
| `depth_pro_lora{suffix}_final.pt` | `training/train_nyu_lora.py` | LoRA pe NYU, ultima epocă |
| `selfsup_v15/selfsup_best.pt` | `training/train_kitti_selfsup_ms.py` | metoda principală KITTI (cel mai mic `val_photometric`) |
| `anchordepth.pt` | `tools/export_anchordepth.py` | modelul **merged** (LoRA topit), cel publicat pe HuggingFace |

De aceea notebook-ul căuta `checkpoints/anchordepth.pt` — fișierul final exportat. Subdirectoarele
`selfsup_vXX/` sunt gitignored prin regula `checkpoints/*/`.

## Jurnalele de antrenare (`*.json`) — astea sunt în repo
Istoricul numeric al fiecărei rulări; le poți folosi direct ca **grafice în teză**.
- **`training_log_v3.json`** — 50 epoci, LoRA supervizat pe NYU. Câmpuri per epocă:
  - `train_loss` (a scăzut 0.090 → 0.059 = a învățat)
  - `loss_components` — `si_log`, `gradient`, `ssim`, `normal`, `affine`
  - `lr` (crește în warmup), `epoch_time` (~419 s)
  - `abs_rel` / `delta1` doar la epocile de validare (final: abs_rel 0.0933, δ<1.25 0.936)
- `training_log_lora.json` / `_lora_unfreeze2.json` — variante LoRA (a doua dezgheață mai mult).
- `training_log_selfsup.json` — rulare self-supervised (probă scurtă).

## `experiments.pid`
PID-ul procesului de antrenare din fundal (scris de `scripts/run_all_experiments.sh`). Pur tehnic.

## De spus la comisie
„Aici se materializează antrenarea: greutățile (excluse din git, au GB), jurnalele JSON pentru
graficele de convergență, și un PID pentru gestionarea rulărilor. Denumirea codifică varianta
(`lora`, `selfsup_v15`) și criteriul (`best` = cea mai bună validare, `final` = ultima epocă)."
