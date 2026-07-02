# 7. `legacy/`

## Ce este
**Cod vechi, abandonat** — versiuni anterioare ale scripturilor, păstrate ca arhivă. Verificat:
**niciun fișier activ (`.py`, `.sh`, notebook) nu importă nimic din `legacy/`**. Complet deconectat
de pipeline-ul curent.

## De ce există (oglinda structurii principale)
| `legacy/` conține | corespondentul actual |
|---|---|
| `training/train_nyu.py`, `train_nyu_v3.py`, `train_nyu_lora_se.py` | `training/train_nyu_lora.py` |
| `training/train_kitti_selfsup.py` (fără „_ms") | `training/train_kitti_selfsup_ms.py` |
| `evaluation/evaluate_v3_variants.py`, `evaluate_v5_tta.py`, `evaluate_v6.py`... | `evaluation/evaluate_kitti.py` |
| `experiment_system/run_experiment.py` | `scripts/run_*.sh` |
| `results/eval_results_*.json` | `results/` |

Se citește ca **istoria proiectului**: NYU → LoRA → prima versiune KITTI self-sup → varianta
multi-scale finală (`_ms`). Versiunile v3/v5/v6/„tta" sunt experimente intermediare.

## De ce e VALOROS pentru teză
1. **Demonstrează procesul iterativ** (v3→v5→v6→v15→v18→v20) — nu ai nimerit din prima.
2. **JSON-urile vechi** conțin cifrele experimentelor abandonate → materialul pentru **rezultatul
   negativ documentat** (self-supervision naiv degradează modelul). „Greșelile" fac parte din contribuție.

## De spus la comisie
„`legacy/` e arhiva versiunilor anterioare — nefolosită de pipeline-ul curent (zero importuri). O
păstrez pentru trasabilitate: documentează drumul de la primele încercări supervizate pe NYU până la
metoda finală multi-scale pe KITTI, inclusiv variantele eșuate care susțin rezultatul negativ."

## Recomandare
Fie o lași și o menționezi explicit în teză ca „arhivă de dezvoltare", fie o muți pe o ramură git
separată (`git branch archive-legacy`) pentru un `main` curat.
