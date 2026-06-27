# Ghid de prezentare licență — AnchorDepth

> Document de studiu pentru susținerea lucrării. Acoperă tot codul sursă, cap-coadă,
> aliniat cu capitolele tezei. Citește-l de sus în jos: fiecare secțiune se bazează
> pe cea anterioară. La final ai o listă de întrebări probabile + răspunsuri.

---

## 0. Mesajul de 30 de secunde (învață-l pe de rost)

AnchorDepth adaptează modelul fundație **Depth Pro** (Apple, 952M parametri) pe domeniul
**KITTI** (condus auto, în aer liber), **fără ground-truth de adâncime la antrenare**
(self-supervised). Antrenez doar **3.6% din parametri** (LoRA rang 8 pe 96 de layere de
atenție + decoder/head + un PoseNet de la zero), ca să încapă pe un GPU de consum de 12GB.
Semnalul de antrenare vine din două surse: (1) **pierderea fotometrică** stil Monodepth2
(reconstruiesc cadrul curent din cadrele vecine folosind adâncimea + poza camerei) și
(2) un **„consistency anchor"** — o ancoră care ține modelul aproape de propriile predicții
zero-shot, ca să nu „uite" cunoștințele excelente ale modelului de bază. Rezultat:
îmbunătățesc Depth Pro pe KITTI, Cityscapes și Make3D, cu câștiguri de până la **−24.7%**
AbsRel cross-domain pe Make3D.

**Originea numelui:** „Anchor" = ancora de consistență care leagă modelul fine-tuned de
baseline-ul zero-shot.

---

## 1. Problema și motivația (Cap. 1)

### 1.1 Estimarea adâncimii monoculare
Din **o singură imagine RGB**, vrem o hartă de adâncime (distanța fiecărui pixel față de
cameră). E o problemă **prost-pusă geometric**: o infinitate de scene 3D pot produce aceeași
poză 2D. Senzorii dedicați (LiDAR, stereo) sunt scumpi → e de dorit doar din RGB.

### 1.2 Adâncime metrică vs. relativă (întrebare sigură la comisie)
- **Metrică** = în metri reali (ex. „mașina e la 7.3 m").
- **Relativă** = corectă doar până la un factor de scară necunoscut (proporțiile sunt bune,
  dar nu valorile absolute).
- Depth Pro produce adâncime **metrică**. Antrenarea self-supervised reintroduce o
  ambiguitate de scară, pe care o corectăm la evaluare prin **median scaling** (vezi §9.3).

### 1.3 De ce self-supervised
Învățarea supervizată cere etichete de adâncime (scumpe, rare). Metodele self-supervised
(Zhou 2017, **Monodepth2** / Godard 2019) învață din **consistența temporală** a unui video:
dacă prezici corect adâncimea + mișcarea camerei, poți reconstrui un cadru din vecinul lui;
eroarea de reconstrucție devine semnalul de antrenare — **fără etichete**.

### 1.4 De ce e relevantă adaptarea pe GPU de 12GB
Depth Pro are 952M parametri. Fine-tuning complet ar cere gradient + stările optimizatorului
AdamW (2 momente per parametru) prin tot modelul → depășește mult cei 12GB ai unui RTX 4070 Ti.
Soluția: **PEFT / LoRA** — îngheț modelul, antrenez doar matrici mici de rang scăzut.
Relevanță: democratizarea adaptării modelelor fundație pe hardware accesibil.

### 1.5 Golul din literatură (contribuția de poziționare)
Există (a) metode self-supervised mici antrenate de la zero (Monodepth2, MonoViT) și
(b) modele fundație supervizate (Depth Pro, Depth Anything). **Nimeni nu le-a combinat.**
Întrebarea nouă: *poate un model fundație mare să fie adaptat self-supervised, cu o fracțiune
din parametri, pe un GPU de consum?* Observație tare: **Depth Pro zero-shot bate deja toate
metodele self-supervised publicate pe KITTI** (AbsRel 0.0866 vs. 0.099 MonoViT). Deci
întrebarea reală: poate adaptarea extrage semnal *în plus* fără a strica un baseline deja foarte bun.

### 1.6 Contribuțiile declarate (lista exactă)
1. Prima evaluare publicată a Depth Pro zero-shot pe KITTI Eigen → nou SOTA.
2. **Rezultat negativ documentat:** self-supervision fotometric naiv **degradează** modelul
   fundație (AbsRel ×5, de la 0.087 la ~0.46), deși loss-ul de antrenare scade.
3. **Adaptarea ancorată prin consistență** (rezultatul pozitiv principal) — îmbunătățește
   KITTI, Cityscapes, Make3D.
4. **Supervizare de pose offline cu VGGT** ca alternativă la PoseNet.
5. **Rețetă reproductibilă de stabilitate numerică** pentru LoRA + precizie mixtă la scară
   de model fundație.

---

## 2. Backbone-ul: arhitectura Depth Pro (`src/depth_pro/`)

> Acesta NU e contribuția ta — e modelul de bază (Apple). Dar trebuie să-l înțelegi ca să
> explici unde intervii cu LoRA. Cod: `src/depth_pro/depth_pro.py`, `network/*`.

### 2.1 Ce face Depth Pro
Dintr-o singură imagine produce **adâncime metrică** rapid (<1s). Rezolvă simultan:
1. **Forma scenei** → o hartă de **adâncime inversă canonică** (`canonical_inverse_depth`,
   adică 1/depth normalizată la o focală de referință).
2. **Câmpul vizual (FOV)** → un cap separat estimează `fov_deg`, din care se deduce focala
   `f_px`. Focala e exact factorul care transformă adâncimea canonică în metri.

> Idee-cheie: adâncimea metrică nu poate fi recuperată dintr-o singură imagine fără scară,
> iar scara e dată de focală. Depth Pro o estimează intern (din FOV) când nu o ai din EXIF.

### 2.2 Fluxul forward (de la imagine la output)
- **Rezoluție fixă 1536×1536** (= 384 × 4; backbone-ul ViT lucrează la 384).
- **Doi encoderi ViT-Large DINOv2** (`embed_dim=1024`, 24 de blocuri fiecare):
  - `patch_encoder` — procesează patch-uri locale la mai multe scale (detaliu fin).
  - `image_encoder` — procesează imaginea globală la rezoluție mică (context de scenă).
- **Piramidă pe 3 nivele** (`encoder.py:_create_pyramid`): 1536 / 768 / 384.
- **Spargere în patch-uri 384×384 suprapuse** (`split`): 25 + 9 + 1 = **35 de patch-uri**
  procesate într-un singur batch de către `patch_encoder`.
- Fiecare patch → grilă de **24×24 token-uri** (384/16). `reshape_feature` scoate token-ul
  CLS și reface harta 2D.
- **Hook-uri** la blocurile timpurii (`[5, 11, 17, 23]`) capturează feature-uri de înaltă
  rezoluție pentru detaliu.
- `merge` reasamblează patch-urile suprapuse în hărți 96×96 / 48×48 / 24×24, tăind marginile
  ca să evite cusăturile.
- Output encoder: **5 hărți multi-rezoluție** cu dimensiuni de canal [256, 256, 512, 1024, 1024].

### 2.3 Decoder + depth head
- **MultiresConvDecoder** (`decoder.py`) = variantă **DPT/FPN**: fuzionează cele 5 nivele
  de la grosier la fin, totul la `dim_decoder=256`. Fiecare nivel are un `FeatureFusionBlock2d`
  (2 blocuri reziduale + upsampling ×2).
- **Depth head** (`depth_pro.py`): Conv→ConvTranspose(×2)→Conv→ReLU→Conv→ReLU. Cele două ReLU
  finale garantează adâncime inversă **non-negativă**. Output = `canonical_inverse_depth`.

### 2.4 Capul de FOV (`fov.py`)
Estimează `fov_deg` dintr-un cap convoluțional ce reduce 24×24 → 1×1. Primește `lowres_feature`
de la decoder **detașat** (`detach()`) → gradientul de la FOV nu strică decoderul. **Important
pentru teză:** pe KITTI predicțiile FOV sunt nesigure, deci la antrenare/eval folosim **focala
reală din calibrare** (P2[0,0]) și înghețăm capul de FOV.

### 2.5 Formula finală (de știut pe de rost)
```
f_px       = 0.5 * 1536 / tan(0.5 * fov_rad)      # din FOV, dacă nu ai focala reală
inv_depth  = canonical_inv_depth * (1536 / f_px)   # de-normalizare la geometria reală
depth_[m]  = 1 / clamp(inv_depth, 1e-4, 1e4)
```
Dacă ai focala reală (EXIF sau calibrare KITTI), o folosești în locul celei din FOV.

### 2.6 Construcția modelului
`create_model_and_transforms(config)` (`depth_pro.py`): creează encoderii + decoder + heads,
definește preprocesarea (`Normalize` cu mean=std=0.5 → range [−1,1]) și încarcă checkpoint-ul.
`set_grad_checkpointing(True)` = economie de VRAM prin **recalcularea activărilor** în backward
(esențial pentru fine-tuning pe 12GB).

---

## 3. LoRA — adaptarea eficientă (inima tehnică)

> Cod: `training/train_nyu_lora.py` (clasa `LoRALinear`, funcția `apply_lora_to_encoder`).
> Aceleași concepte sunt reluate în training-ul KITTI și la export.

### 3.1 Ideea
Un strat liniar calculează `y = W·x`. A antrena tot `W` = milioane de valori. LoRA observă că
**actualizarea** ΔW din fine-tuning are rang intrinsec mic, deci o aproximăm:
```
ΔW ≈ B · A         A ∈ R^(r×in),  B ∈ R^(out×r),  rang r = 8
```
Forward-ul devine:
```
y = W·x + (α/r) · (B · A · x)
```
- `α/r` = scaling (cu α=8, r=8 → 1.0). Decuplează magnitudinea de rank.
- **Inițializare critică:** `A` = Kaiming aleator, `B` = **zerouri**. Deci la pasul 0,
  `B·A = 0` → modelul produce **exact** output-ul zero-shot. Adaptarea pornește lin, fără să
  perturbe modelul preantrenat. (Esențial pentru ancora de consistență — vezi §5.)
- `W` și bias-ul sunt **înghețate** (`requires_grad=False`).

> **Pe tablă:** dreptunghi mare `W` (gri, înghețat) cu `x→y`; ramură paralelă `x → A (in→r) →
> B (r→out) → ×(α/r)`, adunată la ieșirea lui `W`. `A`, `B` verzi (antrenabile).

### 3.2 Unde se injectează: cele 96 de layere
`apply_lora_to_encoder` parcurge **ambii encoderi** (`patch_encoder` + `image_encoder`), iar
în fiecare bloc transformer înlocuiește două proiecții liniare de atenție:
- `attn.qkv` — proiecția fuzionată Query/Key/Value.
- `attn.proj` — proiecția de ieșire a atenției.

24 blocuri × 2 layere × 2 encoderi = **96 de layere de atenție adaptate**.

### 3.3 Ce se antrenează vs. ce e înghețat
1. Întâi se îngheață **tot**.
2. Se aplică LoRA → singurii parametri antrenabili din encoderi devin `lora_A`/`lora_B`.
3. Se **dezgheață complet** decoderul + depth head.
4. **Capul de FOV rămâne înghețat** (folosim focala reală).
5. **PoseNet (ResNet-18) se antrenează de la zero** (doar la KITTI self-sup).

Total antrenabil ≈ **34M / 966M = 3.6%**. De ce e justificat:
- **Memorie:** stările optimizatorului doar pentru 34M, nu 966M → încape pe 12GB.
- **Anti-forgetting:** rang mic = capacitate limitată de a strica reprezentările ViT.

---

## 4. Variantă supervizată: fine-tuning LoRA pe NYU (`training/train_nyu_lora.py`)

> ATENȚIE: acesta e fișierul pe care îl folosește **notebook-ul de inferință** (importă din el
> `LoRALinear` și `apply_lora_to_encoder`). E o variantă **supervizată indoor** (NYU Depth V2,
> cu ground-truth de la Kinect), separată de pipeline-ul principal self-supervised KITTI.

- **Date:** `nyu_depth_v2_labeled.mat` (HDF5), imagini + adâncimi. Augmentări: flip orizontal,
  variație de luminozitate. Normalizare în [−1,1].
- **Loss supervizat** (avem GT):
  - `ScaleInvariantLogLoss` (SI-log): `mean(log_diff²) − 0.5·mean(log_diff)²`, unde
    `log_diff = log(pred) − log(gt)`. Invariant la scala globală, se concentrează pe structura
    relativă.
  - `GradientMatchingLoss`: potrivește gradienții ∂x, ∂y ai log-adâncimii → margini clare.
  - Total: `loss_si + 0.5·loss_grad`.
- **Conversie:** modelul dă inverse depth canonică → se scalează cu `W / 518.8579` (focala NYU)
  → se inversează → adâncime metrică. Mască validă 1mm–10m.
- **Antrenare:** AdamW cu LR discriminative (LoRA 5e-5, decoder/head 1e-4), warmup liniar +
  cosine annealing, FP16 + GradScaler, gradient clipping 1.0, gradient accumulation (batch
  efectiv 4), gradient checkpointing.
- **Checkpoint:** se salvează `model.state_dict()` **după** aplicarea LoRA → conține cheile
  `...attn.qkv.original.weight`, `...attn.qkv.lora_A/lora_B`, plus decoder/head. **De aceea
  notebook-ul trebuie să reconstruiască structura LoRA înainte de a încărca**, altfel cheile
  nu se potrivesc. Fișiere: `depth_pro_lora{suffix}_best.pt` / `_final.pt`.
- **Metrici validare:** `abs_rel` (mic = bine, criteriu pentru „best") și `delta1` (δ<1.25).

---

## 5. Metoda principală: training self-supervised KITTI (`training/train_kitti_selfsup_ms.py`)

> Acesta e fișierul CENTRAL al contribuției (906 linii). Loss-urile și warping-ul sunt în
> `src/depth_pro/selfsup/` (`losses.py`, `warping.py`, `pose_net.py`, `kitti_dataset.py`).

### 5.1 Formula pierderii totale
```
L_total = L_photo + λ_smooth · L_smooth + λ_cons · L_cons
```
- `L_photo` = pierdere fotometrică (warping + SSIM/L1 + minimum reprojection).
- `L_smooth` = smoothness edge-aware pe disparitate (λ_smooth ≈ 1e-3).
- `L_cons` = **consistency anchor** către zero-shot (λ_cons, implicit 1.0; în teză λ=10).

### 5.2 Datele KITTI (`kitti_dataset.py`)
- Triplete **(t−1, t, t+1)**. Două rezoluții simultan:
  - `target_depth`: 1536×1536 (intrarea encoderului), normalizat [−1,1].
  - `target`/`source_±1`: ~416×128 (rezoluție mică pentru PoseNet + loss; warping la 1536²
    ar fi prohibitiv ca VRAM).
- **Augmentări identice pe toate cele 3 cadre** (color jitter) — esențial: warping-ul presupune
  aparență constantă (brightness consistency); culori diferite ar falsifica loss-ul.
- **Intrinsics K** citite din `calib_cam_to_cam.txt`, scalate la rezoluția de pose. Se calculează
  și `inv_K` (pentru backprojection).
- `--stride` (ex. 3 sau 6): KITTI e la 10Hz → cadre redundante; subeșantionăm → mișcare mai
  informativă între cadre.

### 5.3 PoseNet (`pose_net.py`)
- De ce: în self-supervised monocular, ca să reproiectezi un cadru vecin în cel curent, ai
  nevoie de **poza relativă** (mișcarea camerei). Nu e cunoscută → o învățăm.
- ResNet-18 (ImageNet), primul conv modificat să accepte **6 canale** (2 cadre RGB concatenate).
- Output: **6-DoF** = [axis-angle rotație (3), translație (3)], scalat ×0.01 (la start ≈ identitate
  → warp-uri stabile).
- Conversie axis-angle → matrice 4×4 prin **formula Rodrigues**.
- Alternativă: `--vggt-poses` dezactivează complet PoseNet și folosește poze VGGT precomputate.

### 5.4 Pierderea fotometrică (`warping.py` + `losses.py`)
Pipeline warping (diferențiabil):
1. **Backproject:** `P_3D = D(u,v) · K⁻¹ · [u,v,1]ᵀ`.
2. **Proiectează în cadrul source:** `P' = K · T · P_3D`, apoi `pixel = P'[:2]/P'[2]`.
3. **Sample bilinear:** `F.grid_sample(source, pix_coords)` — diferențiabil → gradientul curge
   spre adâncime și poză.

Loss per pixel: `L_p = 0.85·SSIM_loss + 0.15·L1` (standard Monodepth2).
- **Minimum reprojection:** per pixel se ia `min` peste cele 2 surse → tratează **ocluziile**
  (alege sursa unde reconstrucția e bună).
- **Auto-masking:** maschează pixelii unde warping-ul nu bate „identity loss" (obiecte statice /
  fără paralaxă). *Notă onestă pentru comisie:* pe KITTI a fost **dezactivat** (mișcare constantă
  a camerei + un bug NaN în v2), deci în practică `L_photo = min_reproj.mean()`.
- **Smoothness edge-aware:** `|∂x d|·e^(−|∂x I|) + |∂y d|·e^(−|∂y I|)` pe disparitate normalizată
  la medie (trucul Monodepth2 contra contracției adâncimii).

### 5.5 Consistency loss — contribuția metodologică principală
Cu `--zeroshot-depths` se încarcă adâncimile **precomputate** ale Depth Pro zero-shot (ancora).
La fiecare pas:
```
L_cons = | d_pred − d_zs |            (mod "l1")
       sau | log d_pred − log d_zs |  (mod "log")
```
Ponderări opționale:
- `--depth-weight-power`: accentuează pixelii îndepărtați (unde RMSElog/δ suferă).
- `--edge-aware-consistency`: ancoră slabă pe margini (zero-shot e imprecis acolo), puternică
  în zonele netede.

**De ce previne drift-ul (argument-cheie):** pierderea fotometrică singură are ambiguități
(scală nedeterminată, suprafețe fără textură) → modelul ar „deriva" spre soluții care
minimizează fotometric dar sunt geometric greșite (PoseNet compensează adâncimi prost
calibrate dar consistente la warping). Ancora fixează modelul aproape de priorul puternic
zero-shot; fotometria doar îl **rafinează** local. λ controlează echilibrul: λ→0 = fotometric
pur (degradează), λ→∞ = exact zero-shot. **De aici numele AnchorDepth.**

### 5.6 Detalii de antrenare (de menționat că le stăpânești)
- **bfloat16** (nu fp16!) prin `autocast` — exponent pe 8 biți, fără overflow → fără NaN în LoRA.
- **Gradient checkpointing** pe ambii encoderi (memorie).
- **LR discriminative** (AdamW): LoRA 1e-5, decoder/head 1e-4, PoseNet 1e-4 (fără weight decay).
- **CosineAnnealingWarmRestarts** (T0=10) + warmup liniar 2 epoci.
- **Anti-NaN:** skip pas dacă loss-ul nu e finit, sanitizare gradienți, clip norm 1.0,
  verificare NaN/Inf **înainte** de salvarea checkpoint-ului.
- **Checkpoint:** `selfsup_best.pt` (când scade `val_photometric`) / `selfsup_final.pt`;
  conține `depth_model`, `pose_net`, `epoch`, `val_photometric`.
- **Validare:** `val_photometric` + metrici GT lejere pe 50 imagini (LiDAR proiectat, Garg crop,
  median scaling, abs_rel/δ1).

---

## 6. Instrumente offline (`tools/`)

Filosofia comună: **precalculăm o singură dată ce altfel s-ar repeta redundant.**

### 6.1 `precompute_zeroshot_depths.py` — ancora
- `d_zs` (zero-shot) e **constant** pe tot antrenamentul (modelul de bază e înghețat). Dacă l-am
  calcula în buclă, ar însemna un al doilea Depth Pro la fiecare pas → enorm.
- Rulează modelul de bază **o dată** peste tot setul, salvează `dict[sample_idx] → tensor[H,W]`
  în **float16** (jumătate de spațiu), fișier `zeroshot_depths_{split}_s{stride}_{W}x{H}.pt`.
- `--stride` trebuie identic cu cel din antrenare (altfel indicii nu se potrivesc — de aia apare
  în numele fișierului). Aceeași preprocesare + focală reală KITTI ca în antrenare.

### 6.2 `export_anchordepth.py` — îmbinarea LoRA → model de livrare
- La inferență nu vrem dependență de cod LoRA. Pentru că LoRA e liniar, putem pre-aduna:
  ```
  W_nou = W + (α/r) · (B @ A)
  ```
- `merge_lora_into_base` parcurge modelul, calculează `W_nou`, **înlocuiește fizic** fiecare
  `LoRALinear` cu un `nn.Linear` simplu. Verifică: niciun `LoRALinear` rămas, niciun NaN.
- Rezultat: un `state_dict` Depth Pro **standard** care se încarcă cu `strict=True` **fără**
  bibliotecă LoRA. Acesta e `anchordepth.pt` publicat pe HuggingFace.
- Variante: `v15`→KITTI, `v18`→Make3D, `v20`→Cityscapes (toate rank=8, α=8).

### 6.3 `precompute_vggt_poses.py` — poze multi-view (opțional)
- Un PoseNet aleator dă poze slabe la început → contaminează semnalul. Alternativă: **VGGT**
  (model fundație multi-view, CVPR 2025), rulat offline, produce poze de calitate.
- `relative_pose`: `T_target→source = E_source @ inv(E_target)`, cu inversă analitică
  `inv([R|t]) = [Rᵀ | −Rᵀt]`, aliniată exact la convenția din `warping.py`.
- Salvează `{sample_idx: {T_prev, T_next}}`. Degradare elegantă (identitate) dacă VGGT eșuează.
- **Concluzie onestă din teză:** pe KITTI, VGGT **nu** bate PoseNet antrenat — dar e o
  contribuție metodologică (combini două modele fundație pe 12GB prin precalcul).

---

## 7. Evaluarea (`evaluation/`)

### 7.1 Protocolul KITTI Eigen (`evaluate_kitti.py`)
- **697 imagini** de test (din `splits/eigen_test_files.txt`).
- **GT din LiDAR sparse** (Velodyne): se proiectează norul de puncte în imagine prin
  `P2 @ R_rect @ Tr_velo`, se păstrează cel mai apropiat punct per pixel.
- **Garg crop:** decupaj fix (≈40.8%–99.2% vertical, 3.6%–96.4% orizontal), aplicat **identic**
  la GT și predicție.
- **Plafon 80m:** mască `1e-3 < gt < 80`.

### 7.2 Metricile (cu formule — învață-le)
Cu `d` = predicție (după scalare), `d*` = GT, `N` pixeli valizi:
```
AbsRel  = mean(|d − d*| / d*)             (↓)  metrica principală
SqRel   = mean((d − d*)² / d*)            (↓)
RMSE    = sqrt(mean((d − d*)²))           (↓)  în metri
RMSElog = sqrt(mean((log d − log d*)²))   (↓)
δ_i     = max(d/d*, d*/d)
δ<1.25  = mean(δ_i < 1.25)                (↑)  ±25%
δ<1.25² , δ<1.25³                          (↑)
```
Erori: mic = bine. Praguri δ: mare = bine.

### 7.3 Median scaling (întrebare sigură)
Self-supervised recuperează adâncimea doar până la un factor de scară. La evaluare:
```
scale = median(GT) / median(pred)      # mediana = robustă la outlieri
pred  = pred * scale
```
Calculat **per imagine**, pe pixelii valizi. Standard în toată literatura self-supervised →
comparația rămâne corectă. `mean_scale ≈ 1.0` arată că modelul e deja aproape metric.

### 7.4 Cross-domain: Cityscapes + Make3D
- **Cityscapes** (500 imagini): GT din **disparitate stereo** (`depth = f·B/disp`), fără Garg crop,
  plafon 80m. „Zero-shot cross-domain" = testat fără reantrenare pe alt domeniu → demonstrează
  **generalizare**, nu memorare.
- **Make3D** (134 imagini): GT din `.mat` (laser), protocol **C1** (plafon 70m), metrică extra
  **log10**. Atenție la orientarea portrait (fără rotație).
- Ambele refolosesc `load_model` din `evaluate_kitti.py`.

### 7.5 Inferență la evaluare
Un singur forward la 1536×1536, `autocast`, **fără** flip test-time augmentation (cifre oneste).
Pentru modelele fine-tuned se folosește focala reală KITTI (capul FOV nu a fost reantrenat).
Încărcarea checkpoint-ului reconstruiește structura LoRA + filtrează cheile pe formă (`strict=False`).

---

## 8. Rezultatele cheie (Cap. 4–5)

| Benchmark | Variantă | Metrică principală | vs. zero-shot |
|---|---|---|---|
| **KITTI Eigen** (697) | v15 (L1, λ=10) | AbsRel **0.0852** (de la 0.0866) | −1.6%, îmbunătățește 4/7 metrici |
| **Cityscapes** (500) | v20 (L1, λ=20) | AbsRel **0.1085** | îmbunătățește **toate 7** |
| **Make3D** (134) | v18 (log, λ=10) | AbsRel **0.194** (de la 0.2575) | **−24.7%**, SqRel −55.1% |

**Tiparul central (împerecherea variantă–saturație):** câștigul scalează **invers** cu cât de
saturat e baseline-ul pe domeniul țintă. KITTI (cel mai saturat) → câștig mic; Make3D (cel mai
puțin saturat) → câștig mare. Niciun membru singular nu e cel mai bun pe tot, dar familia
domină colectiv zero-shot pe toate trei.

**Ablația-cheie:** fără ancoră, AbsRel 0.087 → 0.458 (degradare ×5.3). λ=10 optim; λ=20
supra-constrânge. Depth-power weighting = catastrofal.

**Limitări (Cap. 5):** un singur GPU, un singur dataset de antrenare (KITTI), un singur model
fundație (Depth Pro), doar outdoor.
**Direcții viitoare:** alte modele fundație (DepthAnything-v2, Marigold), evaluare indoor (NYU),
semnale de distilare mai bogate (feature-maps, incertitudine), VGGT în producție.

---

## 9. Notebook-ul de inferență (`AnchorDepth_Inference_Demo.ipynb`)

Demo-ul vizual: **RGB | AnchorDepth | Depth Pro zero-shot**. Celule:
1. **Setup (local/Colab):** pe Colab clonează repo-ul, instalează dependențele, descarcă
   checkpoint-ul de pe HuggingFace.
2. **Import + device:** adaugă `src/` și `training/` în path, importă `depth_pro` și utilitarele
   LoRA, alege CUDA/CPU.
3. **Încărcare modele:** construiește scheletul Depth Pro fără checkpoint, încarcă
   `checkpoints/anchordepth.pt`. Dacă e checkpoint de antrenare (cu cheia `depth_model`), aplică
   întâi LoRA; dacă e merged/exportat, îl încarcă direct. Opțional încarcă și zero-shot.
4. **Selectare imagine:** local = dialog tkinter; Colab = `files.upload()`.
5. **Inferință + comparație:** `predict_depth` redimensionează la 1536², rulează modelul,
   convertește canonical inverse depth → metri (formula din §2.5), colorează cu colormap pe
   inversa adâncimii, afișează side-by-side și salvează `figures/inference_comparison.png`.

> Notă: checkpoint-ul de 3.6 GB (`anchordepth.pt`) e gitignored — trebuie descărcat din
> HuggingFace (`dariusan3/AnchorDepth`). Pe Colab îl face automat celula de setup.

---

## 10. Întrebări probabile ale comisiei + răspunsuri

1. **De ce LoRA și nu fine-tuning complet?** 952M parametri × stările AdamW + activări nu încap
   pe 12GB. LoRA antrenează ~34M (3.6%), încape; plus regularizare implicită (rang mic) → nu
   strică reprezentările ViT.
2. **De ce e nevoie de consistency loss?** Self-supervision fotometric pur degradează modelul
   fundație ×5 (ablația). Fotometria are soluții degenerate; ancora ține modelul aproape de
   priorul zero-shot și lasă fotometria doar să rafineze.
3. **Ce e median scaling?** Aliniere de scară per imagine (median(GT)/median(pred)) — necesară
   pentru că self-supervised dă adâncime doar până la o scară. Standard în literatură.
4. **Metrică vs. relativă?** Metrică = metri reali; relativă = corectă până la un factor.
   Depth Pro = metric; adaptarea reintroduce ambiguitate de scară → median scaling.
5. **Dacă zero-shot bate deja tot, de ce-l adaptezi?** Ca să vedem dacă extragem semnal *în plus*
   din video neetichetat fără degradare. Răspuns: da, +1.6%…+24.7% după domeniu.
6. **Ce e canonical inverse depth?** 1/depth normalizată la o focală de referință; devine metrică
   după ×(W/f_px). Independența de focală = generalizare fără calibrare.
7. **De ce bfloat16 și nu float16?** fp16 (exponent 5 biți) dă overflow → NaN în LoRA prin multe
   blocuri; bf16 (exponent 8 biți, ca fp32) elimină overflow-ul, accelerat pe Ampere+.
8. **De ce auto-masking?** Maschează pixelii unde warping-ul nu ajută (statici / fără paralaxă).
   (Onest: pe KITTI l-am dezactivat — mișcare constantă + un bug NaN.)
9. **De ce VGGT dacă nu bate PoseNet pe KITTI?** Contribuție metodologică (două modele fundație
   pe 12GB prin precalcul offline) + cea mai bună δ<1.25³; concluzie onestă, nu eșec.
10. **De ce două rezoluții (1536² vs. 416×128)?** Encoderul Depth Pro cere 1536²; fotometria +
    PoseNet rulează la 416×128 din buget de VRAM.
11. **„Identitate la inițializare" pentru LoRA?** B=0 → contribuție nulă la start → modelul =
    exact zero-shot. Esențial: antrenarea pornește fix din punctul de ancorare.
12. **Cum alegi λ? De ce 10?** Sweep: λ=0 degradează, λ=10 optim KITTI, λ=20 supra-constrânge,
    λ→∞ = zero-shot. Optimul urmărește saturația benchmark-ului.
13. **De ce trei benchmark-uri și variante diferite?** Generalizare cross-domain + tiparul
    variantă–saturație (KITTI→v15, Cityscapes→v20, Make3D→v18).
14. **L1 metric vs. log-space?** L1 metric = `|d−d_zs|` (KITTI saturat); log = `|log d − log d_zs|`,
    cuplat cu RMSElog/δ, mai bun pe game largi (Make3D).
15. **Limitări + viitor?** Un GPU, un dataset de antrenare, un model fundație, doar outdoor →
    viitor: alte modele fundație, indoor (NYU), distilare mai bogată, VGGT în producție.
16. **De ce self-supervised dacă KITTI are LiDAR?** LiDAR-ul e folosit doar la *evaluare*. Scopul:
    adaptare din semnal pur fotometric (video neetichetat), care scalează la orice domeniu.

---

## 11. Harta fișierelor (pentru orientare rapidă)
```
src/depth_pro/                         backbone Depth Pro (Apple) — §2
  depth_pro.py, network/{encoder,decoder,fov,vit,vit_factory}.py, utils.py
  selfsup/{losses,warping,pose_net,kitti_dataset}.py   pierderi + warping + date — §5
training/
  train_kitti_selfsup_ms.py            metoda principală self-supervised — §5
  train_nyu_lora.py                    LoRALinear + apply_lora_to_encoder (folosit de notebook) — §3,§4
evaluation/
  evaluate_{kitti,cityscapes,make3d}.py  evaluare + metrici — §7
tools/
  precompute_zeroshot_depths.py        ancora pentru consistency loss — §6.1
  export_anchordepth.py                merge LoRA → anchordepth.pt — §6.2
  precompute_vggt_poses.py             poze VGGT offline (opțional) — §6.3
AnchorDepth_Inference_Demo.ipynb       demo vizual de inferință — §9
docs/THESIS_CAP1..CAP5_*.md            capitolele tezei
```
