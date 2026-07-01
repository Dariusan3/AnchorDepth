"""Generate corrected UML class diagram for AnchorDepth pipeline (16:9, PPT-ready)."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FONT_TITLE = dict(family="DejaVu Sans", weight="bold", size=13, color="white")
FONT_ATTR  = dict(family="DejaVu Sans Mono", size=10, color="#111827")
FONT_METH  = dict(family="DejaVu Sans Mono", size=10, style="italic", color="#075985")

C_ROOT = "#0f172a"
C_LORA = "#7c3aed"
C_INT  = "#059669"
C_HEAD = "#0369a1"
C_SELF = "#ea580c"
C_DATA = "#d97706"
C_LOSS = "#475569"

W, H = 22, 12.4       # figure canvas in "logical" units
fig, ax = plt.subplots(figsize=(W, H), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 58)
ax.axis("off")
ax.set_facecolor("white")

def card(x, y, w, h, title, attrs, methods, color):
    hdr_h = 2.8
    body_h = h - hdr_h
    hdr = FancyBboxPatch((x, y + body_h), w, hdr_h,
                         boxstyle="round,pad=0.02,rounding_size=0.4",
                         linewidth=0, facecolor=color, zorder=2)
    ax.add_patch(hdr)
    ax.text(x + w/2, y + body_h + hdr_h/2, title,
            ha="center", va="center", **FONT_TITLE)
    body = FancyBboxPatch((x, y), w, body_h,
                          boxstyle="round,pad=0.02,rounding_size=0.4",
                          linewidth=1.4, edgecolor=color,
                          facecolor="white", zorder=1)
    ax.add_patch(body)

    line_h = 1.05
    top = y + body_h - 0.7
    for i, a in enumerate(attrs):
        ax.text(x + 0.55, top - i*line_h, a, ha="left", va="top", **FONT_ATTR)
    sep_y = top - len(attrs)*line_h + 0.35
    ax.plot([x + 0.4, x + w - 0.4], [sep_y, sep_y],
            color=color, lw=0.9, alpha=0.55)
    for j, m in enumerate(methods):
        ax.text(x + 0.55, sep_y - 0.7 - j*line_h, m,
                ha="left", va="top", **FONT_METH)

# ---- ROW 1 --------------------------------------------------------
y1 = 41.5
card(2,  y1, 22, 14, "DepthPro(nn.Module)",
     ["encoder: DepthProEncoder",
      "decoder: MultiresConvDecoder",
      "head: nn.Sequential  (→ depth)",
      "fov: FOVNetwork"],
     ["+ forward(x)",
      "    → (canon_inv_depth,",
      "         fov_deg)"],
     C_ROOT)

card(26, y1, 22, 14, "LoRALinear(nn.Module)",
     ["original: nn.Linear  (frozen)",
      "lora_A: Param (r, in)",
      "lora_B: Param (out, r)",
      "rank = 8,  alpha = 8.0"],
     ["+ forward(x) → Tensor",
      "    orig(x) + B(A x)·s"],
     C_LORA)

card(50, y1, 22, 14, "apply_lora_to_encoder()",
     ["targets: block.attn.qkv,",
      "         block.attn.proj",
      "rank = 8,  alpha = 8.0",
      "→ 96 wrapped layers"],
     ["+ __call__(model, r, a)",
      "    → 2.36 M trainable"],
     C_LORA)

card(74, y1, 24, 14, "selfsup / losses.py",
     ["photometric_loss(pred, tgt)",
      "smooth_loss(disp, img)",
      "compute_selfsup_loss(...)",
      "L_cons = ‖d_pred − d_zs‖₁"],
     ["L = L_photo + λ · L_cons",
      "    λ = 10.0 (default)"],
     C_LOSS)

# ---- ROW 2 --------------------------------------------------------
y2 = 22.5
card(2,  y2, 22, 14, "DepthProEncoder",
     ["patch_encoder: ViT-L/16",
      "image_encoder: ViT-L/16",
      "dims_encoder: list[int]",
      "hook_block_ids: (5, 11)"],
     ["+ forward(x)",
      "    → encodings: list[T]"],
     C_INT)

card(26, y2, 22, 14, "MultiresConvDecoder",
     ["dims_encoder: list[int]",
      "dim_decoder = 256",
      "convs, fusions: ModuleList"],
     ["+ forward(encodings)",
      "    → (features,",
      "         features_0)"],
     C_INT)

card(50, y2, 22, 14, "FOVNetwork",
     ["fov_encoder: Optional[ViT]",
      "head: nn.Sequential",
      "num_features: int"],
     ["+ forward(x, feat0.detach())",
      "    → fov_deg"],
     C_INT)

card(74, y2, 24, 14, "Depth Head",
     ["Conv2d(dim, dim//2)",
      "ConvTranspose2d ×2 (up)",
      "Conv2d(→ 1)  canonical inv-D"],
     ["+ head(features)",
      "    → canon_inv_depth"],
     C_HEAD)

# ---- ROW 3 --------------------------------------------------------
y3 = 3.5
card(2,  y3, 22, 14, "PoseNet(nn.Module)",
     ["backbone: resnet18",
      "conv1: 6-ch input (tgt‖src)",
      "pose_head: Conv → 6-dim",
      "global_pool: AdaptiveAvgPool2d"],
     ["+ forward(target, source)",
      "    → pose: (B, 6)"],
     C_SELF)

card(26, y3, 22, 14, "Warper(nn.Module)",
     ["backproject: BackprojectDepth",
      "project: Project3D",
      "batch_size, height, width"],
     ["+ forward(src, depth, T, K)",
      "    → warped: (B, 3, H, W)"],
     C_SELF)

card(50, y3, 22, 14, "KITTIRawDataset(Dataset)",
     ["sequences: list[Path]",
      "zeroshot_depths: dict",
      "K_scaled: (3, 3) intrinsics",
      "img_size = 1536×1536"],
     ["+ __getitem__(idx)",
      "    → (tgt, src−, src+,",
      "         d_zs, K, mask)"],
     C_DATA)

card(74, y3, 24, 14, "Ancoră de consistență",
     ["d_pred = model_LoRA(I)",
      "d_zs   = model_frozen(I)",
      "     (pre-cached offline)"],
     ["λ · L_cons ↦ backward",
      "    prevents drift"],
     C_LOSS)

# ---- Arrows (routed around cards) ---------------------------------
def arr(x0, y0, x1, y1, color="#334155", dashed=False, lw=1.6, style="-|>"):
    ls = (0, (5, 3)) if dashed else "-"
    a = FancyArrowPatch((x0, y0), (x1, y1),
                        arrowstyle=style, mutation_scale=15,
                        color=color, linestyle=ls, lw=lw, zorder=6)
    ax.add_patch(a)

# DepthPro root → internals (composition, all downward)
for cx, target_x in zip([6.5, 12, 17.5, 21.5], [13, 37, 61, 86]):
    arr(cx, y1, target_x, y2 + 14, color=C_INT, lw=1.3)

# LoRA → encoder (dashed = wraps)
arr(37, y1, 13, y2 + 14, color=C_LORA, dashed=True, lw=1.2)
arr(61, y1, 13, y2 + 14, color=C_LORA, dashed=True, lw=1.2)

# encoder → decoder → depth head + fov
arr(24, y2 + 7, 26, y2 + 7, color="#334155")
arr(48, y2 + 8, 74, y2 + 8, color=C_HEAD)
arr(48, y2 + 5, 50, y2 + 5, color="#334155")

# depth head → consistency anchor (dashed)
arr(86, y2, 86, y3 + 14, color=C_LOSS, dashed=True, lw=1.2)

# self-sup row: dataset ↔ warper ; posenet → warper
arr(50, y3 + 8, 48, y3 + 8, color=C_SELF)  # dataset → warper
arr(24, y3 + 7, 26, y3 + 7, color=C_SELF)  # posenet → warper

# losses → anchor
arr(86, y1, 86, y3 + 14, color=C_LOSS, dashed=True, lw=1.1)

# ---- Legend --------------------------------------------------------
legend_items = [
    ("Model fundamental",   C_ROOT),
    ("Adaptare LoRA",       C_LORA),
    ("Componente interne",  C_INT),
    ("Depth head",          C_HEAD),
    ("Auto-supervizare",    C_SELF),
    ("Date",                C_DATA),
    ("Funcții de pierdere", C_LOSS),
]
lx = 3
for name, col in legend_items:
    sq = mpatches.Rectangle((lx, 1.0), 1.5, 1.1,
                            facecolor=col, edgecolor="none")
    ax.add_patch(sq)
    ax.text(lx + 2.0, 1.55, name, va="center", ha="left",
            family="DejaVu Sans", size=10.5, color="#111827")
    lx += 3.4 + len(name) * 0.62

# Title
ax.text(50, 57.0, "Diagrama de clase a pipeline-ului AnchorDepth",
        ha="center", va="center",
        family="DejaVu Sans", weight="bold", size=18, color="#0f172a")

plt.tight_layout()
out = "/home/ubuntu/ml-depth-pro/anchordepth_class_diagram.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
