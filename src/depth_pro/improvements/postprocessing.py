"""Post-processing refinements for Depth Pro predictions.

Includes:
  - Guided bilateral filtering (edge-preserving depth smoothing using RGB)
  - Median scaling correction

Usage:
    from depth_pro.improvements.postprocessing import guided_filter_depth
    depth_refined = guided_filter_depth(depth_map, rgb_image, radius=8, eps=0.01)
"""

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def guided_filter_depth(
    depth: np.ndarray,
    rgb: np.ndarray,
    radius: int = 8,
    eps: float = 0.01,
) -> np.ndarray:
    """Apply guided bilateral filter using RGB image to refine depth edges.

    The RGB image provides edge guidance so depth boundaries align with
    object boundaries, producing sharper, more accurate depth maps.

    Args:
        depth: Depth map (H, W) float32.
        rgb: RGB guide image (H, W, 3) uint8 or float32 [0-1].
        radius: Filter radius (larger = smoother).
        eps: Regularization (smaller = sharper edges).

    Returns:
        Filtered depth map (H, W) float32.
    """
    if not HAS_CV2:
        raise ImportError("OpenCV required for guided filtering: pip install opencv-python-headless")

    if rgb.dtype == np.uint8:
        guide = rgb.astype(np.float32) / 255.0
    else:
        guide = rgb.astype(np.float32)

    # Convert to grayscale guide
    if guide.ndim == 3:
        guide_gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
    else:
        guide_gray = guide

    depth_f32 = depth.astype(np.float32)

    # Apply guided filter
    filtered = _guided_filter(guide_gray, depth_f32, radius, eps)

    return filtered


def _guided_filter(guide, src, radius, eps):
    """Guided filter — uses OpenCV ximgproc if available, else fallback."""
    if hasattr(cv2, 'ximgproc'):
        return cv2.ximgproc.guidedFilter(guide, src, radius, eps)

    # Fallback: box-filter based implementation
    mean_g = cv2.boxFilter(guide, -1, (radius, radius))
    mean_s = cv2.boxFilter(src, -1, (radius, radius))
    mean_gs = cv2.boxFilter(guide * src, -1, (radius, radius))
    mean_gg = cv2.boxFilter(guide * guide, -1, (radius, radius))

    cov_gs = mean_gs - mean_g * mean_s
    var_g = mean_gg - mean_g * mean_g

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    return mean_a * guide + mean_b


def median_scale_correction(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    mask: np.ndarray = None,
) -> tuple[np.ndarray, float]:
    """Apply median scaling to align predicted depth to ground truth.

    Args:
        pred_depth: Predicted depth (H, W).
        gt_depth: Ground truth depth (H, W).
        mask: Valid pixel mask (H, W) bool.

    Returns:
        Tuple of (scaled_depth, scale_factor).
    """
    if mask is None:
        mask = (gt_depth > 1e-3) & (pred_depth > 1e-3)

    scale = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
    return pred_depth * scale, scale
