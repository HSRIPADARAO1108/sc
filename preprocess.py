from PIL import Image, ImageFilter
import numpy as np
from collections import deque


def _connected_components(binary_bool):
    """
    Simple 8-connectivity BFS connected-component labeling in pure Python
    (no opencv/scipy needed). Only practical because we run this on a
    downscaled image (a few hundred px per side), not the full-res photo.

    Returns a list of components, each a list of (y, x) coordinates.
    """
    h, w = binary_bool.shape
    visited = np.zeros_like(binary_bool, dtype=bool)
    components = []

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i in range(h):
        for j in range(w):
            if binary_bool[i, j] and not visited[i, j]:
                queue = deque([(i, j)])
                visited[i, j] = True
                comp = []
                while queue:
                    y, x = queue.popleft()
                    comp.append((y, x))
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < h and 0 <= nx < w
                                and binary_bool[ny, nx]
                                and not visited[ny, nx]):
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                components.append(comp)

    return components


def _isolate_character(binary_255, min_size_fraction=0.15):
    """
    Keep only the connected blob(s) that plausibly ARE the character (the
    largest blob, plus any others at least `min_size_fraction` of its
    size, to tolerate a pen stroke that got split into 2-3 pieces).
    Zeroes out everything else — isolated noise specks that survived
    thresholding but aren't part of the real stroke.
    """
    components = _connected_components(binary_255 > 0)
    if not components:
        return binary_255

    sizes = [len(c) for c in components]
    max_size = max(sizes)
    keep_threshold = max(3, max_size * min_size_fraction)

    cleaned = np.zeros_like(binary_255)
    for comp, size in zip(components, sizes):
        if size >= keep_threshold:
            ys = [p[0] for p in comp]
            xs = [p[1] for p in comp]
            cleaned[ys, xs] = 255

    return cleaned


def _dilate(binary_255, iterations=1):
    """
    Thicken white (255) strokes by 'iterations' pixels in every direction.
    Pure-numpy morphological dilation using a 3x3 cross-shaped element.
    """
    result = binary_255.astype(np.uint8)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        result = np.maximum.reduce([result, up, down, left, right])
    return result


def preprocess_image(uploaded_file, mode="Binary", invert=False, debug=False,
                      stroke_boost=2, on_threshold=0.30):
    """
    Convert an uploaded character image into a 3x5 (15-pixel) binary/bipolar
    pattern compatible with the patterns trained in patterns.py.

    Real photos of paper have two problems a naive threshold can't handle:
      (a) fine texture/grain/JPEG noise scattered across the whole sheet
      (b) large-scale lighting gradients / shadows (vignetting) that are
          genuinely dark over big areas, not just noise

    This pipeline handles both:
      1. Grayscale conversion, downscaled to a manageable working size
      2. Illumination correction: estimate the local background shading
         with a large-radius blur, then flag pixels that are much darker
         than their OWN neighborhood as ink. This cancels out shadows/
         vignetting since they vary slowly and get absorbed into the
         "local background" estimate, while ink is a sharp local outlier.
      3. Automatic background polarity detection + optional manual invert
      4. Connected-component filtering: keep only the blob(s) that are
         big enough to plausibly be the character, discard leftover
         scattered noise specks
      5. Crop to the bounding box of the surviving stroke, with a margin
      6. Dilate (thicken) the stroke so thin lines survive downsizing
      7. Resize to 3x5 using PIL's BOX filter (area-averaging)
      8. Adaptive per-cell threshold (fraction of the grid's own max)
      9. Flatten row-major (matches patterns.py) and convert to
         bipolar (-1/1) if required

    Parameters
    ----------
    stroke_boost : int
        Dilation passes applied before downsizing. Raise this for thin
        pen/pencil writing; lower/disable for already-thick block chars.
    on_threshold : float
        Fraction (0-1) of the resized grid's max intensity a cell must
        reach to count as "on". Lower = more sensitive to faint/thin
        strokes; higher = stricter (less noise-prone).
    """

    # ---- 1. Load, grayscale, downscale ---------------------------------
    img = Image.open(uploaded_file).convert("L")
    max_dim = 400
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    arr = np.array(img).astype(np.float64)
    w, h = img.size

    # ---- 2. Illumination-corrected ink detection ------------------------
    # Blur radius must be well beyond typical stroke width so the local
    # "background" estimate isn't itself dragged down by the ink.
    radius = max(w, h) // 4
    bg = np.array(img.filter(ImageFilter.GaussianBlur(radius=radius))).astype(np.float64)
    diff = np.clip(bg - arr, 0, None)   # positive where pixel is darker than its surroundings
    dmax = diff.max()

    if dmax > 0:
        binary = (diff > dmax * 0.20).astype(np.uint8) * 255
    else:
        binary = np.zeros_like(arr, dtype=np.uint8)

    # ---- 3. Manual invert override (rarely needed with illumination
    #          correction, but kept for edge cases / light-on-dark ink) --
    if invert:
        binary = 255 - binary

    # ---- 4. Drop scattered noise, keep the real stroke -------------------
    binary = _isolate_character(binary)

    # ---- 5. Crop to bounding box of the character (+ small margin) ------
    ys, xs = np.where(binary > 0)
    if len(xs) > 0 and len(ys) > 0:
        margin = 3
        x_min = max(0, xs.min() - margin)
        x_max = min(binary.shape[1] - 1, xs.max() + margin)
        y_min = max(0, ys.min() - margin)
        y_max = min(binary.shape[0] - 1, ys.max() + margin)
        cropped = binary[y_min:y_max + 1, x_min:x_max + 1]
    else:
        # Nothing detected (blank image) — fall back to the full frame
        cropped = binary

    # ---- 6. Thicken thin strokes before downsizing ------------------------
    if stroke_boost > 0:
        cropped = _dilate(cropped, iterations=stroke_boost)

    # ---- 7. Resize to 3x5 using PIL's BOX (area-averaging) filter ---------
    cropped_img = Image.fromarray(cropped.astype(np.uint8), mode="L")
    resized_img = cropped_img.resize((3, 5), resample=Image.BOX)
    resized = np.array(resized_img).astype(np.float64)

    # ---- 8. Adaptive threshold (fraction of the grid's own max value) -----
    max_val = resized.max()
    if max_val > 0:
        final_binary = np.where(resized >= max_val * on_threshold, 1, 0)
    else:
        final_binary = np.zeros_like(resized, dtype=int)

    pattern = final_binary.flatten()

    if debug:
        return pattern, cropped, resized

    # ---- 9. Binary or Bipolar -------------------------------------------
    if mode == "Binary":
        return pattern
    else:
        return np.where(pattern == 0, -1, 1)
