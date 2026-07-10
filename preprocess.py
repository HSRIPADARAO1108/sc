from PIL import Image
import numpy as np


def _otsu_threshold(img_np):
    """
    Compute Otsu's threshold manually with numpy (no opencv needed).
    Returns the integer threshold (0-255) that best separates the
    image into foreground/background.
    """
    hist, _ = np.histogram(img_np, bins=256, range=(0, 256))
    total = img_np.size

    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    weight_b = 0.0
    max_variance = 0.0
    threshold = 127  # sane fallback

    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue

        weight_f = total - weight_b
        if weight_f == 0:
            break

        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f

        variance_between = weight_b * weight_f * (mean_b - mean_f) ** 2

        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t

    return threshold


def _dilate(binary_255, iterations=1):
    """
    Thicken white (255) strokes by 'iterations' pixels in every direction.
    Pure-numpy morphological dilation (no opencv/scipy needed) using a
    3x3 cross-shaped structuring element.

    Why this matters: thin pen/font strokes (often just 1-3px wide in a
    cropped image) get averaged toward 0 when box-downsized into a tiny
    3x5 grid, silently erasing entire bars/columns. Thickening the stroke
    BEFORE downsizing ensures it still contributes meaningfully to the
    cell average it lands in.
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

    Pipeline (pure PIL + numpy, no opencv):
      1. Grayscale conversion
      2. Otsu thresholding -> clean binary image (0 / 255)
      3. Automatic background detection (so the character is always the
         foreground/white pixels, regardless of upload color scheme)
      4. Crop to the bounding box of the character, with a small margin
      5. Dilate (thicken) the stroke so thin lines survive downsizing
      6. Resize the cropped character to 3x5 using PIL's BOX filter
         (averages pixels, similar to opencv's INTER_AREA)
      7. Threshold each cell using a fraction of the max cell value
         (adaptive, so it isn't thrown off by thin/faint strokes)
      8. Flatten row-major (matches patterns.py layout) and convert to
         bipolar (-1/1) if required

    Parameters
    ----------
    stroke_boost : int
        Number of dilation passes applied before downsizing. Increase this
        (e.g. to 3-4) if you're photographing thin pen/pencil writing.
        Set to 0 to disable (useful for already-thick block characters).
    on_threshold : float
        Fraction (0-1) of the resized grid's max intensity a cell must
        reach to be counted as "on". Lower this if characters are still
        coming out too sparse; raise it if noise is being picked up.
    """

    # ---- 1. Load & grayscale -----------------------------------------
    img = Image.open(uploaded_file).convert("L")
    img_np = np.array(img)

    # ---- 2. Otsu threshold -> binary 0/255 ----------------------------
    thresh = _otsu_threshold(img_np)
    binary = np.where(img_np > thresh, 255, 0).astype(np.uint8)

    # ---- 3. Make sure character pixels are white (255) -----------------
    corners = [
        binary[0, 0], binary[0, -1],
        binary[-1, 0], binary[-1, -1]
    ]
    background_is_white = sum(c == 255 for c in corners) >= 2

    if background_is_white:
        binary = 255 - binary

    if invert:
        binary = 255 - binary

    # ---- 4. Crop to bounding box of the character (+ small margin) ------
    ys, xs = np.where(binary > 0)
    if len(xs) > 0 and len(ys) > 0:
        margin = 2
        x_min = max(0, xs.min() - margin)
        x_max = min(binary.shape[1] - 1, xs.max() + margin)
        y_min = max(0, ys.min() - margin)
        y_max = min(binary.shape[0] - 1, ys.max() + margin)
        cropped = binary[y_min:y_max + 1, x_min:x_max + 1]
    else:
        # Nothing detected (blank image) — fall back to the full frame
        cropped = binary

    # ---- 5. Thicken thin strokes before downsizing ------------------------
    if stroke_boost > 0:
        cropped = _dilate(cropped, iterations=stroke_boost)

    # ---- 6. Resize to 3x5 using PIL's BOX (area-averaging) filter ---------
    cropped_img = Image.fromarray(cropped.astype(np.uint8), mode="L")
    resized_img = cropped_img.resize((3, 5), resample=Image.BOX)
    resized = np.array(resized_img).astype(np.float64)

    # ---- 7. Adaptive threshold (fraction of the grid's own max value) -----
    max_val = resized.max()
    if max_val > 0:
        final_binary = np.where(resized >= max_val * on_threshold, 1, 0)
    else:
        final_binary = np.zeros_like(resized, dtype=int)

    pattern = final_binary.flatten()

    if debug:
        return pattern, cropped, resized

    # ---- 8. Binary or Bipolar -------------------------------------------
    if mode == "Binary":
        return pattern
    else:
        return np.where(pattern == 0, -1, 1)
