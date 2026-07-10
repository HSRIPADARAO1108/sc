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


def preprocess_image(uploaded_file, mode="Binary", invert=False, debug=False):
    """
    Convert an uploaded character image into a 3x5 (15-pixel) binary/bipolar
    pattern compatible with the patterns trained in patterns.py.

    Pipeline (pure PIL + numpy, no opencv):
      1. Grayscale conversion
      2. Otsu thresholding -> clean binary image (0 / 255)
      3. Automatic background detection (so the character is always the
         foreground/white pixels, regardless of upload color scheme)
      4. Crop to the bounding box of the character (removes empty margins)
      5. Resize the cropped character to 3x5 using PIL's BOX filter
         (averages pixels, similar to opencv's INTER_AREA)
      6. Re-threshold to obtain crisp 0/1 pixels
      7. Flatten row-major (matches patterns.py layout) and convert to
         bipolar (-1/1) if required
    """

    # ---- 1. Load & grayscale -----------------------------------------
    img = Image.open(uploaded_file).convert("L")
    img_np = np.array(img)

    # ---- 2. Otsu threshold -> binary 0/255 ----------------------------
    thresh = _otsu_threshold(img_np)
    binary = np.where(img_np > thresh, 255, 0).astype(np.uint8)

    # ---- 3. Make sure character pixels are white (255) -----------------
    # Sample the four corners to guess the background colour. If most
    # corners are white, the background is white, so flip so the
    # character (foreground) becomes the white/foreground pixels.
    corners = [
        binary[0, 0], binary[0, -1],
        binary[-1, 0], binary[-1, -1]
    ]
    background_is_white = sum(c == 255 for c in corners) >= 2

    if background_is_white:
        binary = 255 - binary

    if invert:
        # Manual override exposed in the UI, in case auto-detection guesses wrong
        binary = 255 - binary

    # ---- 4. Crop to bounding box of the character -----------------------
    ys, xs = np.where(binary > 0)
    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cropped = binary[y_min:y_max + 1, x_min:x_max + 1]
    else:
        # Nothing detected (blank image) — fall back to the full frame
        cropped = binary

    # ---- 5. Resize to 3x5 using PIL's BOX (area-averaging) filter ---------
    cropped_img = Image.fromarray(cropped.astype(np.uint8), mode="L")
    resized_img = cropped_img.resize((3, 5), resample=Image.BOX)
    resized = np.array(resized_img)

    # ---- 6. Re-threshold (majority vote per cell) --------------------------
    final_binary = np.where(resized > 127, 1, 0)

    pattern = final_binary.flatten()

    if debug:
        return pattern, cropped, resized

    # ---- 7. Binary or Bipolar -------------------------------------------
    if mode == "Binary":
        return pattern
    else:
        return np.where(pattern == 0, -1, 1)
