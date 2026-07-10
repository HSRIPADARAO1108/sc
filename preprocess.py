from PIL import Image
import numpy as np
import cv2


def preprocess_image(uploaded_file, mode="Binary", invert=False, debug=False):
    """
    Convert an uploaded character image into a 3x5 (15-pixel) binary/bipolar
    pattern compatible with the patterns trained in patterns.py.

    Why the old version failed on T / H / I:
    A naive `img.resize((3, 5))` squashes the ENTIRE image (including empty
    margins) into a 3x5 grid using blurry interpolation, and never checks
    whether the background is black or white. Since the trained patterns are
    tight outlines that exactly fill their 3x5 grid, any uploaded image that
    isn't already perfectly cropped/colored produces a mismatched pattern.

    This version fixes that with:
      1. Grayscale conversion
      2. Otsu thresholding -> clean binary image (0 / 255)
      3. Automatic background detection (so the character is always the
         foreground/white pixels, regardless of upload color scheme)
      4. Crop to the bounding box of the character (removes empty margins)
      5. Resize the cropped character to 3x5 using AREA interpolation
         (averages pixels instead of blurring/skipping them)
      6. Re-threshold to obtain crisp 0/1 pixels
      7. Flatten row-major (matches patterns.py layout) and convert to
         bipolar (-1/1) if required
    """

    # ---- 1. Load & grayscale -----------------------------------------
    img = Image.open(uploaded_file).convert("L")
    img_np = np.array(img)

    # ---- 2. Otsu threshold -> binary 0/255 ----------------------------
    _, binary = cv2.threshold(
        img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

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

    # ---- 5. Resize to 3x5 using area interpolation -----------------------
    resized = cv2.resize(
        cropped.astype(np.uint8), (3, 5), interpolation=cv2.INTER_AREA
    )

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
