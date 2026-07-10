from PIL import Image
import numpy as np
import cv2


def preprocess_image(uploaded_file, mode="Binary"):

    # -----------------------------
    # Read Image
    # -----------------------------
    image = Image.open(uploaded_file).convert("L")

    img = np.array(image)

    # -----------------------------
    # Blur to remove noise
    # -----------------------------
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # -----------------------------
    # Otsu Threshold
    # Character = White
    # Background = Black
    # -----------------------------
    _, thresh = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # -----------------------------
    # Find Character
    # -----------------------------
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return np.zeros(15)

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    cropped = thresh[y:y+h, x:x+w]

    # -----------------------------
    # Add Padding
    # -----------------------------
    pad = 20

    cropped = cv2.copyMakeBorder(
        cropped,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # -----------------------------
    # Resize
    # -----------------------------
    resized = cv2.resize(
        cropped,
        (3,5),
        interpolation=cv2.INTER_AREA
    )

    # -----------------------------
    # Binary Conversion
    # -----------------------------
    binary = (resized > 128).astype(int)

    binary = binary.flatten()

    if mode == "Binary":
        return binary

    bipolar = np.where(binary == 0, -1, 1)

    return bipolar
