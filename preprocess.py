# preprocess.py

from PIL import Image
import numpy as np


def preprocess_image(uploaded_file, mode="Binary"):

    # Open image
    img = Image.open(uploaded_file)

    # Convert to grayscale
    img = img.convert("L")

    # Resize to 3x5 pixels
    img = img.resize((3, 5))

    img_array = np.array(img)

    # Threshold
    binary = np.where(img_array < 128, 1, 0)

    # Flatten to 15 elements
    binary = binary.flatten()

    if mode == "Binary":
        return binary

    bipolar = np.where(binary == 0, -1, 1)

    return bipolar
