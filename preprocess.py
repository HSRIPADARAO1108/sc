from PIL import Image
import numpy as np


def preprocess_image(uploaded_file, mode="Binary"):

    # Open image
    img = Image.open(uploaded_file).convert("L")

    # Resize directly to 3x5
    img = img.resize((3, 5))

    img = np.array(img)

    # Convert to Binary
    binary = np.where(img < 128, 1, 0)

    binary = binary.flatten()

    if mode == "Binary":
        return binary

    bipolar = np.where(binary == 0, -1, 1)

    return bipolar
