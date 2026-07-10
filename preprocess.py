from PIL import Image, ImageOps
import numpy as np

def preprocess_image(uploaded_file, mode="Binary", invert=False, debug=False,
                     stroke_boost=0, on_threshold=0.5):
    """
    Preprocess uploaded character image into a 5x5 pattern for Hebb network.
    """

    img = Image.open(uploaded_file).convert("L")

    if invert:
        img = ImageOps.invert(img)

    # Resize while preserving aspect ratio
    img.thumbnail((100,100))

    # Fit on white canvas
    canvas = Image.new("L",(100,100),255)
    x = (100-img.width)//2
    y = (100-img.height)//2
    canvas.paste(img,(x,y))

    # Threshold
    arr = np.array(canvas)
    binary = np.where(arr<180,255,0).astype(np.uint8)

    # Resize to 5x5
    small = Image.fromarray(binary).resize((5,5),Image.Resampling.BOX)
    small = np.array(small)

    pattern = np.where(small>=128,1,0).astype(int)

    if debug:
        return pattern.flatten(), binary, small

    if mode=="Binary":
        return pattern.flatten()

    return np.where(pattern.flatten()==0,-1,1)
