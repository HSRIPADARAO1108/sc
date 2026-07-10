from PIL import Image, ImageOps, ImageFilter
import numpy as np


def preprocess_image(
        uploaded_file,
        mode="Binary",
        invert=False,
        debug=False,
        stroke_boost=1,
        on_threshold=0.3
):

    # -----------------------------
    # Read Image
    # -----------------------------

    img = Image.open(uploaded_file).convert("L")


    # -----------------------------
    # Invert
    # -----------------------------

    if invert:

        img = ImageOps.invert(img)



    # -----------------------------
    # Crop character
    # -----------------------------

    arr = np.array(img)


    # Detect foreground

    mask = arr < 200


    if np.any(mask):

        coords = np.argwhere(mask)

        y_min,x_min = coords.min(axis=0)

        y_max,x_max = coords.max(axis=0)


        img = img.crop(
            (
                x_min,
                y_min,
                x_max+1,
                y_max+1
            )
        )



    # -----------------------------
    # Make strokes thicker
    # -----------------------------

    for i in range(stroke_boost):

        img = img.filter(
            ImageFilter.MaxFilter(3)
        )



    # -----------------------------
    # Resize keeping shape
    # -----------------------------

    img.thumbnail(
        (90,90)
    )


    canvas = Image.new(
        "L",
        (90,90),
        255
    )


    x = (90-img.width)//2

    y = (90-img.height)//2


    canvas.paste(
        img,
        (x,y)
    )



    # -----------------------------
    # Threshold
    # -----------------------------

    arr=np.array(canvas)


    binary=np.where(
        arr < 150,
        1,
        0
    )



    # -----------------------------
    # Resize to 5x3
    # -----------------------------

    small = Image.fromarray(
        (binary*255).astype(np.uint8)
    )


    small = small.resize(
        (3,5),
        Image.Resampling.BILINEAR
    )


    small=np.array(small)



    # -----------------------------
    # Cell activation
    # -----------------------------

    pattern=(small > 80).astype(int)



    # Flatten

    pattern=pattern.flatten()



    # Check

    if pattern.size != 15:

        raise Exception(
            "Invalid pattern size"
        )



    # Debug output

    if debug:

        return (
            pattern,
            binary,
            pattern.reshape(5,3)
        )



    # Binary output

    if mode=="Binary":

        return pattern



    # Bipolar output

    return np.where(
        pattern==0,
        -1,
        1
    )
