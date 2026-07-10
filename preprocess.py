from PIL import Image, ImageOps, ImageFilter
import numpy as np


def preprocess_image(
        uploaded_file,
        mode="Binary",
        invert=False,
        debug=False,
        stroke_boost=0,
        on_threshold=0.5
    ):
    """
    Convert uploaded character image into 5x3 pattern
    for Hebb Character Recognition.

    Output:
    5 rows × 3 columns = 15 pixels
    """


    # -------------------------------
    # Read Image
    # -------------------------------

    img = Image.open(uploaded_file).convert("L")


    # -------------------------------
    # Invert if required
    # -------------------------------

    if invert:
        img = ImageOps.invert(img)



    # -------------------------------
    # Crop Character Region
    # -------------------------------

    arr = np.array(img)


    # Detect dark pixels

    mask = arr < 200


    if np.any(mask):

        coords = np.argwhere(mask)


        y1,x1 = coords.min(axis=0)
        y2,x2 = coords.max(axis=0)


        img = img.crop(
            (
                x1,
                y1,
                x2+1,
                y2+1
            )
        )



    # -------------------------------
    # Add stroke thickness
    # -------------------------------

    if stroke_boost > 0:

        for _ in range(stroke_boost):

            img = img.filter(
                ImageFilter.MaxFilter(3)
            )



    # -------------------------------
    # Place in square canvas
    # -------------------------------

    img.thumbnail(
        (100,100)
    )


    canvas = Image.new(
        "L",
        (100,100),
        255
    )


    x = (100-img.width)//2
    y = (100-img.height)//2


    canvas.paste(
        img,
        (x,y)
    )



    # -------------------------------
    # Convert to binary
    # -------------------------------

    arr = np.array(canvas)


    binary = np.where(
        arr < 180,
        255,
        0
    ).astype(np.uint8)



    # -------------------------------
    # Resize to 3x5
    # -------------------------------

    small = Image.fromarray(binary).resize(
        (3,5),
        Image.Resampling.BOX
    )


    small = np.array(small)



    # -------------------------------
    # Convert intensity to pattern
    # -------------------------------

    pattern = np.zeros_like(
        small,
        dtype=int
    )


    pattern[small >= (255*on_threshold)] = 1



    # Flatten 15 pixels

    pattern = pattern.flatten()



    # Safety check

    if pattern.size != 15:

        raise ValueError(
            f"Preprocessing failed. Expected 15 pixels but got {pattern.size}"
        )



    # -------------------------------
    # Return
    # -------------------------------

    if debug:

        return (
            pattern,
            binary,
            small
        )



    if mode=="Binary":

        return pattern



    else:

        return np.where(
            pattern==0,
            -1,
            1
        )
