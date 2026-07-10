import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from patterns import patterns_binary, patterns_bipolar
from hebb import HebbNetwork
from preprocess import preprocess_image

# ------------------------------------------
# Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Hebb Character Recognition",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------------------
# Title
# ------------------------------------------

st.title("🧠 Character Recognition using Hebb Learning Rule")

st.write(
"""
This application demonstrates **Character Recognition**
using the **Hebb Learning Rule**.

Supported Characters

• L
• T
• H
• I
• E
"""
)

st.markdown("---")

# ------------------------------------------
# Select Data Type
# ------------------------------------------

mode = st.radio(
    "Select Data Type",
    ["Binary", "Bipolar"]
)

# ------------------------------------------
# Preprocessing Options
# ------------------------------------------

col1, col2 = st.columns(2)

with col1:
    invert_colors = st.checkbox(
        "Invert Colors",
        value=False,
        help="Tick this if the app is picking up the background instead "
             "of your character (auto-detection guessed wrong)."
    )

with col2:
    show_debug = st.checkbox(
        "Show Preprocessing Steps",
        value=True,
        help="Displays the cropped and resized versions of your image so "
             "you can see exactly what the network is seeing."
    )

with st.expander("Advanced: fine-tune preprocessing"):
    stroke_boost = st.slider(
        "Stroke Thickness Boost", 0, 5, 2,
        help="Thickens thin pen/pencil/font strokes before downsizing to "
             "3x5, so they don't get averaged away to nothing. Increase "
             "this if characters keep coming out blank or wrong; "
             "decrease it if strokes are already thick and bars are "
             "bleeding into each other."
    )
    on_threshold = st.slider(
        "Cell Sensitivity", 0.10, 0.60, 0.30, step=0.05,
        help="How much of a 3x5 grid cell needs to be covered by the "
             "stroke for it to count as 'on'. Lower = more sensitive "
             "(good for faint/thin strokes). Higher = stricter "
             "(good if you're getting extra noisy pixels)."
    )

st.caption(
    "Tip: draw/write your character as a bold stroke with good contrast "
    "against the background, and try to keep it roughly centered in the "
    "image."
)

# ------------------------------------------
# Upload Image
# ------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Character Image",
    type=["png", "jpg", "jpeg"]
)

# ------------------------------------------
# Main Program
# ------------------------------------------

if uploaded_file is not None:

    # Display Uploaded Image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    # --------------------------------------
    # Select Patterns
    # --------------------------------------

    if mode == "Binary":
        patterns = patterns_binary
    else:
        patterns = patterns_bipolar

    # --------------------------------------
    # Train Hebb Network
    # --------------------------------------

    network = HebbNetwork()

    network.train(patterns)

    # --------------------------------------
    # Image Preprocessing
    # --------------------------------------

    if show_debug:
        sample, cropped, resized = preprocess_image(
            uploaded_file, mode, invert=invert_colors, debug=True,
            stroke_boost=stroke_boost, on_threshold=on_threshold
        )

        st.markdown("---")
        st.subheader("Preprocessing Steps")

        d1, d2, d3 = st.columns(3)
        with d1:
            st.write("**1. Cropped + stroke-boosted**")
            st.image(cropped, width=150, clamp=True)
        with d2:
            st.write("**2. Resized to 3×5**")
            st.image(
                np.kron(resized, np.ones((20, 20))),
                width=150,
                clamp=True
            )
        with d3:
            st.write("**3. Raw cell intensity (0-255)**")
            st.table(resized.astype(int).reshape(5, 3))

        # Re-derive the final pattern in the requested mode
        if mode == "Bipolar":
            sample = np.where(sample == 0, -1, 1)
    else:
        sample = preprocess_image(
            uploaded_file, mode, invert=invert_colors,
            stroke_boost=stroke_boost, on_threshold=on_threshold
        )

    st.markdown("---")

    st.subheader("Processed Input Pattern")

    st.table(sample.reshape(5, 3))

    # --------------------------------------
    # Prediction
    # --------------------------------------

    prediction, net_values = network.predict(sample)

    st.markdown("---")

    st.success(f"✅ Predicted Character : {prediction}")

    # --------------------------------------
    # Net Values
    # --------------------------------------

    st.subheader("Net Values")

    df = pd.DataFrame({
        "Character": list(net_values.keys()),
        "Net Value": list(net_values.values())
    })

    df = df.sort_values(
        by="Net Value",
        ascending=False
    )

    st.table(df)

    # --------------------------------------
    # Best Match
    # --------------------------------------

    st.info(
        f"""
Best Match : {prediction}

The character having the highest Net Value
is selected as the recognized character.
"""
    )

    # --------------------------------------
    # Weight Matrices
    # --------------------------------------

    st.markdown("---")

    st.subheader("Hebb Weight Matrices")

    for label, weight in zip(network.labels, network.weights):

        st.write(f"### Character : {label}")

        st.table(weight.reshape(5, 3))

    # --------------------------------------
    # Bias Values
    # --------------------------------------

    st.markdown("---")

    st.subheader("Bias Values")

    bias_df = pd.DataFrame({
        "Character": network.labels,
        "Bias": network.bias
    })

    st.table(bias_df)

    # --------------------------------------
    # Explanation
    # --------------------------------------

    st.markdown("---")

    st.subheader("Explanation")

    st.write(f"""
1. The uploaded image is converted into a **3×5 binary/bipolar pattern**
   (cropped to the character's bounding box, then resized).

2. The Hebb Network compares this pattern with all trained character patterns.

3. Net values are calculated using

**Net = W·X + B**

where

- W = Weight Vector
- X = Input Pattern
- B = Bias

4. The character having the **highest Net Value** is selected.

**Predicted Character = {prediction}**
""")
