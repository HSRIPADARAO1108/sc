import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

from patterns import patterns_binary
from patterns import patterns_bipolar
from hebb import HebbNetwork
from preprocess import preprocess_image

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Hebb Character Recognition",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Character Recognition using Hebb Learning Rule")

st.write(
    """
This project demonstrates **Character Recognition**
using the **Hebb Learning Rule**.

Supported Characters:

- L
- T
- H
- I
- E
"""
)

st.markdown("---")

# ----------------------------------------------------
# Select Mode
# ----------------------------------------------------

mode = st.radio(
    "Select Data Type",
    ["Binary", "Bipolar"]
)

# ----------------------------------------------------
# Upload Image
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Character Image",
    type=["png", "jpg", "jpeg"]
)

# ----------------------------------------------------
# Start Recognition
# ----------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    # -------------------------
    # Load Patterns
    # -------------------------

    if mode == "Binary":
        patterns = patterns_binary
    else:
        patterns = patterns_bipolar

    # -------------------------
    # Train Network
    # -------------------------

    network = HebbNetwork()

    network.train(patterns)

    # -------------------------
    # Preprocess Image
    # -------------------------

    sample = preprocess_image(uploaded_file, mode)

    st.markdown("---")

    st.subheader("Processed Input Pattern")

    st.table(sample.reshape(5, 3))

    # -------------------------
    # Prediction
    # -------------------------

    prediction, nets = network.predict(sample)

    st.markdown("---")

    st.success(f"✅ Predicted Character : {prediction}")

    # -------------------------
    # Net Values
    # -------------------------

    st.subheader("Net Values")

    df = pd.DataFrame(
        {
            "Character": list(nets.keys()),
            "Net Value": list(nets.values())
        }
    )

    st.table(df)

    st.markdown("---")

    # -------------------------
    # Weight Matrices
    # -------------------------

    st.subheader("Hebb Weight Matrix")

    for label, weight in zip(network.labels, network.weights):

        st.write(f"### Character : {label}")

        st.table(weight.reshape(5, 3))

        st.markdown("---")

    st.info(
        """
Prediction is based on the **highest Net Value**.
The character whose weight vector gives the maximum
net value is selected as the output.
"""
    )
