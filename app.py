import streamlit as st
import pandas as pd
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

    sample = preprocess_image(uploaded_file, mode)

    st.markdown("---")

    st.subheader("Processed Input Pattern")

    st.table(sample.reshape(5,3))

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

        st.table(weight.reshape(5,3))

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
1. The uploaded image is converted into a **3×5 binary/bipolar pattern**.

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
