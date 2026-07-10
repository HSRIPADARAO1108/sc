import streamlit as st
import numpy as np
from PIL import Image

from patterns import patterns_binary
from patterns import patterns_bipolar
from hebb import HebbNetwork
from preprocess import preprocess_image


st.set_page_config(
    page_title="Hebb Character Recognition",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Character Recognition using Hebb Learning")

st.write("Recognize simple characters using Hebb Learning Rule.")

st.markdown("---")


mode = st.radio(
    "Select Data Type",
    ["Binary", "Bipolar"]
)


uploaded_file = st.file_uploader(
    "Upload Character Image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image,
             caption="Uploaded Image",
             width=200)

    if mode == "Binary":

        patterns = patterns_binary

    else:

        patterns = patterns_bipolar

    network = HebbNetwork()

    network.train(patterns)

    sample = preprocess_image(uploaded_file, mode)

    st.subheader("Input Pattern")

    st.write(sample.reshape(5,3))

    prediction, net = network.predict(sample)

    st.success(f"Predicted Character : {prediction}")

    st.info(f"Net Value : {net}")

    st.subheader("Weight Matrix")

    for label, weight in zip(network.labels, network.weights):

        st.write(f"{label}")

        st.write(weight.reshape(5,3))

        st.markdown("---")
