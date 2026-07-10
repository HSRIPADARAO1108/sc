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

Supported Characters:

- L
- T
- H
- I
- E
"""
)

st.markdown("---")


# ------------------------------------------
# Select Mode
# ------------------------------------------

mode = st.radio(
    "Select Data Type",
    ["Binary", "Bipolar"]
)



# ------------------------------------------
# Controls
# ------------------------------------------

col1, col2 = st.columns(2)


with col1:

    invert_colors = st.checkbox(
        "Invert Colors",
        False
    )


with col2:

    show_debug = st.checkbox(
        "Show Preprocessing Steps",
        True
    )



with st.expander("Advanced Settings"):

    stroke_boost = st.slider(
        "Stroke Thickness Boost",
        0,
        5,
        2
    )


    on_threshold = st.slider(
        "Cell Sensitivity",
        0.10,
        0.60,
        0.30,
        0.05
    )


st.caption(
    "Upload a clear character image with good contrast."
)



# ------------------------------------------
# Upload
# ------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Character Image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)



# ------------------------------------------
# Main Application
# ------------------------------------------

if uploaded_file:


    try:

        image = Image.open(uploaded_file)


        st.image(
            image,
            caption="Uploaded Image",
            width=250
        )



        # Choose Dataset

        if mode=="Binary":

            patterns = patterns_binary

        else:

            patterns = patterns_bipolar



        # Train

        network = HebbNetwork()

        network.train(patterns)



        # ----------------------------------
        # Preprocess
        # ----------------------------------

        if show_debug:


            sample, cropped, resized = preprocess_image(
                uploaded_file,
                mode,
                invert=invert_colors,
                debug=True,
                stroke_boost=stroke_boost,
                on_threshold=on_threshold
            )


            st.markdown("---")

            st.subheader(
                "Preprocessing Steps"
            )


            c1,c2,c3 = st.columns(3)



            # Cropped Image

            with c1:

                st.write(
                    "**1. Cropped Image**"
                )


                crop_img = np.array(cropped)


                if crop_img.max() <= 1:

                    crop_img = (
                        crop_img * 255
                    ).astype(
                        np.uint8
                    )


                st.image(
                    crop_img,
                    width=150,
                    clamp=True
                )




            # Resize Preview

            with c2:

                st.write(
                    "**2. 5×3 Grid**"
                )


                resized = np.array(resized)


                if resized.size == 15:


                    grid = resized.reshape(
                        5,
                        3
                    )


                    preview = np.kron(
                        grid,
                        np.ones(
                            (
                                20,
                                20
                            )
                        )
                    )


                    preview = (
                        preview * 255
                    ).astype(
                        np.uint8
                    )


                    st.image(
                        preview,
                        width=150,
                        clamp=True
                    )


                else:

                    st.warning(
                        "Invalid resize size"
                    )



            # Pixel Table

            with c3:


                st.write(
                    "**3. Pixel Values**"
                )


                arr = np.array(resized)


                if arr.size == 15:


                    arr = arr.reshape(
                        5,
                        3
                    )


                    st.table(
                        arr.astype(int)
                    )


                else:

                    st.error(
                        "Invalid pixel count"
                    )



        else:


            sample = preprocess_image(
                uploaded_file,
                mode,
                invert=invert_colors,
                stroke_boost=stroke_boost,
                on_threshold=on_threshold
            )



        # ----------------------------------
        # Validate Pattern
        # ----------------------------------

        sample = np.array(sample).flatten()



        if sample.size != 15:


            st.error(
f"""
Preprocessing Error

Expected:
15 pixels

Received:
{sample.size}

Update preprocess.py output.
"""
            )

            st.stop()



        # Convert Bipolar

        if mode=="Bipolar":

            sample = np.where(
                sample==0,
                -1,
                1
            )



        # ----------------------------------
        # Input Pattern
        # ----------------------------------

        st.markdown("---")

        st.subheader(
            "Processed Input Pattern"
        )


        st.table(
            sample.reshape(
                5,
                3
            )
        )



        # ----------------------------------
        # Prediction
        # ----------------------------------

        prediction, net_values = network.predict(
            sample
        )



        st.markdown("---")


        st.success(
            f"✅ Predicted Character : {prediction}"
        )



        # ----------------------------------
        # Net Values
        # ----------------------------------

        st.subheader(
            "Net Values"
        )


        df = pd.DataFrame(
            {
                "Character":
                list(net_values.keys()),

                "Net Value":
                list(net_values.values())
            }
        )


        df = df.sort_values(
            by="Net Value",
            ascending=False
        )


        st.table(df)



        # ----------------------------------
        # Weights
        # ----------------------------------

        st.markdown("---")

        st.subheader(
            "Hebb Weight Matrices"
        )



        for label, weight in zip(
            network.labels,
            network.weights
        ):

            st.write(
                f"### Character : {label}"
            )


            weight = np.array(weight)


            if weight.size==15:

                st.table(
                    weight.reshape(
                        5,
                        3
                    )
                )


            else:

                st.write(weight)



        # ----------------------------------
        # Bias
        # ----------------------------------

        st.markdown("---")


        st.subheader(
            "Bias Values"
        )


        bias_df = pd.DataFrame(
            {
                "Character":
                network.labels,

                "Bias":
                network.bias
            }
        )


        st.table(
            bias_df
        )



        # ----------------------------------
        # Explanation
        # ----------------------------------

        st.markdown("---")


        st.subheader(
            "Explanation"
        )


        st.write(
f"""
### Working Principle

1. Input image is converted into a **5×3 binary/bipolar pattern**.

2. Hebb Learning Rule calculates weight vectors.

3. Net value calculation:

**Net = W × X + B**

Where:

- W = Weight Matrix
- X = Input Pattern
- B = Bias

4. Character having maximum net value is selected.


### Recognized Character

## {prediction}

"""
        )



    except Exception as e:


        st.error(
            "Application Error"
        )

        st.exception(e)
