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
# Data Type
# ------------------------------------------

mode = st.radio(
    "Select Data Type",
    ["Binary", "Bipolar"]
)


# ------------------------------------------
# Preprocessing Controls
# ------------------------------------------

col1, col2 = st.columns(2)


with col1:

    invert_colors = st.checkbox(
        "Invert Colors",
        value=False
    )


with col2:

    show_debug = st.checkbox(
        "Show Preprocessing Steps",
        value=True
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
        step=0.05
    )



st.caption(
"Draw the character clearly with high contrast and keep it centered."
)



# ------------------------------------------
# Upload Image
# ------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Character Image",
    type=["png","jpg","jpeg"]
)



# ------------------------------------------
# Main Logic
# ------------------------------------------

if uploaded_file:


    try:

        image = Image.open(uploaded_file)


        st.image(
            image,
            caption="Uploaded Image",
            width=250
        )


        # Select Patterns

        if mode == "Binary":

            patterns = patterns_binary

        else:

            patterns = patterns_bipolar



        # Train Network

        network = HebbNetwork()

        network.train(patterns)



        # ----------------------------------
        # Preprocessing
        # ----------------------------------


        if show_debug:


            result = preprocess_image(
                uploaded_file,
                mode,
                invert=invert_colors,
                debug=True,
                stroke_boost=stroke_boost,
                on_threshold=on_threshold
            )


            sample, cropped, resized = result



            st.markdown("---")

            st.subheader(
                "Preprocessing Steps"
            )


            c1,c2,c3 = st.columns(3)



            with c1:

                st.write(
                    "Cropped Image"
                )

                st.image(
                    cropped,
                    width=150
                )


            with c2:

                st.write(
                    "5×3 Grid"
                )


                display_img = np.array(resized)


                if display_img.size == 15:

                    display_img = display_img.reshape(5,3)


                    st.image(
                        np.kron(
                            display_img,
                            np.ones((20,20))
                        ),
                        width=150
                    )

                else:

                    st.warning(
                        "Resize output is not 5×3"
                    )



            with c3:

                st.write(
                    "Pixel Values"
                )


                arr = np.array(resized)


                if arr.size == 15:

                    arr = arr.reshape(5,3)

                    st.table(
                        arr.astype(int)
                    )

                else:

                    st.error(
                        "Invalid resized image size"
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
        # Force 15 Pixel Vector
        # ----------------------------------

        sample = np.array(sample)



        sample = sample.flatten()



        if sample.size != 15:

            st.error(
                f"""
Preprocessing Error:

Expected 15 pixels (5×3)

Received:
{sample.size}

Check preprocess.py output.
"""
            )

            st.stop()



        # Bipolar conversion

        if mode == "Bipolar":

            sample = np.where(
                sample==0,
                -1,
                1
            )



        # ----------------------------------
        # Display Pattern
        # ----------------------------------


        st.markdown("---")

        st.subheader(
            "Processed Input Pattern"
        )


        st.table(
            sample.reshape(5,3)
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
                "Character": list(net_values.keys()),

                "Net Value": list(net_values.values())
            }
        )


        df = df.sort_values(
            "Net Value",
            ascending=False
        )


        st.table(df)



        # ----------------------------------
        # Weight Matrix
        # ----------------------------------


        st.markdown("---")


        st.subheader(
            "Hebb Weight Matrices"
        )


        for label,weight in zip(
            network.labels,
            network.weights
        ):


            st.write(
                f"### Character : {label}"
            )


            w = np.array(weight)


            if w.size==15:

                st.table(
                    w.reshape(5,3)
                )

            else:

                st.write(w)



        # ----------------------------------
        # Bias
        # ----------------------------------


        st.markdown("---")


        st.subheader(
            "Bias Values"
        )


        bias_df=pd.DataFrame(
            {
                "Character":network.labels,

                "Bias":network.bias
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

1. Image is converted into a 5×3 pixel pattern.

2. Hebb learning algorithm calculates weights.

3. Net value:

\[
Net = W.X + B
\]

4. Character with maximum net value is selected.


### Final Prediction:

## {prediction}

"""
        )



    except Exception as e:


        st.error(
            "Application Error"
        )

        st.exception(e)
