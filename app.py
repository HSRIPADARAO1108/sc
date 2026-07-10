import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from patterns import patterns_binary, patterns_bipolar
from hebb import HebbNetwork
from preprocess import preprocess_image


# ------------------------------------------
# Page Setup
# ------------------------------------------

st.set_page_config(
    page_title="Hebb Character Recognition",
    page_icon="🧠",
    layout="centered"
)


# ------------------------------------------
# Title
# ------------------------------------------

st.title("🧠 Hebb Character Recognition System")

st.write(
"""
Recognition of characters using **Hebb Learning Rule**

Supported Characters:

L | T | H | I | E
"""
)


st.markdown("---")



# ------------------------------------------
# Mode
# ------------------------------------------

mode = st.radio(
    "Select Input Type",
    [
        "Binary",
        "Bipolar"
    ]
)



# ------------------------------------------
# Options
# ------------------------------------------

col1,col2 = st.columns(2)


with col1:

    invert = st.checkbox(
        "Invert Image",
        False
    )


with col2:

    debug = st.checkbox(
        "Show Processing",
        True
    )



with st.expander("Advanced"):

    stroke_boost = st.slider(
        "Stroke Boost",
        0,
        5,
        2
    )


    threshold = st.slider(
        "Sensitivity",
        0.1,
        0.8,
        0.3
    )



# ------------------------------------------
# Upload
# ------------------------------------------

file = st.file_uploader(
    "Upload Character Image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)



if file:


    try:


        # Display image

        img = Image.open(file)


        st.image(
            img,
            caption="Original Image",
            width=250
        )



        # ----------------------------------
        # Select Training Pattern
        # ----------------------------------

        if mode=="Binary":

            patterns = patterns_binary

        else:

            patterns = patterns_bipolar



        # ----------------------------------
        # Train Network
        # ----------------------------------

        network = HebbNetwork()


        # Multiple Hebb updates

        for epoch in range(5):

            network.train(patterns)



        # ----------------------------------
        # Preprocessing
        # ----------------------------------

        if debug:


            sample,cropped,resized = preprocess_image(
                file,
                mode,
                invert=invert,
                debug=True,
                stroke_boost=stroke_boost,
                on_threshold=threshold
            )


            st.markdown("---")

            st.subheader(
                "Preprocessing"
            )


            c1,c2,c3 = st.columns(3)



            with c1:


                st.write(
                    "Cropped"
                )


                crop=np.array(cropped)


                if crop.max()<=1:

                    crop=(crop*255).astype(
                        np.uint8
                    )


                st.image(
                    crop,
                    width=150,
                    clamp=True
                )



            with c2:


                st.write(
                    "5 × 3 Grid"
                )


                grid=np.array(resized)


                if grid.size==15:


                    grid=grid.reshape(
                        5,
                        3
                    )


                    preview=np.kron(
                        grid,
                        np.ones(
                            (
                                25,
                                25
                            )
                        )
                    )


                    preview=(
                        preview*255
                    ).astype(
                        np.uint8
                    )


                    st.image(
                        preview,
                        width=150
                    )



            with c3:


                st.write(
                    "Pixel Matrix"
                )


                if grid.size==15:

                    st.table(
                        grid
                    )



        else:


            sample = preprocess_image(
                file,
                mode,
                invert=invert,
                stroke_boost=stroke_boost,
                on_threshold=threshold
            )



        # ----------------------------------
        # Prepare Input
        # ----------------------------------

        sample=np.array(sample).flatten()



        if sample.size!=15:


            st.error(
                f"""
Wrong input size.

Expected 15 pixels

Received {sample.size}
"""
            )

            st.stop()



        if mode=="Bipolar":

            sample=np.where(
                sample==0,
                -1,
                1
            )



        # ----------------------------------
        # Input Display
        # ----------------------------------

        st.markdown("---")

        st.subheader(
            "Input Pattern"
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

        prediction,net_values,confidence = network.predict(
            sample
        )



        st.success(
            f"Prediction : {prediction}"
        )


        st.info(
            f"Confidence : {confidence:.2f}%"
        )



        # ----------------------------------
        # Net Scores
        # ----------------------------------

        st.subheader(
            "Character Scores"
        )


        result=pd.DataFrame(
            {
                "Character":
                net_values.keys(),

                "Score":
                net_values.values()
            }
        )


        result=result.sort_values(
            "Score",
            ascending=False
        )


        st.table(result)



        # ----------------------------------
        # Weight Display
        # ----------------------------------

        st.markdown("---")

        st.subheader(
            "Hebb Weight Matrix"
        )


        for label,weight in zip(
            network.labels,
            network.weights
        ):


            st.write(
                f"Character : {label}"
            )


            st.table(
                weight.reshape(
                    5,
                    3
                )
            )



        # ----------------------------------
        # Explanation
        # ----------------------------------

        st.markdown("---")


        st.subheader(
            "Algorithm"
        )


        st.write(
f"""
1. Image converted into 5×3 binary pattern.

2. Hebb rule updates weights.

3. Net calculation:

Net = W × X + B

4. Maximum net value character is selected.

Final Result:

{prediction}
"""
        )



    except Exception as e:


        st.error(
            "Application Error"
        )

        st.exception(e)
