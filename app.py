import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from patterns import patterns_binary, patterns_bipolar
from hebb import HebbNetwork
from preprocess import preprocess_image



# ------------------------------------------
# Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Hebb Character Recognition",
    page_icon="🧠",
    layout="centered"
)



# ------------------------------------------
# Title
# ------------------------------------------

st.title(
    "🧠 Hebb Character Recognition"
)


st.write(
"""
Character recognition using Hebb Learning Rule

Characters:

**L | T | H | I | E**
"""
)


st.divider()



# ------------------------------------------
# Mode
# ------------------------------------------

mode = st.radio(
    "Select Mode",
    [
        "Binary",
        "Bipolar"
    ]
)



# ------------------------------------------
# Options
# ------------------------------------------

invert = st.checkbox(
    "Invert Image",
    False
)


debug = st.checkbox(
    "Show Preprocessing",
    True
)



stroke = st.slider(
    "Stroke Boost",
    0,
    5,
    1
)


threshold = st.slider(
    "Threshold",
    0.1,
    0.8,
    0.4
)



# ------------------------------------------
# Upload
# ------------------------------------------

file = st.file_uploader(
    "Upload Character",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)



if file:


    try:


        image = Image.open(file)


        st.image(
            image,
            caption="Original Image",
            width=250
        )



        # Select patterns

        if mode=="Binary":

            patterns = patterns_binary

        else:

            patterns = patterns_bipolar



        # Train

        network = HebbNetwork()

        network.train(
            patterns
        )



        # ----------------------------------
        # Preprocessing
        # ----------------------------------

        if debug:


            sample,cropped,resized = preprocess_image(
                file,
                mode,
                invert=invert,
                debug=True,
                stroke_boost=stroke,
                on_threshold=threshold
            )



            st.divider()


            st.subheader(
                "Preprocessing"
            )



            c1,c2,c3 = st.columns(3)



            with c1:


                st.write(
                    "Cropped"
                )


                crop=np.array(cropped)


                crop=(
                    crop*255
                ).astype(
                    np.uint8
                )


                st.image(
                    crop,
                    width=150,
                    clamp=True
                )



            with c2:


                st.write(
                    "5×3 Grid"
                )


                grid=np.array(
                    resized
                )


                if grid.size==15:


                    grid=grid.reshape(
                        5,
                        3
                    )


                    show=np.kron(
                        grid,
                        np.ones(
                            (
                                25,
                                25
                            )
                        )
                    )


                    show=(
                        show*255
                    ).astype(
                        np.uint8
                    )


                    st.image(
                        show,
                        width=150
                    )



            with c3:


                st.write(
                    "Matrix"
                )


                st.table(
                    grid
                )



        else:


            sample = preprocess_image(
                file,
                mode,
                invert=invert,
                stroke_boost=stroke,
                on_threshold=threshold
            )



        # ----------------------------------
        # Prepare Input
        # ----------------------------------

        sample=np.array(
            sample
        ).flatten()



        if sample.size != 15:

            st.error(
                f"Expected 15 pixels but got {sample.size}"
            )

            st.stop()



        if mode=="Bipolar":

            sample=np.where(
                sample==0,
                -1,
                1
            )



        st.divider()


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

        prediction,scores,confidence = network.predict(
            sample
        )



        st.success(
            f"🎯 Prediction : {prediction}"
        )


        st.info(
            f"Confidence : {confidence}%"
        )



        # ----------------------------------
        # Scores
        # ----------------------------------

        st.subheader(
            "Character Scores"
        )


        df=pd.DataFrame(
            {
                "Character":
                list(scores.keys()),

                "Score":
                list(scores.values())
            }
        )


        df=df.sort_values(
            "Score",
            ascending=False
        )


        st.table(df)



        # ----------------------------------
        # Weight
        # ----------------------------------

        st.divider()


        st.subheader(
            "Weight Matrix"
        )


        for label,weight in network.weights.items():


            st.write(
                label
            )


            st.table(
                np.array(weight).reshape(
                    5,
                    3
                )
            )



    except Exception as e:


        st.error(
            "Application Error"
        )

        st.exception(e)
