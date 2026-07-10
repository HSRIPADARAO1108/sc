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

st.title(
    "🧠 Hebb Character Recognition System"
)


st.write(
"""
Character Recognition using **Hebb Learning Rule**

Supported Characters:

**L | T | H | I | E**
"""
)


st.divider()



# ------------------------------------------
# Mode Selection
# ------------------------------------------

mode = st.radio(
    "Select Data Type",
    [
        "Binary",
        "Bipolar"
    ]
)



# ------------------------------------------
# Controls
# ------------------------------------------

col1,col2 = st.columns(2)


with col1:

    invert = st.checkbox(
        "Invert Image",
        False
    )


with col2:

    debug = st.checkbox(
        "Show Processing Steps",
        True
    )



with st.expander(
    "Advanced Settings"
):


    stroke_boost = st.slider(
        "Stroke Thickness",
        0,
        5,
        1
    )


    threshold = st.slider(
        "Cell Threshold",
        0.1,
        0.8,
        0.4,
        0.05
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


        # Show original

        img = Image.open(file)


        st.image(
            img,
            caption="Original Image",
            width=250
        )



        # ----------------------------------
        # Select Patterns
        # ----------------------------------

        if mode=="Binary":

            patterns = patterns_binary

        else:

            patterns = patterns_bipolar



        # ----------------------------------
        # Train Hebb Network
        # ----------------------------------

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
                stroke_boost=stroke_boost,
                on_threshold=threshold
            )


            st.divider()

            st.subheader(
                "Preprocessing Steps"
            )


            c1,c2,c3 = st.columns(3)



            # Cropped

            with c1:


                st.write(
                    "Cropped"
                )


                crop=np.array(cropped)


                crop = (
                    crop*255
                ).astype(
                    np.uint8
                )


                st.image(
                    crop,
                    width=150,
                    clamp=True
                )



            # Grid

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
                        width=150,
                        clamp=True
                    )


                else:


                    st.error(
                        "Grid size error"
                    )



            # Matrix

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

        sample=np.array(
            sample
        ).flatten()



        if sample.size != 15:


            st.error(
f"""
Invalid input.

Expected 15 pixels

Received {sample.size}
"""
            )

            st.stop()



        # Bipolar conversion

        if mode=="Bipolar":

            sample=np.where(
                sample==0,
                -1,
                1
            )



        # ----------------------------------
        # Display Input
        # ----------------------------------

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

        prediction,net_values,confidence = network.predict(
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
                list(net_values.keys()),

                "Score":
                list(net_values.values())
            }
        )


        df=df.sort_values(
            "Score",
            ascending=False
        )


        st.table(df)



        # ----------------------------------
        # Weight Matrix
        # ----------------------------------

        st.divider()


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
                np.array(weight).reshape(
                    5,
                    3
                )
            )



        # ----------------------------------
        # Explanation
        # ----------------------------------

        st.divider()


        st.subheader(
            "Algorithm"
        )


        st.write(
f"""
1. Image is converted into a 5×3 binary pattern.

2. Hebb learning rule calculates weights.

3. Net value:

Net = W × X + B

4. Highest score character is selected.

Final Character:

## {prediction}
"""
        )



    except Exception as e:


        st.error(
            "Application Error"
        )

        st.exception(e)
