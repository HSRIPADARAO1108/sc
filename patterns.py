import numpy as np


# ----------------------------------
# Binary Patterns
# 5 rows x 3 columns = 15 pixels
# ----------------------------------

patterns_binary = {

    "L": np.array([
        1,0,0,
        1,0,0,
        1,0,0,
        1,0,0,
        1,1,1
    ]),


    "T": np.array([
        1,1,1,
        0,1,0,
        0,1,0,
        0,1,0,
        0,1,0
    ]),


    "H": np.array([
        1,0,1,
        1,0,1,
        1,1,1,
        1,0,1,
        1,0,1
    ]),


    "I": np.array([
        1,1,1,
        0,1,0,
        0,1,0,
        0,1,0,
        1,1,1
    ]),


    "E": np.array([
        1,1,1,
        1,0,0,
        1,1,1,
        1,0,0,
        1,1,1
    ])
}



# ----------------------------------
# Bipolar Conversion
# ----------------------------------

patterns_bipolar = {}


for key,value in patterns_binary.items():

    patterns_bipolar[key] = np.where(
        value==0,
        -1,
        1
    )
