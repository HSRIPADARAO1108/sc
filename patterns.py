import numpy as np


# ==================================================
# Character Patterns
# Size: 5 rows × 3 columns = 15 inputs
# 1 = Stroke
# 0 = Background
# ==================================================


patterns_binary = {


    "L": np.array([
        1,0,0,
        1,0,0,
        1,0,0,
        1,0,0,
        1,1,1
    ], dtype=int),



    "T": np.array([
        1,1,1,
        0,1,0,
        0,1,0,
        0,1,0,
        0,1,0
    ], dtype=int),



    "H": np.array([
        1,0,1,
        1,0,1,
        1,1,1,
        1,0,1,
        1,0,1
    ], dtype=int),



    "I": np.array([
        1,1,1,
        0,1,0,
        0,1,0,
        0,1,0,
        1,1,1
    ], dtype=int),



    "E": np.array([
        1,1,1,
        1,0,0,
        1,1,1,
        1,0,0,
        1,1,1
    ], dtype=int)

}



# ==================================================
# Convert Binary → Bipolar
# 0 becomes -1
# 1 becomes +1
# ==================================================


patterns_bipolar = {}


for label, pattern in patterns_binary.items():

    patterns_bipolar[label] = np.where(
        pattern == 0,
        -1,
        1
    )



# ==================================================
# Display Helper Function
# ==================================================

def display_pattern(pattern):

    """
    Convert 15 pixel vector into 5×3 matrix
    """

    return np.array(pattern).reshape(
        5,
        3
    )



# ==================================================
# Check Pattern Size
# ==================================================

for name,pattern in patterns_binary.items():

    if pattern.size != 15:

        raise ValueError(
            f"{name} pattern size error"
        )
