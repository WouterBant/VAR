from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
import os
import cv2

ARUCO_DICT = {
    "DICT_ARUCO_MIP_36h12": cv2.aruco.DICT_ARUCO_MIP_36h12,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

GRASS_OFFSET = 1.8
ROBOT_CAMERA_OFFSET = 2

path_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "..",
    "assets",
    "dijkstra_wall_avoidance_path.npy",
)
# path_path = os.path.join(
#     os.path.dirname(__file__),
#     "..",
#     "..",
#     "..",
#     "assets",
#     "dijkstra_wall_avoidance_path.npy",
# )
PATH = np.load(path_path)
# reverse path
# PATH = PATH[::-1]
print(PATH)


X_OFFSET = 20
Y_OFFSET = 10
POSITIONS: List[Dict[str, Any]] = [
    {"height": 8, "z": 11.5, "y": 0, "x": 7, "ids": [82]},
    {"height": 8, "z": 11, "y": 0, "x": 41.5, "ids": [80]},
    {"height": 8, "z": 11.5, "y": 0, "x": 67.5, "ids": [92]},

    {"height": 8, "z": 11.5, "y": 0, "x": 94, "ids": [87]},
    {"height": 8, "z": 11.5, "y": 0, "x": 130, "ids": [85]},
    {"height": 8, "z": 11.5, "y": 0, "x": 158, "ids": [91]},

    {"height": 8, "z": 11.5, "y": 0, "x": 179, "ids": [89]},
    {"height": 8, "z": 11.5, "y": 0, "x": 213, "ids": [79]},
    {"height": 8, "z": 11.5, "y": 0, "x": 226, "ids": [128]},
    {"height": 8, "z": 11.5, "y": 0, "x": 239, "ids": [86]},

    {"height": 8, "z": 6, "y": 9, "x": 240, "ids": [78]},
    {"height": 8, "z": 6, "y": 39, "x": 240, "ids": [84]},
    {"height": 8, "z": 6, "y": 68, "x": 240, "ids": [90]},

    {"height": 8, "z": 6, "y": 85, "x": 266, "ids": [67]},
    {"height": 8, "z": 6, "y": 85, "x": 290, "ids": [57]},
    {"height": 8, "z": 6, "y": 85, "x": 312, "ids": [64]},
    {"height": 8, "z": 6, "y": 85, "x": 327, "ids": [129]},

    {"height": 8, "z": 6.5, "y": 95, "x": 320, "ids": [65]},
    {"height": 8, "z": 6.5, "y": 115, "x": 320, "ids": [71]},
    {"height": 8, "z": 6.5, "y": 140, "x": 320, "ids": [72]},

    {"height": 8, "z": 11, "y": 184, "x": 320, "ids": [115]},
    {"height": 8, "z": 11, "y": 210, "x": 320, "ids": [114]},
    {"height": 8, "z": 11, "y": 237, "x": 320, "ids": [122]},

    {"height": 8, "z": 11, "y": 98, "x": 85, "ids": [94]},
    {"height": 8, "z": 11, "y": 132, "x": 85, "ids": [93]},
    {"height": 8, "z": 11, "y": 167, "x": 85, "ids": [106]},

    {"height": 8, "z": 11, "y": 93, "x": 85, "ids": [113]},
    {"height": 8, "z": 11, "y": 127, "x": 85, "ids": [117]},
    {"height": 8, "z": 11, "y": 161, "x": 85, "ids": [116]},

    {"height": 8, "z": 7, "y": 92, "x": 0, "ids": [95]},
    {"height": 8, "z": 7, "y": 128, "x": 0, "ids": [118]},
    {"height": 8, "z": 7, "y": 153, "x": 0, "ids": [119]},

    {"height": 8, "z": 7, "y": 173, "x": 0, "ids": [102]},
    {"height": 8, "z": 7, "y": 208, "x": 0, "ids": [105]},
    {"height": 8, "z": 7, "y": 237, "x": 0, "ids": [104]},

    {"height": 8, "z": 7, "y": 259, "x": 0, "ids": [38]},
    {"height": 8, "z": 7, "y": 292, "x": 0, "ids": [18]},
    {"height": 8, "z": 7, "y": 321, "x": 0, "ids": [58]},

    {"height": 8, "z": 7, "y": 344, "x": 0, "ids": [41]},
    {"height": 8, "z": 7, "y": 372, "x": 0, "ids": [0]},
    {"height": 8, "z": 7, "y": 387, "x": 0, "ids": [16]},
    {"height": 8, "z": 7, "y": 408, "x": 0, "ids": [11]},


    {"height": 8, "z": 9, "y": 410, "x": 18, "ids": [2]},
    {"height": 8, "z": 9, "y": 410, "x": 43, "ids": [13]},
    {"height": 8, "z": 9, "y": 410, "x": 72, "ids": [15]},

    {"height": 8, "z": 9, "y": 410, "x": 97, "ids": [4]},
    {"height": 8, "z": 9, "y": 410, "x": 111, "ids": [17]},
    {"height": 8, "z": 9, "y": 410, "x": 149, "ids": [5]},

    {"height": 8, "z": 9, "y": 410, "x": 172, "ids": [7]},
    {"height": 8, "z": 9, "y": 410, "x": 205, "ids": [6]},
    {"height": 8, "z": 9, "y": 410, "x": 235, "ids": [3]},

    {"height": 8, "z": 9, "y": 410, "x": 252, "ids": [74]},
    {"height": 8, "z": 9, "y": 410, "x": 281, "ids": [75]},
    {"height": 8, "z": 9, "y": 410, "x": 316, "ids": [56]},

    {"height": 8, "z": 9, "y": 410, "x": 345, "ids": [32]},
    {"height": 8, "z": 9, "y": 410, "x": 372, "ids": [34]},
    {"height": 8, "z": 9, "y": 410, "x": 400, "ids": [25]},

    {"height": 8, "z": 9, "y": 350, "x": 120, "ids": [39]},
    {"height": 8, "z": 9, "y": 350, "x": 134, "ids": [23]},
    {"height": 8, "z": 9, "y": 350, "x": 159, "ids": [40]},
    {"height": 8, "z": 9, "y": 350, "x": 178, "ids": [63]},
    {"height": 8, "z": 9, "y": 350, "x": 205, "ids": [68]},
    {"height": 8, "z": 9, "y": 350, "x": 239, "ids": [66]},

    {"height": 8, "z": 8.5, "y": 258, "x": 84, "ids": [53]},
    {"height": 8, "z": 9, "y": 297, "x": 84, "ids": [37]},
    {"height": 8, "z": 6.5, "y": 322.5, "x": 84, "ids": [54]},

    {"height": 8, "z": 9.5, "y": 255.5, "x": 84, "ids": [1]},
    {"height": 8, "z": 9.5, "y": 277.5, "x": 84, "ids": [12]},
    {"height": 8, "z": 10.5, "y": 298, "x": 84, "ids": [8]},
    {"height": 8, "z": 9.5, "y": 319.5, "x": 84, "ids": [70]},


    {"height": 8, "z": 7, "y": 329, "x": 92.5, "ids": [61]},
    {"height": 8, "z": 8, "y": 329, "x": 128, "ids": [62]},
    {"height": 8, "z": 8, "y": 329, "x": 153, "ids": [59]},

    {"height": 8, "z": 10.5, "y": 329, "x": 176, "ids": [109]},
    {"height": 8, "z": 10, "y": 329, "x": 206.5, "ids": [110]},
    {"height": 8, "z": 9, "y": 329, "x": 238, "ids": [111]},

    {"height": 8, "z": 7.5, "y": 261.5, "x": 248.5, "ids": [31]},
    {"height": 8, "z": 7.5, "y": 289, "x": 248.5, "ids": [29]},
    {"height": 8, "z": 7.5, "y": 321, "x": 248.5, "ids": [22]},

    {"height": 8, "z": 6.5, "y": 251, "x": 230.5, "ids": [60]},
    {"height": 8, "z": 6.5, "y": 251, "x": 198, "ids": [27]},
    {"height": 8, "z": 6.5, "y": 251, "x": 169, "ids": [26]},

    {"height": 8, "z": 11, "y": 255, "x": 153, "ids": [100]},
    {"height": 8, "z": 12, "y": 220, "x": 153, "ids": [99]},
    {"height": 8, "z": 12, "y": 190, "x": 153, "ids": [103]},

    {"height": 8, "z": 11.5, "y": 167, "x": 153, "ids": [108]},
    {"height": 8, "z": 12, "y": 135, "x": 153, "ids": [101]},
    {"height": 8, "z": 12, "y": 102.5, "x": 153, "ids": [83]},

    {"height": 8, "z": 13, "y": 107, "x": 153, "ids": [98]},
    {"height": 8, "z": 14, "y": 135, "x": 153, "ids": [96]},
    {"height": 8, "z": 14, "y": 149, "x": 153, "ids": [107]},

    {"height": 8, "z": 10.5, "y": 185, "x": 153, "ids": [97]},
    {"height": 8, "z": 11.5, "y": 212, "x": 153, "ids": [121]},
    {"height": 8, "z": 10.5, "y": 241, "x": 153, "ids": [120]},

    {"height": 8, "z": 10, "y": 251, "x": 169, "ids": [14]},
    {"height": 8, "z": 8.5, "y": 251, "x": 198, "ids": [10]},
    {"height": 8, "z": 7, "y": 251, "x": 230.5, "ids": [21]},

    {"height": 8, "z": 5, "y": 262, "x": 248.5, "ids": [24]},
    {"height": 8, "z": 5, "y": 289, "x": 248.5, "ids": [28]},
    {"height": 8, "z": 5, "y": 305, "x": 248.5, "ids": [55]},
    {"height": 8, "z": 5, "y": 326, "x": 248.5, "ids": [47]},

    
    {"height": 8, "z": 7.5, "y": 245, "x": 320, "ids": [73]},
    {"height": 8, "z": 7, "y": 210, "x": 320, "ids": [69]},
    {"height": 8, "z": 7.5, "y": 184, "x": 320, "ids": [33]},

    {"height": 8, "z": 8.5, "y": 170, "x": 335, "ids": [42]},
    {"height": 8, "z": 9, "y": 170, "x": 362, "ids": [30]},
    {"height": 8, "z": 8, "y": 170, "x": 393.5, "ids": [20]},

    {"height": 8, "z": 8, "y": 179.5, "x": 410, "ids": [88]},
    {"height": 8, "z": 8, "y": 201, "x": 410, "ids": [77]},
    {"height": 8, "z": 8.5, "y": 245, "x": 410, "ids": [81]},

    {"height": 8, "z": 8.5, "y": 264, "x": 410, "ids": [46]},
    {"height": 8, "z": 7.5, "y": 290, "x": 410, "ids": [44]},
    {"height": 8, "z": 7.5, "y": 335.5, "x": 410, "ids": [43]},

    {"height": 8, "z": 51.5, "y": 141, "x": 153, "ids": [127]},
    {"height": 8, "z": 51.5, "y": 135.5, "x": 85, "ids": [126]},
    {"height": 8, "z": 33, "y": 217, "x": 320, "ids": [320]},
]

@dataclass
class MarkerInfo:
    height: float
    x: float
    y: float
    z: float
    id: int

MARKER_ID_2_LOCATION: Dict[int, MarkerInfo] = {}
for pos in POSITIONS:
    for marker_id in pos["ids"]:
        if marker_id in MARKER_ID_2_LOCATION:
            raise ValueError(f"Duplicate marker id: {marker_id}")
        MARKER_ID_2_LOCATION[marker_id] = MarkerInfo(
            height=pos["height"],
            x=420-pos["x"],
            y=pos["y"],
            z=pos["z"],
            id=pos["ids"][0]
        )

if __name__ == "__main__":
    print(MARKER_ID_2_LOCATION)