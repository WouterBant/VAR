from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
import os
import cv2

ARUCO_DICT = {
    # "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,  #not this one
    # "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000, # not this one
    # "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,  # NNOT THis one
    # "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,  # not this one
    # "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL, # NOT THIS ONE
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
PATH = np.load(path_path)*2
print(PATH)


X_OFFSET = 20
Y_OFFSET = 10
POSITIONS: List[Dict[str, Any]] = [
    {"height": 8, "z": 5, "y": 7, "x": 0.0, "ids": [67]},
    {"height": 8, "z": 5, "y": 31, "x": 0.0, "ids": [57]},
    {"height": 8, "z": 5, "y": 53, "x": 0.0, "ids": [64]},
    {"height": 8, "z": 5, "y": 95, "x": 0.0, "ids": [65]},
    {"height": 8, "z": 5, "y": 115, "x": 0.0, "ids": [71]},
    {"height": 8, "z": 5, "y": 140, "x": 0.0, "ids": [72]},
    {"height": 8, "z": 5, "y": 179, "x": 0.0, "ids": [73]},
    {"height": 8, "z": 5, "y": 210, "x": 0.0, "ids": [69]},
    {"height": 8, "z": 5, "y": 240, "x": 0.0, "ids": [33]},
    {"height": 8, "z": 5, "y": 267, "x": 0.0, "ids": [42]},
    {"height": 8, "z": 5, "y": 293, "x": 0.0, "ids": [30]},
    {"height": 8, "z": 5, "y": 327, "x": 0.0, "ids": [20]},
    {"height": 8, "z": 5, "y": 0.0, "x": 12, "ids": [66]},
    {"height": 8, "z": 5, "y": 0.0, "x": 44, "ids": [68]},
    {"height": 8, "z": 5, "y": 0.0, "x": 74, "ids": [63]},
    {"height": 8, "z": 5, "y": 0.0, "x": 95, "ids": [38]},
    {"height": 8, "z": 5, "y": 0.0, "x": 127, "ids": [18]},
    {"height": 8, "z": 5, "y": 0.0, "x": 158, "ids": [58]},
    {"height": 8, "z": 5, "y": 0.0, "x": 176, "ids": [41]},
    {"height": 8, "z": 5, "y": 0.0, "x": 204, "ids": [0]},
    {"height": 8, "z": 5, "y": 0.0, "x": 216, "ids": [16]},
    {"height": 8, "z": 5, "y": 0.0, "x": 248, "ids": [11]},
    {"height": 8, "z": 5, "y": 20, "x": 255, "ids": [2]},
    {"height": 8, "z": 5, "y": 44, "x": 255, "ids": [13]},
    {"height": 8, "z": 5, "y": 73, "x": 255, "ids": [15]},
    {"height": 8, "z": 10.5, "y": 98, "x": 255, "ids": [4]},
    {"height": 8, "z": 10.5, "y": 112, "x": 255, "ids": [17]},
    {"height": 8, "z": 10.5, "y": 155, "x": 255, "ids": [5]},
    {"height": 8, "z": 5, "y": 173, "x": 255, "ids": [7]},
    {"height": 8, "z": 5, "y": 208, "x": 255, "ids": [6]},
    {"height": 8, "z": 5, "y": 238, "x": 255, "ids": [3]},
    {"height": 8, "z": 5, "y": 340, "x": 92, "ids": [76]},
    {"height": 8, "z": 5, "y": 340, "x": 119, "ids": [36]},
    {"height": 8, "z": 5, "y": 340, "x": 135, "ids": [45]},
    {"height": 8, "z": 5, "y": 340, "x": 155, "ids": [35]},
    {"height": 8, "z": 5, "y": 340, "x": 176, "ids": [46]},
    {"height": 8, "z": 5, "y": 340, "x": 202, "ids": [44]},
    {"height": 8, "z": 5, "y": 340, "x": 239, "ids": [43]},
    {"height": 8, "z": 5, "y": 7, "x": 85, "ids": [61]},
    {"height": 8, "z": 5, "y": 31, "x": 85, "ids": [62]},
    {"height": 8, "z": 5, "y": 53, "x": 85, "ids": [59]},
    {"height": 8, "z": 5, "y": 7, "x": 85, "ids": [39]},
    {"height": 8, "z": 5, "y": 31, "x": 85, "ids": [23]},
    {"height": 8, "z": 5, "y": 53, "x": 85, "ids": [40]},
    {"height": 8, "z": 5, "y": 179, "x": 85, "ids": [74]},
    {"height": 8, "z": 5, "y": 210, "x": 85, "ids": [75]},
    {"height": 8, "z": 5, "y": 240, "x": 85, "ids": [56]},
    {"height": 8, "z": 5, "y": 267, "x": 85, "ids": [32]},
    {"height": 8, "z": 5, "y": 293, "x": 85, "ids": [34]},
    {"height": 8, "z": 5, "y": 327, "x": 85, "ids": [25]},
    {"height": 8, "z": 5, "y": 180, "x": 85, "ids": [9]},
    {"height": 8, "z": 5, "y": 205, "x": 85, "ids": [19]},
    {"height": 8, "z": 5, "y": 219, "x": 85, "ids": [48]},
    {"height": 8, "z": 5, "y": 265, "x": 85, "ids": [52]},
    {"height": 8, "z": 5, "y": 296, "x": 85, "ids": [50]},
    {"height": 8, "z": 5, "y": 328, "x": 85, "ids": [51]},
    {"height": 8, "z": 5, "y": 170, "x": 95, "ids": [26]},
    {"height": 8, "z": 5, "y": 170, "x": 127, "ids": [27]},
    {"height": 8, "z": 5, "y": 170, "x": 161, "ids": [60]},
    {"height": 8, "z": 5, "y": 170, "x": 95, "ids": [14]},
    {"height": 8, "z": 5, "y": 170, "x": 140, "ids": [10]},
    {"height": 8, "z": 5, "y": 170, "x": 160, "ids": [21]},
    {"height": 8, "z": 5, "y": 153, "x": 170, "ids": [22]},
    {"height": 8, "z": 5, "y": 124, "x": 170, "ids": [29]},
    {"height": 8, "z": 5, "y": 92, "x": 170, "ids": [31]},
    {"height": 8, "z": 5, "y": 93, "x": 170, "ids": [47]},
    {"height": 8, "z": 5, "y": 116, "x": 170, "ids": [55]},
    {"height": 8, "z": 5, "y": 131, "x": 170, "ids": [28]},
    {"height": 8, "z": 5, "y": 160, "x": 170, "ids": [24]},
    {"height": 8, "z": 5, "y": 255, "x": 244, "ids": [53]},
    {"height": 8, "z": 5, "y": 255, "x": 204, "ids": [37]},
    {"height": 8, "z": 5, "y": 255, "x": 177, "ids": [54]},
]

@dataclass
class MarkerInfo:
    height: float
    x: float
    y: float
    z: float

MARKER_ID_2_LOCATION: Dict[int, MarkerInfo] = {}
for pos in POSITIONS:
    for marker_id in pos["ids"]:
        if marker_id in MARKER_ID_2_LOCATION:
            raise ValueError(f"Duplicate marker id: {marker_id}")
        MARKER_ID_2_LOCATION[marker_id] = MarkerInfo(
            height=pos["height"],
            x=pos["x"] + X_OFFSET,
            y=pos["y"] + Y_OFFSET,
            z=pos["z"],
        )

if __name__ == "__main__":
    print(MARKER_ID_2_LOCATION)