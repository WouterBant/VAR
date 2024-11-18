from typing import List, Dict, Any
from dataclasses import dataclass
import cv2

ARUCO_DICT = {
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_ARUCO_MIP_36h12": cv2.aruco.DICT_ARUCO_MIP_36h12,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

GRASS_OFFSET = 1.8
ROBOT_CAMERA_OFFSET = 2

# TODO check this again
POSITIONS: List[Dict[str, Any]] = [
    {"height": 23.5, "code": "36h11-16", "z": 106.5-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": -281, "x": -502, "ids": [16]},
    {"height": 8, "code": "36h11-19", "z": 51.5-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": -184, "x": -502, "ids": [3, 151, 19]},
    {"height": 20.5, "code": "7x7-37", "z": 98.5-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": -95.5, "x": -502, "ids": [37, 95]},
    {"height": 8, "code": "36h11-29", "z": 50.5-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": 9.5, "x": -502, "ids": [29]},
    {"height": 27.5, "code": "36h11-8", "z": 104.3-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": 9.5, "x": -502, "ids": [8, 82]},
    {"height": 20.5, "code": "7x7-27", "z": 99.3-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": 295, "x": -502, "ids": [1, 27, 60, 108, 574]},
    {"height": 23.5, "code": "36h11-36", "z": 106.9-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": 394, "x": -502, "ids": [36]},
    {"height": 8, "code": "36h11-59", "z": 51-ROBOT_CAMERA_OFFSET, "y": 186.5, "x": -313, "ids": [14, 59]},  # gras
    {"height": 8, "code": "36h11-89", "z": 46-ROBOT_CAMERA_OFFSET, "y": 445.9, "x": -78, "ids": [89]},	#gras
    {"height": 8, "code": "36h11-99", "z": 49.5-ROBOT_CAMERA_OFFSET, "y": 445.9, "x": 80, "ids": [99]},	#gras
    {"height": 8, "code": "36h11-69", "z": 47-ROBOT_CAMERA_OFFSET, "y": 183.2, "x": 364.6, "ids": [69, 111]},	#gras
    {"height": 8, "code": "36h11-79", "z": 49.3-ROBOT_CAMERA_OFFSET, "y": 454.4, "x": 408.5, "ids": [79]},	#gras
    {"height": 23.5, "code": "36h11-26", "z": 103.8-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": 395, "x": 421.5, "ids": [26, 120, 181]},
    {"height": 20.5, "code": "36h12-67", "z": 95.4-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": 318, "x": 421.5, "ids": [67, 439]},
    {"height": 27.5, "code": "36h11-18", "z": 98.5-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": -5.3, "x": 421.5, "ids": [18, 129]},
    {"height": 20.5, "code": "36h12-47", "z": 96.3-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": -97, "x": 421.5, "ids": [47]},
    {"height": 8, "code": "36h11-9", "z": 47.8-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": -176.6, "x": 408.5, "ids": [9, 112]},
    {"height": 23.5, "code": "36h11-66", "z": 106.4-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": -295.5, "x": 421.5, "ids": [66]},
    {"height": 8, "code": "36h11-39", "z": 49.2-ROBOT_CAMERA_OFFSET, "y": -446.4, "x": 80, "ids": [39]},  # gras
    {"height": 8, "code": "36h11-49", "z": 49.5-ROBOT_CAMERA_OFFSET, "y": -445.2, "x": -78, "ids": [49, 213]},  # gras
    {"height": 20.5, "code": "36h12-57", "z": 100-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": -534.5, "x": 0, "ids": [57]},	
    {"height": 23.5, "code": "36h11-46", "z": 165.6-ROBOT_CAMERA_OFFSET-GRASS_OFFSET, "y": -534.5, "x": 0, "ids": [46]},
    {"height": 20.5, "code": "7x7-7", "z": 97.5-ROBOT_CAMERA_OFFSET, "y": 553.2, "x": -48, "ids": [7]},  # gras	
    {"height": 23.5, "code": "36h11-56", "z": 164-ROBOT_CAMERA_OFFSET, "y": 553.2, "x": -48, "ids": [4, 56]},  # gras
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
            x=pos["x"],
            y=pos["y"],
            z=pos["z"],
        )

if __name__ == "__main__":
    print(MARKER_ID_2_LOCATION)