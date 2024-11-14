import cv2
import matplotlib.pyplot as plt
import numpy as np


ARUCO_DICT = {
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_ARUCO_MIP_36h12": cv2.aruco.DICT_ARUCO_MIP_36h12,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

MARKER_ID_2_SIZE = {
    0: 0.1,
    1: 0.1,
    2: 0.1,
}


class MarkerDetection:
    def __init__(self, config):
        self.config = config
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

    def detect(self, frame):
        all_corners, all_marker_ids, all_distances, all_sizes = [], [], [], []
        for desired_aruco_dictionary in ARUCO_DICT.keys():
            this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(
                ARUCO_DICT[desired_aruco_dictionary]
            )
            this_aruco_parameters = cv2.aruco.DetectorParameters()
            this_aruco_parameters.adaptiveThreshWinSizeMin = 3
            # this_aruco_parameters.adaptiveThreshWinSizeMax = 15
            # this_aruco_parameters.adaptiveThreshWinSizeStep = 4
            # this_aruco_parameters.minMarkerPerimeterRate = 0.04
            # this_aruco_parameters.minMarkerPerimeterRate = 0.01
            this_aruco_parameters.maxMarkerPerimeterRate = 4.0
            this_aruco_parameters.polygonalApproxAccuracyRate = 0.02
            # this_aruco_parameters.minMarkerDistanceRate = 0.5

            (corners, ids, rejected) = cv2.aruco.detectMarkers(
                frame, this_aruco_dictionary, parameters=this_aruco_parameters
            )

            assert (
                ids is None or len(corners) == len(ids)
            ), f"Number of corners and IDs do not match, corners: {len(corners)}, ids: {len(ids)}"

            if ids is None:
                continue

            for marker_corner, marker_id in zip(corners, ids):
                if marker_id is not None:
                    marker_size = MARKER_ID_2_SIZE.get(
                        marker_id[0], 30.0
                    )  # TODO fix this filter out impossible ids
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        marker_corner,
                        marker_size,
                        np.array(
                            [
                                [290.46301, 0.0, 312.90291],
                                [0.0, 290.3703, 203.01488],
                                [0.0, 0.0, 1.0],
                            ]
                        ),
                        np.array(
                            [
                                -2.79797e-01,
                                6.43090e-02,
                                -6.80000e-05,
                                1.96700e-03,
                                0.00000e00,
                            ]
                        ),
                    )

                    assert (
                        len(marker_id) == 1
                    ), f"More than one marker id detected: {marker_id}"
                    assert (
                        len(marker_corner) == 1
                    ), f"More than one marker corner detected: {marker_corner}"

                    all_corners.extend(marker_corner)
                    all_marker_ids.append(marker_id[0])
                    all_distances.append(np.linalg.norm(tvecs))
                    all_sizes.append(marker_size)

            if self.config.get("notebook_display"):
                print(f"Marker IDs: {all_marker_ids}")
                print(f"Marker Sizes: {all_sizes}")
                print(f"Distances: {all_distances}")
                print(f"Corners: {all_corners}")

        for marker_corner, marker_id in zip(all_corners, all_marker_ids):
            corners = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners

            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))

            cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(marker_id),
                (top_left[0], top_left[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        if self.config.get("notebook_display"):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_frame)
            plt.show()
        else:
            cv2.imshow("Frame", frame)
            key = (
                cv2.waitKey(1) & 0xFF
            )  # Wait for 1ms to allow for frame update and get the pressed key
            if key == ord("q"):  # Press 'q' to quit the display window
                cv2.destroyAllWindows()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return {
            "corners": all_corners,
            "marker_ids": all_marker_ids,
            "marker_distances": all_distances,
            "marker_sizes": all_sizes,
            "frame": rgb_frame,
        }
