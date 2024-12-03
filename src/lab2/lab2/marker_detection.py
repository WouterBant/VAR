import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
import os

try:
    if "get_ipython" in globals() and "IPKernelApp" in get_ipython().config:
        from consts import MARKER_ID_2_LOCATION, ARUCO_DICT
except AttributeError:
    from .consts import MARKER_ID_2_LOCATION, ARUCO_DICT


class MarkerDetection:
    def __init__(self, config):
        self.config = config
        self.aruco_params = self._initialize_aruco_params()
        self.frame_nmbr = 0

    def _initialize_aruco_params(self):
        aruco_params = cv2.aruco.DetectorParameters()
        if not self.config.get("use_custom_detector_parameters"):
            return aruco_params
        detector_config = self.config.get("detector_parameters", {})
        for key, value in detector_config.items():
            print(f"Setting {key} to {value}")
            setattr(aruco_params, key, value)

    def detect(self, frame):
        if self.config.get("save_images"):
            os.makedirs("input_images", exist_ok=True)
            cv2.imwrite(
                f"input_images/{self.frame_nmbr}.jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
            self.frame_nmbr += 1
        resize_factor = self.config.get("resize_factor", 3)
        frame_size = (frame.shape[1] * resize_factor, frame.shape[0] * resize_factor)
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_CUBIC)

        all_corners, all_marker_ids, all_distances = [], [], []
        all_sizes, all_tvecs, all_rvecs = [], [], []

        for desired_aruco_dictionary in ARUCO_DICT.keys():
            this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(
                ARUCO_DICT[desired_aruco_dictionary]
            )

            (corners, ids, rejected) = cv2.aruco.detectMarkers(
                frame, this_aruco_dictionary, parameters=self.aruco_params
            )

            assert (
                ids is None or len(corners) == len(ids)
            ), f"Number of corners and IDs do not match, corners: {len(corners)}, ids: {len(ids)}"

            if ids is None:
                continue

            K = (
                np.array(
                    [
                        [290.46301, 0.0, 312.90291],
                        [0.0, 290.3703, 203.01488],
                        [0.0, 0.0, 1.0],
                    ]
                )
                * resize_factor
            )
            K[2, 2] = 1.0

            for marker_corner, marker_id in zip(corners, ids):
                if marker_id is not None and marker_id[0] in MARKER_ID_2_LOCATION:
                    marker_size = MARKER_ID_2_LOCATION[marker_id[0]].height
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        marker_corner,
                        marker_size,
                        K,
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

                    if abs(tvecs[0][0][1]) > 120:
                        print(f"skipping marker {marker_id[0]}")
                        continue

                    if tvecs[0][0][2] < MARKER_ID_2_LOCATION[marker_id[0]].z:
                        print(f"skipping marker {marker_id[0]}")
                        continue

                    distance_ground = np.sqrt(
                        tvecs[0][0][2] ** 2 - MARKER_ID_2_LOCATION[marker_id[0]].z ** 2
                    )

                    if abs(tvecs[0][0][0]) > distance_ground:
                        print(f"skipping marker {marker_id[0]}")
                        print("h")
                        continue

                    all_corners.append(marker_corner[0])
                    all_marker_ids.append(marker_id[0])
                    all_tvecs.append(tvecs[0][0])
                    all_rvecs.append(rvecs[0][0])

                    all_distances.append(tvecs[0][0][2])
                    all_sizes.append(marker_size)

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
            # pass
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return {
            "corners": all_corners,
            "marker_ids": all_marker_ids,
            "marker_distances": all_distances,
            "marker_sizes": all_sizes,
            "frame": rgb_frame,
            "tvecs": all_tvecs,
            "rvecs": all_rvecs,
        }
