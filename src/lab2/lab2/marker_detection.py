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

    def undistort_image(self, img, K, dist_coeffs, model="default"):
        h, w = img.shape[:2]

        if model == "default":
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                K, dist_coeffs, (w, h), 1, (w, h)
            )
            undistorted = cv2.undistort(img, K, dist_coeffs, None, new_camera_matrix)
        elif model == "rational":
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                K, dist_coeffs, (w, h), 1, (w, h)
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                K, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
            )
            undistorted = cv2.remap(img, map1, map2, cv2.INTER_LANCZOS4)

        elif model == "thin_prism":
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                K, dist_coeffs, (w, h), 1, (w, h)
            )
            scale_factor = 1.0
            scaled_K = K.copy()
            scaled_K[0, 0] *= scale_factor
            scaled_K[1, 1] *= scale_factor
            scaled_new_camera_matrix = new_camera_matrix.copy()
            scaled_new_camera_matrix[0, 0] *= scale_factor
            scaled_new_camera_matrix[1, 1] *= scale_factor

            map1, map2 = cv2.initUndistortRectifyMap(
                scaled_K,
                dist_coeffs,
                None,
                scaled_new_camera_matrix,
                (int(w * scale_factor), int(h * scale_factor)),
                cv2.CV_32FC1,
            )
            undistorted = cv2.remap(
                img, map1, map2, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101
            )
        else:
            raise ValueError(
                "Invalid rectification model. Choose 'default', 'rational', or 'thin_prism'."
            )
        x, y, w, h = roi
        undistorted = undistorted[y : y + h, x : x + w]
        return undistorted

    def _undistort_image(self, img):
        if self.config.get("undistort_method") == "default":
            img = self.undistort_image(
                img,
                np.array(
                    [
                        [290.46301, 0.0, 312.90291],
                        [0.0, 290.3703, 203.01488],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [-2.79797e-01, 6.43090e-02, -6.80000e-05, 1.96700e-03, 0.00000e00]
                ),
            )
        elif self.config.get("undistort_method") == "rational":
            img = self.undistort_image(
                img,
                np.array(
                    [
                        [273.20605262, 0.0, 320.87089782],
                        [0.0, 273.08427035, 203.25003755],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [
                            -0.14005281,
                            -0.1463477,
                            -0.00050158,
                            0.00081933,
                            0.00344204,
                            0.17342913,
                            -0.26600101,
                            -0.00599146,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    ]
                ),
                model="rational",
            )
        elif self.config.get("undistort_method") == "thin_prism":
            img = self.undistort_image(
                img,
                np.array(
                    [
                        [274.61629303, 0.0, 305.28148118],
                        [0.0, 274.71260003, 192.29090248],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [
                            -0.29394562,
                            0.11084073,
                            -0.00548286,
                            -0.00508527,
                            -0.02123716,
                            0.0,
                            0.0,
                            0.0,
                            0.019926,
                            -0.00193285,
                            0.01534379,
                            -0.00206454,
                        ]
                    ]
                ),
                model="thin_prism",
            )
        return img

    # def get_camera_direction(self, rvecs, tvecs):
    #     # Convert rotation vector to rotation matrix
    #     rmat, _ = cv2.Rodrigues(rvecs[0])

    #     # The third column of the rotation matrix represents the camera's forward direction
    #     # in world coordinates
    #     print(rmat.T)
    #     forward_direction = rmat[:, 2]

    #     # Calculate yaw angle (rotation around Y axis) from the forward direction
    #     # Using arctangent of x/z components
    #     yaw = np.degrees(np.arctan2(-forward_direction[0], -forward_direction[2]))

    #     return yaw

    def get_camera_direction(self, rvecs, tvecs):
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvecs[0])

        # Get Euler angles (pitch, yaw, roll)
        euler_angles = -cv2.decomposeProjectionMatrix(
            np.hstack((rmat, np.zeros((3, 1))))
        )[6]

        # Get yaw (rotation around y-axis)
        yaw = euler_angles[1][0]  # Take the yaw angle

        # Normalize to -180 to 180 range
        if yaw > 180:
            yaw -= 360

        return yaw

    def get_camera_angles(self, rvecs):
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvecs[0])

        # Get Euler angles in degrees (pitch, yaw, roll)
        euler_angles = -cv2.decomposeProjectionMatrix(
            np.hstack((rmat, np.zeros((3, 1))))
        )[6]

        # Extract each angle
        pitch = euler_angles[0][0]  # X-axis rotation (up/down tilt)
        yaw = euler_angles[1][0]  # Y-axis rotation (left/right direction)
        roll = euler_angles[2][0]  # Z-axis rotation (tilt sideways)

        # Normalize angles to -180 to 180 range
        for angle in [pitch, yaw, roll]:
            if angle > 180:
                angle -= 360

        return pitch, yaw, roll

    def get_global_viewing_angle(self, robot_position, marker_position):
        """
        Calculate viewing angle in global coordinate system

        Args:
            robot_position: (x, y) tuple of robot's position in global coordinates
            marker_position: (x, y) tuple of marker's known position in global coordinates

        Returns:
            angle in degrees (-180 to 180) in global coordinate system
        """
        # Get direction vector from robot to marker
        dx = marker_position[0] - robot_position[0]
        dy = marker_position[1] - robot_position[1]

        # Calculate angle using arctan2 (handles all quadrants correctly)
        angle = np.degrees(np.arctan2(dy, dx))

        # Normalize to -180 to 180 range
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360

        return angle

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
        # frame = cv2.convertScaleAbs(frame, alpha=1.6, beta=0)  # 3, 0 works
        # frame = self._undistort_image(frame)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # _, frame = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        # frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20)
        # frame = cv2.bitwise_not(frame)
        # frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20)

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

            K = np.array(
                [
                    [290.46301, 0.0, 312.90291],
                    [0.0, 290.3703, 203.01488],
                    [0.0, 0.0, 1.0],
                ]
            )*resize_factor
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

                    # print(rvecs.shape, tvecs.shape, marker_corner.shape)
                    # print(tvecs)
                    # print(self.get_camera_direction(rvecs, tvecs), "eeeeeee")
                    # print(self.get_camera_angles(rvecs), "fffffffffff")

                    assert (
                        len(marker_id) == 1
                    ), f"More than one marker id detected: {marker_id}"
                    assert (
                        len(marker_corner) == 1
                    ), f"More than one marker corner detected: {marker_corner}"

                    # if abs(tvecs[0][0][1]) > 120:
                    #     print(f"skipping marker {marker_id[0]}")
                    #     continue

                    all_corners.extend(marker_corner)
                    all_marker_ids.append(marker_id[0])
                    all_tvecs.append(tvecs[0][0])
                    all_distances.append(
                        np.linalg.norm(tvecs)
                    )  # TODO this seems to work better
                    all_rvecs.append(rvecs[0][0])
                    # all_distances.append(
                    # tvecs[0][0][2]
                    # )  # TODO maybe this is good or the above
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
