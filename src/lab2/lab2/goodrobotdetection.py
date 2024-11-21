import cv2
import numpy as np
import os


class RobotDetection:
    def __init__(self, config):
        self.counter = 0
        self.config = config

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
    
    def crop_image(self, img, x_perc, y_perc):
        h, w = img.shape[:2]
        x = int(w * x_perc) // 2
        y = int(h * y_perc)
        return img[y:, x:w-x]

    def white_image(self, img):
        h, w = img.shape[:2]

        img[:h//3, :w//8] = [255, 255, 255]
        img[:h//3, -w//8:] = [255, 255, 255]
        img[:h//4, -w//4:-w//8] = [255, 255, 255]
        img[:h//4, w//8:w//4] = [255, 255, 255]
        # img[:h//6, w//4:-w//4] = [255, 255, 255]
        return img
    
    def find_black_objects(self, img):
        image_hsv = cv2.cvtColor(
            img, cv2.COLOR_BGR2HSV
        )  # Convert to HSV for segmentation

        # Define the black color range in HSV
        lower_black = np.array([0, 0, 0])  # Start from very low values
        # upper_black = np.array([180, 255, 100]) # Limit value to capture darker areas
        upper_black = np.array([180, 255, 85])  # Limit value to capture darker areas

        # Create a binary mask where black colors are white and the rest are black
        mask = cv2.inRange(image_hsv, lower_black, upper_black)
        # mask = cv2.bitwise_not(mask)

        # Apply some morphological operations to clean up the mask
        # Erode to remove small black artifacts/ connect larger objects with dilate
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
        # mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

        # Optionally, smooth the edges of the mask/ DO NOT THINK THIS HELPS
        # mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find contours of the segmented regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def get_contour_info(self, contours, min_width=50):
        """
        For each contour, check if the bottom line width is greater than `min_width`.
        If so, calculate the middle of the contour, its width, and its height.

        :param contours: List of contours (from cv2.findContours)
        :param min_width: Minimum width of the bottom line to consider
        :return: List of tuples containing (middle_x, middle_y, width, height) for each valid contour
        """
        valid_contours_info = []

        for contour in contours:
            # Get the bounding rectangle of the contour
            x, y, width, height = cv2.boundingRect(contour)

            # Check if the bottom line width is greater than the threshold
            if width >= min_width:
                # Calculate the middle of the bottom line
                middle_x = x + width // 2
                y = y + height  # Middle of the bounding box height-wise
                
                # Append contour info
                valid_contours_info.append((middle_x, y, width, height))
        return valid_contours_info
    
    def make_decision(self, frame, contours_info):
        if len(contours_info) == 0:
            return False, set()
        height_frame, width_frame = frame.shape[:2]
        left_range = width_frame // 3
        right_range = width_frame - width_frame // 3
        in_danger = False
        positions_danger = set()
        for middle_x, y, width, height in contours_info:
            if y > height_frame // 2:
                in_danger = True
            if y > height_frame // 4:
                if middle_x in range(0, left_range):
                    positions_danger.add("left")
                elif middle_x in range(left_range, right_range):
                    positions_danger.add("middle")
                elif middle_x in range(right_range, width_frame):
                    positions_danger.add("right")
        return in_danger, positions_danger            
    
    def draw_contours(self, img, contours):
        return cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    def detect(
        self, frame, show_line=True, show_contour_area=True, show_live_detection=True
    ):
        frame = self._undistort_image(frame)
        frame = self.crop_image(frame, 0.48, 0.8)
        frame = self.white_image(frame)
        contours = self.find_black_objects(frame)
        contours_info = self.get_contour_info(contours)
        in_danger, positions_danger = self.make_decision(frame, contours_info)

        if not self.config.get("notebook_display"):
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()

        return in_danger, positions_danger

