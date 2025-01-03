import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class LineFollower:
    def __init__(self, config):
        self.config = config
        self.movement = {
            "linear_speed": self.config.get("initial_linear_speed"),
            "angular_speed": self.config.get("initial_angular_speed"),
        }
        self.frame = 0
        self.last_x = None
        self.last_distance = 0

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
            # undistorted = cv2.undistort(img, K, dist_coeffs, None, new_camera_matrix)
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

    def convert_to_cv2_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.config.get("debug") > 3:
            self.display_image("Input Image", img)
        return img

    def crop_to_bottom_roi(self, img, roi_height_ratio=0.3):
        height, width = img.shape[:2]
        cropped_img = img[int(height * (1 - roi_height_ratio)) :, :]
        return cropped_img

    def crop_to_top_roi(self, img, roi_height_ratio=0.5):
        height, width = img.shape[:2]
        cropped_img = img[: int(height * roi_height_ratio), :]
        if self.config.get("debug") > 3:
            self.display_image("Cropped Image", cropped_img)
        return cropped_img

    def crop_sides(self, img, roi_width_ratio=0.8):
        height, width = img.shape[:2]
        cropped_img = img[
            :, int(width * (1 - roi_width_ratio)) : int(width * roi_width_ratio)
        ]
        if self.config.get("debug") > 3:
            self.display_image("Cropped Image", cropped_img)
        return cropped_img

    def convert_to_gray(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray

    def binarize_image(self, img):
        _, binary = cv2.threshold(
            img, self.config.get("binary_threshold"), 255, cv2.THRESH_BINARY
        )
        if self.config.get("debug") > 3:
            self.display_image("Binary Image", binary)
        return binary

    def blur_image(self, img):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        if self.config.get("debug") > 3:
            self.display_image("Blurred Image", img)
        return img

    def dilate_image(self, img):
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        if self.config.get("debug") > 3:
            self.display_image("dilated Image", img)
        return img

    def sobel_edge_detection(self, img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_magnitude = np.uint8(
            gradient_magnitude * 255 / np.max(gradient_magnitude)
        )
        edges = cv2.threshold(
            gradient_magnitude,
            self.config.get("sobel_threshold"),
            255,
            cv2.THRESH_BINARY,
        )[1]
        if self.config.get("debug") > 3:
            self.display_image("Sobel Edges", edges)
        return edges

    def canny_edge_detection(self, img):
        edges = cv2.Canny(
            img, self.config.get("canny_lower"), self.config.get("canny_upper")
        )
        if self.config.get("debug") > 3:
            self.display_image("Edges Image", edges)
        return edges

    def enhance_vertical_edges(self, img):
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        enhanced = cv2.filter2D(img, -1, kernel)
        enhanced = np.absolute(enhanced)
        if self.config.get("debug") > 3:
            self.display_image("Enhanced Vertical Edges", enhanced)
        return enhanced

    def hough_transform(self, img):
        lines = cv2.HoughLinesP(
            img,
            1,  # rho
            np.pi / 180,  # theta
            self.config.get("hough_threshold"),  # threshold
            minLineLength=self.config.get("hough_min_line_length"),
            maxLineGap=self.config.get("hough_max_line_gap"),
        )
        return lines

    def get_best_line(self, lines):
        best_line = None
        best_length = None
        best_x = None
        best_loss = float("inf")

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            x = x1 if y1 < y2 else x2
            distance_to_previous_line = abs(x - self.last_x)
            if (
                self.config.get("loss") == "distance"
                or self.config.get("loss") == "distanceLength"
            ):
                loss = distance_to_previous_line
            elif self.config.get("loss") == "length":
                loss = -length
            else:
                raise ValueError("Invalid loss function")
            if (
                self.config.get("filter_low_angle")
                <= angle
                <= self.config.get("filter_high_angle")
                and loss < best_loss
            ):
                if (
                    best_line is not None
                    and self.config.get("loss") == "distanceLength"
                    and abs(best_x - x) < 20
                    and length < best_length
                ):
                    continue
                best_loss = loss
                best_line = line
                best_x = x
                best_length = length
                if self.config.get("debug") > 1:
                    print(f"Found line with length {length} and angle {angle}")
            elif self.config.get("debug") > 1:
                print(f"Discarding line with length {length} and angle {angle}")

        if best_line is None:
            return None

        x1, y1, x2, y2 = best_line[0]
        x = x1 if y1 < y2 else x2
        self.last_x = x

        return best_line

    def action(self, best_line):
        cur_lin = self.movement["linear_speed"]
        cur_ang = self.movement["angular_speed"]

        if best_line is None:
            # slightly decrease the linear speed
            action = (0.98 * cur_lin, cur_ang)
            self.movement["linear_speed"] = action[0]
        else:
            x1, y1, x2, y2 = best_line[0]
            # make sure x1 < x2
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            line_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            line_angle = (
                (line_angle - 90) / 180 if line_angle > 0 else (90 + line_angle) / 180
            )
            line_angle *= 10  # TODO increase the angle
            lmbda = self.config.get("smooth_lambda")
            new_ang = lmbda * cur_ang + (1 - lmbda) * line_angle

            max_angular_speed = self.config.get("max_angular_speed", 0.5)
            new_ang = max(-max_angular_speed, min(max_angular_speed, new_ang))

            if self.config.get("debug") > 0:
                direction = "left" if line_angle > 0 else "right"
                print(line_angle)
                print(f"Turning {direction} with: {abs(line_angle)} rad/s")

            action = (cur_lin, new_ang)
            self.movement["angular_speed"] = action[1]

        if self.config.get("debug") > 0:
            print(f"Old action: {(cur_lin, cur_ang)} -> New action: {action}")
        return action

    def easy_action(self, best_line, width, height):
        cur_lin = self.movement["linear_speed"]
        cur_ang = self.movement["angular_speed"]

        if best_line is None:
            # slightly decrease the linear speed
            action = (0.98 * cur_lin, cur_ang)
            self.movement["linear_speed"] = action[0]
        else:
            x1, y1, x2, y2 = best_line[0]
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            line_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            line_angle = (
                (line_angle - 90) / 180 if line_angle > 0 else (90 + line_angle) / 180
            )
            line_angle = abs(line_angle)

            if self.config.get("horizon_center"):
                # x = x1 if y1 > y2 else x2  # keep horizon point center

                # project horizon point
                if y1 > y2:
                    x1, x2, y1, y2 = x2, x1, y2, y1
                # (y1 - y2) / (x1 - x2) = (y1 - height) / (x1 - x)
                # x = x1 - (y1 - height) * (x1 - x2) / (y1 - y2)
                x = x1 - (y1 - height) * (x1 - x2) / (y1 - y2)
                if self.config.get("debug") > 1:
                    print(f"{x1}, {x2}, {y1}, {y2}, {height }")
                    print(f"frame: {self.frame}")
                    print(f"Projected horizon point: {x}")
                    print(f"width: {width}")
                new_ang = (width / 2 - x) / (6 * width)
            else:
                x = x2 if y1 > y2 else x1  # keep closest point center

                if x < width / 2:  # point is too far left
                    new_ang = line_angle  # go to left to center it
                else:
                    new_ang = -line_angle
            new_ang *= 10

            cur_distance = abs(x - width / 2)
            # if self.last_distance > cur_distance:
            #     new_ang = 0  # already going to the right direction
            #     if self.config.get("debug") > 0:
            #         print("Already going in the right direction")
            self.last_distance = cur_distance

            if (cur_distance < 30 and line_angle * 180 < 10) or self.frame > 15:
                # we are now sufficiently under the line
                self.config["horizon_center"] = True  # start following the horizon

            lmbda = self.config.get("smooth_lambda")
            new_ang = lmbda * cur_ang + (1 - lmbda) * new_ang

            max_angular_speed = self.config.get("max_angular_speed", 0.5)
            new_ang = max(-max_angular_speed, min(max_angular_speed, new_ang))

            if self.config.get("debug") > 0:
                print(f"Line angle: {line_angle}")
                direction = "left" if new_ang > 0 else "right"
                print(f"Turning {direction} with: {abs(new_ang)} rad/s")

            action = (cur_lin, new_ang)
            self.movement["angular_speed"] = action[1]

        if self.config.get("debug") > 0:
            print(f"Old action: {(cur_lin, cur_ang)} -> New action: {action}")
        return action

    def display_image(self, title, image):
        plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
        plt.axis("off")
        plt.show()

    def draw_lines(self, img, lines):
        # Create a copy of the image to draw lines on
        line_image = img.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(
                    line_image, (x1, y1), (x2, y2), (0, 255, 0), 2
                )  # Draw green lines

        return line_image

    def save_action_on_best_line(self, action, img, best_line):
        x1, y1, x2, y2 = best_line[0]

        plt.figure(figsize=(10, 8))
        plt.imshow(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )  # Convert BGR to RGB for correct colors
        plt.plot([x1, x2], [y1, y2], "g-", linewidth=2)  # Draw green line

        # Set title with action information
        plt.title(f"Linear: {action[0]:.2f}, Angular: {action[1]:.2f}")

        # Remove axes for cleaner look
        plt.axis("off")

        # Save and close
        os.makedirs("video", exist_ok=True)
        plt.savefig(f"video/line_follower_{self.frame}.png")
        plt.close()

    def pipeline(self, img):
        # image loading
        img = self.convert_to_cv2_image(img)

        if self.config.get("save_images"):
            os.makedirs("video", exist_ok=True)
            plt.imsave(f"video/initial_image_{self.frame}.png", img)

        # undistort image
        # if self.config.get("undistort_image"):
        #     img = self._undistort_image(img)

        # if self.config.get("debug") > 3:
        #     self.display_image("Original Image", img)

        # # crop image to roi to avoid having to deal with unimportant lines
        # img = self.crop_to_top_roi(img)
        # img = self.crop_sides(img)
        # img_cropped = img.copy()

        # # binarize
        # img = self.convert_to_gray(img)
        # img = self.binarize_image(img)

        # # blur and dilate to remove noise and fill gaps in the
        # if self.config.get("blur_image", False):
        #     img = self.blur_image(img)
        # if self.config.get("dilate_image", False):
        #     img = self.dilate_image(img)

        # # edge detection
        # if self.config.get("method") == "canny":
        #     edges = self.canny_edge_detection(img)
        # elif self.config.get("method") == "sobel":
        #     edges = self.sobel_edge_detection(img)

        # if self.config.get("enhance_vertical_edges"):
        #     edges = self.enhance_vertical_edges(img)

        # # hough transform
        # lines = self.hough_transform(edges)

        # # get the best line
        # if self.last_x is None:
        #     self.last_x = img_cropped.shape[1] // 2
        # best_line = self.get_best_line(lines)

        # if self.config.get("debug") > 3 and best_line is not None:
        #     result = self.draw_lines(img_cropped, lines)
        #     self.display_image("Detected Lines", result)

        #     result = self.draw_lines(img_cropped, [best_line])
        #     self.display_image("Best Line", result)

        # # determine the action with the best line
        # if self.config.get("easy_action"):
        #     action = self.easy_action(
        #         best_line, width=img.shape[1], height=img.shape[0]
        #     )
        # else:
        #     action = self.action(best_line)

        # if self.config.get("save_images") > 0 and best_line is not None:
        #     result = self.draw_lines(img_cropped, [best_line])
        #     self.save_action_on_best_line(action, result, best_line)
        # self.frame += 1

        return (0, 0)
