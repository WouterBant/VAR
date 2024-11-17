import os
import cv2
import numpy as np
from typing import Dict
import textwrap
from IPython import get_ipython

try:
    if "get_ipython" in globals() and "IPKernelApp" in get_ipython().config:
        from marker_detection import MarkerDetection
        from localization import Localization
        from robot_detection import RobotDetection
        from movement_contoller import MovementController
except AttributeError:
    from .marker_detection import MarkerDetection
    from .localization import Localization
    from .robot_detection import RobotDetection
    from .movement_contoller import MovementController


class PipeLine:
    def __init__(self, config):
        self.config: Dict[any, any] = config
        self.marker_detector = MarkerDetection(config=self.config)
        self.localization = Localization(config=self.config)
        self.robot_detector = RobotDetection(config=self.config)
        self.movement_controller = MovementController(config=self.config)
        self.frame_nmbr = 0

    def run(self, cv_image):
        if self.config.get("detect_marker"):
            marker_detection_results = self.marker_detector.detect(cv_image)
            location = self.localization.triangulate(marker_detection_results)
            if self.config.get("debug") > 2:
                print(f"Location: {location}")
            if self.config.get("save_images"):
                os.makedirs("marker_images", exist_ok=True)
                frame = marker_detection_results["frame"]
                cv2.imwrite(
                    f"marker_images/{self.frame_nmbr}.jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )

        if self.config.get("detect_robot"):
            img, mask = self.robot_detector.detect(cv_image)
            if self.config.get("debug") > 2:
                print(f"Robot detection: {img}, {mask}")
            if self.config.get("save_images"):
                os.makedirs("robot_images", exist_ok=True)
                cv2.imwrite(f"robot_images/{self.frame_nmbr}.jpg", img)
                cv2.imwrite(f"robot_images/{self.frame_nmbr}_mask.jpg", mask)

        # inDanger, distances = False, None
        # if self.config.get("avoid_obstacles"):
        #     inDanger, distances = self.object_detector.detect(cv_image)
        cmd = self.movement_controller.move_to_target(
            location, (False, set())
        )  # TODO false, set should be replaced by the output of robot detector
        if self.config.get("save_images"):
            self.save_movement_image(cv_image, cmd)
        self.frame_nmbr += 1
        return cmd

    def save_movement_image(self, cv_image, cmd):
        os.makedirs("movement_images", exist_ok=True)
        title = (
            f"Linear: {cmd.linear.x:.2f}, "
            f"Angular: {cmd.angular.z:.2f},\n"
            f"Current Position: {self.localization.previous_location},\n"
            f"Target Position: ({self.config.get('target_x_location')}, {self.config.get('target_y_location')})"
        )
        print(title)
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        title = (
            f"Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}, "
            f"Current Position: {self.localization.previous_location.tolist()}, "
            f"Target Position: ({self.config.get('target_x_location')}, {self.config.get('target_y_location')})"
        )
        wrapped = textwrap.wrap(title, width=50)
        line_h = cv2.getTextSize("T", font, font_scale, thickness)[0][1] + 20
        canvas = np.full(
            (line_h * len(wrapped) + cv_image.shape[0], cv_image.shape[1], 3),
            (0, 0, 0),
            dtype=np.uint8,
        )

        for i, line in enumerate(wrapped):
            size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            cv2.putText(
                canvas,
                line,
                ((canvas.shape[1] - size[0]) // 2, int(0.9 * line_h * (i + 1))),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        canvas[len(wrapped) * line_h :] = cv_image
        cv2.imwrite(
            f"movement_images/movement_{self.frame_nmbr}.png",
            canvas,
        )

    def __call__(self, cv_image):
        return self.run(cv_image)
