import os
import cv2
import numpy as np
import textwrap
from .movement_contoller import MovementController
from .marker_detection import MarkerDetection
from .localization import Localization
from geometry_msgs.msg import Twist


class PipeLine:
    def __init__(self, config):
        self.config = config
        self.marker_detector = MarkerDetection(config=self.config)
        self.localization = Localization(config=self.config)
        self.movement_controller = MovementController(config=self.config)
        self.frame_nmbr = 0

    def pipeline(self, cv_image):
        marker_detection_results = self.marker_detector.detect(cv_image)
        location_t, pose_t = self.localization.triangulate(marker_detection_results)

        if self.config.get("save_images"):
            os.makedirs("marker_images", exist_ok=True)
            frame = marker_detection_results["frame"]
            cv2.imwrite(
                f"marker_images/{self.frame_nmbr}.jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
        cmd = self.movement_controller.move_to_target(location_t, pose_t)
        if self.config.get("save_images"):
            self.save_movement_image(marker_detection_results["frame"], cmd, pose_t)
        self.frame_nmbr += 1
        return cmd

    def save_movement_image(self, cv_image, cmd, pose):
        os.makedirs("movement_images", exist_ok=True)
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        title = (
            f"Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}, Pose: {pose}\n"
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

    def __call__(self, image):
        return self.pipeline(image)
