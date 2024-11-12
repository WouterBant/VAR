import rclpy
from rclpy.node import Node
import os
import yaml
import cv2
import numpy as np
from .marker_detection import MarkerDetection
from .localization import Localization
from sensor_msgs.msg import CompressedImage
from collections import deque as Deque


class MarkerDetectorNode(Node):
    def __init__(self):
        super().__init__("marker_detector_node")

        # Initialize LineFollower class
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "configs",
            "lab2",
            "marker_detection_config.yaml",
        )
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.marker_detector = MarkerDetection(config=self.config)
        self.localization = Localization(config=self.config)

        self.image_sub = self.create_subscription(
            CompressedImage,
            "/rae/right/image_raw/compressed",
            self.image_callback,
            10,  # TODO maybe one
        )

        self.queue = Deque(maxlen=5)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        distances, coordinates = self.marker_detector.detect(cv_image)
        location = self.localization.triangulate(distances, coordinates)
        self.queue.append(location)
        


def main(args=None):
    rclpy.init(args=args)
    node = MarkerDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
