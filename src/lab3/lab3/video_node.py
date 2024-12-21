import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import os
import yaml
from datetime import datetime


class VideoNode(Node):
    def __init__(self):
        super().__init__("video_node")

        self.bridge = CvBridge()

        self.create_output_directory()
        self.frame_number = 0

        self.load_config()
        if self.config.get("use_compressed_images"):
            self.image_sub = self.create_subscription(
                CompressedImage,
                "/rae/right/image_raw/compressed",
                self.image_callback,
                2,
            )
        else:
            self.image_sub = self.create_subscription(
                Image,
                "/rae/right/image_raw",
                self.image_callback,
                10,
            )

    def load_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "configs",
            "lab3",
            "config.yaml",
        )

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def create_output_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "assets",
            "mazeweek2_frames",
        )
        self.output_directory = os.path.join(base_dir, f"capture_{timestamp}")
        os.makedirs(self.output_directory, exist_ok=True)

    def image_callback(self, msg):
        if self.config.get("use_compressed_images"):
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            print("got image")
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        filename = os.path.join(
            self.output_directory,
            f"frame_{self.frame_number:06d}.{self.config.get('image_format', 'jpg')}",
        )
        cv2.imwrite(filename, cv_image)
        self.frame_number += 1


def main(args=None):
    rclpy.init(args=args)
    node = VideoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
