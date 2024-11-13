import rclpy
from rclpy.node import Node
import os
import yaml
import cv2
import numpy as np
from .pipeline import PipeLine
from sensor_msgs.msg import CompressedImage
from collections import deque as Deque
from geometry_msgs.msg import Twist


class MarkerDetectorNode(Node):
    def __init__(self):
        super().__init__("marker_detector_node")

        self.load_config()
        self.pipeline = PipeLine(config=self.config)
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/rae/right/image_raw/compressed",
            self.image_callback,
            10,
        )

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)

        self.queue = Deque(maxlen=5)
        self.timer = self.create_timer(0.8, self.timer_callback)

    def load_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "configs",
            "lab2",
            "config.yaml",
        )
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        action = self.pipeline(cv_image)
        self.queue.append(action)

    def timer_callback(self):
        if len(self.queue) == 0:
            return
        average_linear = sum(i for (i, _) in self.queue) / len(self.queue)
        average_angular = sum(i for (_, i) in self.queue) / len(self.queue)
        self.queue.clear()
        twist_msg = Twist()
        twist_msg.linear.x = average_linear  # Forward/backward
        twist_msg.angular.z = average_angular  # Rotation angle per second
        self.cmd_vel_pub.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MarkerDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
