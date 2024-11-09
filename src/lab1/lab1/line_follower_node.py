#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from lab1.line_follower import LineFollower
import numpy as np
import cv2
import os
import yaml


class LineFollowerNode(Node):
    def __init__(self):
        super().__init__("line_following_node")

        # Initialize LineFollower class
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "configs",
            "lab1",
            "line_follower_config.yaml",
        )
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.line_follower = LineFollower(config=self.config)

        # Create publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)  # TODO set this to one

        # Subscribe to camera image
        # self.image_sub = self.create_subscription(
        #     Image, "/rae/right/image_raw", self.image_callback, 1  # TODO set this to one
        # )
        self.image_sub = self.create_subscription(
            CompressedImage, "/rae/right/image_raw/compressed", self.image_callback, 10
        )

        self.bridge = CvBridge()

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process image through your pipeline
        linear_vel, angular_vel = self.line_follower.pipeline(cv_image)

        # Create and publish Twist message
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel  # Forward/backward
        twist_msg.angular.z = angular_vel  # Rotation angle per second

        self.cmd_vel_pub.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
