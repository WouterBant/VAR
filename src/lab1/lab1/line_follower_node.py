#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from lab1.line_follower import LineFollower
import numpy as np
import cv2
import os


class LineFollowerNode(Node):
    def __init__(self):
        super().__init__("line_following_node")

        # Initialize your LineFollower class
        debug_mode = int(os.getenv("DEBUG", "0"))
        self.line_follower = LineFollower(debug=debug_mode)

        # Create publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.timer = self.create_timer(
            3.0, self.timer_callback
        )  # 1 Hz frequency comment out later

        # Subscribe to camera image
        self.image_sub = self.create_subscription(
            Image, "/rae/stereo_front/image_raw", self.image_callback, 10
        )

        self.bridge = CvBridge()

    def timer_callback(self):
        # cv_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv_image = cv2.imread("assets/classroom.png")
        # horizontally flip image
        cv_image = cv2.flip(cv_image, 1)

        # Process image through your pipeline
        linear_vel, angular_vel = self.line_follower.pipeline(cv_image)

        # Create and publish Twist message
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel  # Forward/backward
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = angular_vel  # Rotation angle per second

        self.cmd_vel_pub.publish(twist_msg)

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Process image through your pipeline
        linear_vel, angular_vel = self.line_follower.pipeline(cv_image)

        # Create and publish Twist message
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel  # Forward/backward
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
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
