import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import os
import yaml
from .pipeline import PipeLine
from sensor_msgs.msg import CompressedImage  # , Image
from control_msgs.msg import DynamicJointState
from collections import deque as Deque
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from playsound import playsound
from .wheel_odometry import WheelOdometry


class CurlingNode(Node):
    def __init__(self):
        super().__init__("marker_detector_node")

        self.load_config()
        self.pipeline = PipeLine(config=self.config)
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/rae/right/image_raw/compressed",
            self.image_callback,
            2,
        )
        # self.image_sub = self.create_subscription(
        #     Image,
        #     "/rae/right/image_raw",
        #     self.image_callback,
        #     10,
        # )

        self.joint_state_sub = self.create_subscription(
            DynamicJointState,
            "/dynamic_joint_states",
            self.joint_state_callback,
            10,
        )

        self.left_wheel_value = None
        self.right_wheel_value = None
        self.want_to_stop = 0

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.bridge = CvBridge()

        self.queue: Deque[Twist] = Deque(maxlen=5)
        self.timer = self.create_timer(1.05, self.timer_callback)

        # self.loc = (0, 0)
        # self.pose = 90
        self.wheel_odometry = WheelOdometry()
        self.d_left_right = (0, 0)

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
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        action = self.pipeline(cv_image, self.d_left_right)
        self.queue.append(action)

    def joint_state_callback(self, msg):
        self.loc, self.pose = self.wheel_odometry.joint_state_callback(msg)

    def timer_callback(self):
        if self.config["print_wheel_values"]:
            if self.left_wheel_value and self.right_wheel_value:
                pass
                print(f"Left Wheel - Value: {self.left_wheel_value[0]}")
                print(f"Right Wheel - Value: {self.right_wheel_value[0]}")

        if len(self.queue) == 0:
            return
        average_linear = sum(i.linear.x for i in self.queue) / len(self.queue)
        average_angular = sum(i.angular.z for i in self.queue) / len(self.queue)
        self.queue.clear()
        twist_msg = Twist()
        twist_msg.linear.x = average_linear  # Forward/backward
        twist_msg.angular.z = average_angular  # Rotation angle per second
        print(f"Linear: {twist_msg.linear.x}, Angular: {twist_msg.angular.z}")
        if average_linear == 0:
            self.want_to_stop += 1
            if self.want_to_stop > 5:
                playsound(r"C:\Users\woute\sound.wav")
                assert 1 == 2
        else:
            self.want_to_stop = 0
        self.cmd_vel_pub.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CurlingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
