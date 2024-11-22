import rclpy
from rclpy.node import Node
import os
import yaml
from .pipeline import PipeLine
from sensor_msgs.msg import Image
from control_msgs.msg import DynamicJointState
from collections import deque as Deque
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from .wheel_odometry import WheelOdometry


class CurlingNode(Node):
    def __init__(self):
        super().__init__("marker_detector_node")

        self.load_config()
        self.pipeline = PipeLine(config=self.config)
        # self.image_sub = self.create_subscription(
        #     CompressedImage,
        #     "/rae/right/image_raw/compressed",
        #     self.image_callback,
        #     10,
        # )
        self.image_sub = self.create_subscription(
            Image,
            "/rae/right/image_raw",
            self.image_callback,
            10,
        )

        self.joint_state_sub = self.create_subscription(
            DynamicJointState,
            "/dynamic_joint_states",
            self.joint_state_callback,
            10,
        )

        self.left_wheel_value = None
        self.right_wheel_value = None

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.bridge = CvBridge()
        self.delay = 0

        self.queue: Deque[Twist] = Deque(maxlen=5)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.loc = (0, 0)
        self.pose = 90
        self.wheel_odometry = WheelOdometry()

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
        # config_path = "/home/angelo/ros2_ws/VAR/configs/lab2/config.yaml"
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.delay += 1
        if self.delay % 4 != 0:
            return
        action = self.pipeline(cv_image, self.loc, self.pose)
        self.queue.append(action)

    def joint_state_callback(self, msg):
        self.loc, self.pose = self.wheel_odometry.joint_state_callback(msg)

    def timer_callback(self):
        if self.config["print_wheel_values"]:
            if self.left_wheel_value and self.right_wheel_value:
                pass
                # print(f"Left Wheel - Value: {self.left_wheel_value[0]}")
                # print(f"Right Wheel - Value: {self.right_wheel_value[0]}")

        if len(self.queue) == 0:
            return
        average_linear = sum(i.linear.x for i in self.queue) / len(self.queue)
        average_angular = sum(i.angular.z for i in self.queue) / len(self.queue)
        self.queue.clear()
        twist_msg = Twist()
        # if average_linear == 0:
        #     assert 1 == 2
        twist_msg.linear.x = average_linear  # Forward/backward
        twist_msg.angular.z = average_angular  # Rotation angle per second
        self.cmd_vel_pub.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CurlingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
