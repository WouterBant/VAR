import rclpy
from rclpy.node import Node
import os
import yaml
from sensor_msgs.msg import CompressedImage, Image
from collections import deque as Deque
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from .pipeline import PipeLine


class MazeNode(Node):
    def __init__(self):
        super().__init__("marker_detector_node")

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

        self.pipeline = PipeLine(config=self.config)

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.bridge = CvBridge()

        self.queue: Deque[Twist] = Deque(maxlen=5)
        self.timer = self.create_timer(1.05, self.timer_callback)

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

    def image_callback(self, msg):
        if self.config.get("use_compressed_images"):
            image = self.bridge.compressed_imgmsg_to_cv2(msg)
        else:
            image = self.bridge.imgmsg_to_cv2(msg)
        cmd = self.pipeline(image)
        self.queue.append(cmd)

    def timer_callback(self):
        if len(self.queue) == 0:
            return
        average_linear = sum(i.linear.x for i in self.queue) / len(self.queue)
        average_angular = sum(i.angular.z for i in self.queue) / len(self.queue)
        self.queue.clear()
        twist_msg = Twist()
        twist_msg.linear.x = average_linear  # Forward/backward
        twist_msg.angular.z = average_angular  # Rotation angle per second
        print(f"Linear: {twist_msg.linear.x}, Angular: {twist_msg.angular.z}")
        self.cmd_vel_pub.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MazeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
