import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import os
import datetime
import numpy as np


class ImageSaverNode(Node):
    def __init__(self):
        super().__init__("image_saver_node")
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/rae/right/image_raw/compressed",
            self.image_callback,
            10,
        )
        self.bridge = CvBridge()
        self.current_image = None
        self.image_directory = "saved_images"

        # Ensure the output directory exists
        os.makedirs(self.image_directory, exist_ok=True)

        self.get_logger().info(
            "ImageSaverNode is running. Press Enter to save the image."
        )

    def image_callback(self, msg):
        """Callback function to receive and store the image."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def save_image(self):
        """Save the current image with a unique filename."""
        if self.current_image is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
            filename = os.path.join(self.image_directory, f"image_{timestamp}.png")
            cv2.imwrite(filename, self.current_image)
            self.get_logger().info(f"Saved image as {filename}")
        else:
            self.get_logger().warn("No image available to save.")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if input() == "":
                node.save_image()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ImageSaverNode.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
