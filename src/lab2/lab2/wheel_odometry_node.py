# import rclpy
# from rclpy.node import Node
# from control_msgs.msg import DynamicJointState
# from geometry_msgs.msg import Point
# import math
# from .live_map import LiveMap


# class WheelOdometryNode(Node):
#     def __init__(self):
#         super().__init__("wheel_odometry_node")

#         # Constants for robot geometry and wheel properties
#         self.WHEEL_DIAMETER = 0.0252  # meters
#         self.WHEEL_CIRCUMFERENCE = math.pi * self.WHEEL_DIAMETER  # meters
#         self.WHEEL_BASE = 0.095  # distance between wheel centers in meters

#         # Init livemap
#         self.live_map = LiveMap("/home/angelo/ros2_ws/VAR/assets/voetbalveld.jpg")

#         # State variables
#         self.last_left_wheel_value = None
#         self.last_right_wheel_value = None
#         self.last_left_velo = 0
#         self.last_right_velo = 0
#         self.current_x = 0.0  # x position in centimeters
#         self.current_y = 0.0  # y position in centimeters
#         self.current_theta = 0.0  # orientation angle

#         # Subscription to joint states
#         self.joint_state_sub = self.create_subscription(
#             DynamicJointState, "/dynamic_joint_states", self.joint_state_callback, 10
#         )

#         # Publisher for robot pose
#         self.pose_pub = self.create_publisher(Point, "/robot_pose", 10)

#     def joint_state_callback(self, msg):
#         # Extract joint values
#         joint_data = {
#             joint_name: {
#                 interface_name: value
#                 for interface_name, value in zip(interface.interface_names, interface.values)
#             }
#             for joint_name, interface in zip(msg.joint_names, msg.interface_values)
#         }

#         # Extract velocities for the left and right wheels
#         left_wheel_velocity = joint_data.get("left_wheel_joint", {}).get("velocity", 0.0)
#         right_wheel_velocity = joint_data.get("right_wheel_joint", {}).get("velocity", 0.0)

#         alpha = 0.1
#         if left_wheel_velocity > 5:
#             left_wheel_velocity = alpha * left_wheel_velocity + (1 - alpha) * self.last_left_velo
#         if right_wheel_velocity > 5:
#             right_wheel_velocity = alpha * right_wheel_velocity + (1 - alpha) * self.last_right_velo

#         # Log the velocities (optional)
#         # self.get_logger().info(f"Left Wheel Velocity: {left_wheel_velocity:.2f}")
#         # self.get_logger().info(f"Right Wheel Velocity: {right_wheel_velocity:.2f}")

#         self.last_left_velo = left_wheel_velocity
#         self.last_right_velo = right_wheel_velocity

#         joint_data = {
#             joint_name: interface.values[0]
#             for joint_name, interface in zip(msg.joint_names, msg.interface_values)
#         }

#         left_wheel_value = joint_data.get("left_wheel_joint", None) * 0.794/self.WHEEL_CIRCUMFERENCE # The amount of wheelturns)
#         right_wheel_value = joint_data.get("right_wheel_joint", None) * 0.794/self.WHEEL_CIRCUMFERENCE # The amount of wheelturns)

#         # Skip if no previous values
#         if self.last_left_wheel_value is None or self.last_right_wheel_value is None:
#             self.last_left_wheel_value = left_wheel_value
#             self.last_right_wheel_value = right_wheel_value
#             return

#         # Calculate wheel rotations since last update
#         left_rotation_diff = left_wheel_value - self.last_left_wheel_value
#         right_rotation_diff = right_wheel_value - self.last_right_wheel_value

#         # Convert rotations to distance traveled
#         left_distance = left_rotation_diff * self.WHEEL_CIRCUMFERENCE
#         right_distance = right_rotation_diff * self.WHEEL_CIRCUMFERENCE

#         # Calculate robot movement
#         linear_velocity = (left_distance + right_distance) / 2
#         # angular_velocity = (right_distance - left_distance) / self.WHEEL_BASE

#         # if abs(left_distance - right_distance) > 0.1:  # Threshold to ignore noise
#         if abs(left_wheel_velocity-right_wheel_velocity) > 1:
#             angular_velocity = (right_distance - left_distance) / self.WHEEL_BASE
#             self.current_theta += angular_velocity
#             self.current_theta = (self.current_theta + math.pi) % (2 * math.pi) - math.pi
#         else:
#             angular_velocity = 0


#         # Update robot pose
#         # self.current_theta += angular_velocity

#         # Calculate new x, y positions
#         dx = linear_velocity * math.cos(self.current_theta)
#         dy = linear_velocity * math.sin(self.current_theta)

#         self.current_x += dx #* 12  # convert to centimeters
#         self.current_y += dy #* 12  # convert to centimeters

#         # Update map
#         self.live_map.update_plot((self.current_x, self.current_y))

#         # Publish robot pose
#         pose_msg = Point()
#         pose_msg.x = self.current_x
#         pose_msg.y = self.current_y
#         self.pose_pub.publish(pose_msg)

#         # Update last wheel values
#         self.last_left_wheel_value = left_wheel_value
#         self.last_right_wheel_value = right_wheel_value

#         # Optional: Log current position
#         self.get_logger().info(
#             f"Robot Position: x={self.current_x:.2f}cm, y={self.current_y:.2f}cm"
#         )


# def main(args=None):
#     rclpy.init(args=args)
#     node = WheelOdometryNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()

import rclpy
from rclpy.node import Node
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import Point
import math
from .live_map import LiveMap
import os


class WheelOdometryNode(Node):
    def __init__(self):
        super().__init__("wheel_odometry_node")

        # Constants for robot geometry and wheel properties
        self.WHEEL_DIAMETER = 0.0252  # meters
        self.WHEEL_CIRCUMFERENCE = math.pi * self.WHEEL_DIAMETER  # meters
        self.WHEEL_BASE = 0.095  # distance between wheel centers in meters

        # Scaling factors, to fix movement/rotation speed
        self.ROTATION_SCALE = (
            0.2  # TODO: Test if this is correct value for footballfield
        )
        self.MOVEMENT_SCALE = (
            0.313479624 / 2
        )  # Seems to be correct, not 100% sure though

        image_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "assets",
            "voetbalveld.jpg",
        )
        # image_path = "/home/angelo/ros2_ws/VAR/assets/voetbalveld.jpg"
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
        self.live_map = LiveMap(image_path, config_path)

        # State variables
        self.last_left_wheel_value = None
        self.last_right_wheel_value = None
        self.current_x = 0.0  # x position in centimeters
        self.current_y = 0.0  # y position in centimeters
        self.current_theta = math.pi / 2  # orientation angle

        # Subscription to joint states
        self.joint_state_sub = self.create_subscription(
            DynamicJointState, "/dynamic_joint_states", self.joint_state_callback, 10
        )

        # Publisher for robot pose
        self.pose_pub = self.create_publisher(Point, "/robot_pose", 10)

    def joint_state_callback(self, msg):
        # Extract joint values
        joint_data = {
            joint_name: interface.values[0]
            for joint_name, interface in zip(msg.joint_names, msg.interface_values)
        }

        left_wheel_value = joint_data.get("left_wheel_joint", None)
        right_wheel_value = joint_data.get("right_wheel_joint", None)

        # Skip if no previous values
        if self.last_left_wheel_value is None or self.last_right_wheel_value is None:
            self.last_left_wheel_value = left_wheel_value
            self.last_right_wheel_value = right_wheel_value
            return

        # Calculate wheel rotations since last update
        left_rotation_diff = left_wheel_value - self.last_left_wheel_value
        right_rotation_diff = right_wheel_value - self.last_right_wheel_value

        # Convert rotations to distance traveled
        left_distance = left_rotation_diff * self.WHEEL_CIRCUMFERENCE
        right_distance = right_rotation_diff * self.WHEEL_CIRCUMFERENCE

        # Calculate robot movement
        linear_velocity = (left_distance + right_distance) / 2 * self.MOVEMENT_SCALE
        # Calculate and scale angular velocity
        angular_velocity = (right_distance - left_distance) / self.WHEEL_BASE
        scaled_angular_velocity = angular_velocity * self.ROTATION_SCALE

        # Update robot pose
        self.current_theta += scaled_angular_velocity
        self.current_theta = (self.current_theta + math.pi) % (2 * math.pi) - math.pi

        # Calculate new x, y positions
        dx = linear_velocity * math.cos(self.current_theta)
        dy = linear_velocity * math.sin(self.current_theta)

        self.current_x += dx * 100  # convert to centimeters
        self.current_y += dy * 100  # convert to centimeters

        self.live_map.update_plot((self.current_x, self.current_y))

        # Publish robot pose
        pose_msg = Point()
        pose_msg.x = self.current_x
        pose_msg.y = self.current_y
        self.pose_pub.publish(pose_msg)

        # Update last wheel values
        self.last_left_wheel_value = left_wheel_value
        self.last_right_wheel_value = right_wheel_value

        # Optional: Log current position
        self.get_logger().info(
            f"Robot Position: x={self.current_x:.2f}cm, y={self.current_y:.2f}cm, theta={math.degrees(self.current_theta):.2f}°"
        )


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
