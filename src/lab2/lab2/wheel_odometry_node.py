# import rclpy
# from rclpy.node import Node
# from control_msgs.msg import DynamicJointState
# from geometry_msgs.msg import Pose2D
# import math
# from .live_map import LiveMap

# # Robot parameters (converted to centimeters)
# wheel_radius = 1.22  # cm
# wheel_base = 9.5    # cm

# class OdometryNode(Node):
#     def __init__(self):
#         super().__init__('robot_odometry')

#         # Initialize live map
#         self.live_map = LiveMap("/home/angelo/ros2_ws/VAR/assets/voetbalveld.jpg")

#         # Initial pose in centimeters
#         self.x = 0.0  # cm
#         self.y = 0.0  # cm
#         self.theta = math.pi / 2  # radians (90 degrees)
#         self.last_time = None

#         # Initialize wheel values
#         self.left_wheel_value = 0.0
#         self.right_wheel_value = 0.0

#         # Define the publisher
#         self.pose_publisher = self.create_publisher(Pose2D, '/robot_pose', 10)

#         # Define the subscriber
#         self.joint_states_subscription = self.create_subscription(
#             DynamicJointState,
#             '/dynamic_joint_states',
#             self.joint_state_callback,
#             10
#         )

#         self.get_logger().info('Odometry Node has been started.')

#     def joint_state_callback(self, msg):
#         # Extract the current timestamp from the message
#         current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

#         # Calculate the time difference (dt)
#         if self.last_time is not None:
#             dt = current_time - self.last_time
#         else:
#             dt = 0.0  # First message, no time difference to calculate
#         self.last_time = current_time  # Update the timestamp

#         # Extract wheel rotation data
#         joint_data = {
#             joint_name: interface.values
#             for joint_name, interface in zip(msg.joint_names, msg.interface_values)
#         }

#         new_left_wheel_value = joint_data.get("left_wheel_joint", [0.0])[0]
#         new_right_wheel_value = joint_data.get("right_wheel_joint", [0.0])[0]

#         if not new_right_wheel_value == self.right_wheel_value or not new_left_wheel_value == self.left_wheel_value:
#             # Directly assign new wheel values
#             self.left_wheel_value = joint_data.get("left_wheel_joint", [0.0])[0]
#             self.right_wheel_value = joint_data.get("right_wheel_joint", [0.0])[0]

#             # Calculate distances based on wheel rotation
#             d_left = 2 * math.pi * wheel_radius * self.left_wheel_value
#             d_right = 2 * math.pi * wheel_radius * self.right_wheel_value

#             # Optional: Apply noise threshold (ignore tiny changes)
#             if abs(d_left) < 1e-4 and abs(d_right) < 1e-4:
#                 d_left = d_right = 0.0

#             # Compute velocities
#             v = (d_right + d_left) / 2.0
#             omega = (d_right - d_left) / wheel_base

#             # Threshold small velocities
#             if abs(v) < 1e-3:
#                 v = 0.0

#             # Threshold the difference in wheel rotations for straight-line motion
#             delta_rotations = abs(new_right_wheel_value - new_left_wheel_value)
#             rotation_threshold = 0.01  # Adjust as needed

#             if delta_rotations > rotation_threshold:
#                 # Calculate angular velocity as before
#                 omega = (d_right - d_left) / wheel_base
#             else:
#                 # Assume straight-line motion, set omega to 0
#                 omega = 0.0

#             # Update pose based on the velocities and time step
#             self.x += v * math.cos(self.theta) * dt
#             self.y += v * math.sin(self.theta) * dt
#             self.theta = (self.theta + omega * dt) % (2 * math.pi)
#             if self.theta > math.pi:
#                 self.theta -= 2 * math.pi
#             elif self.theta < -math.pi:
#                 self.theta += 2 * math.pi

#             # Update the plot (optional)
#             self.live_map.update_plot((self.x, self.y))

#             # Publish updated pose
#             pose_msg = Pose2D()
#             pose_msg.x = self.x
#             pose_msg.y = self.y
#             pose_msg.theta = self.theta
#             self.pose_publisher.publish(pose_msg)

#             # Log the pose
#             self.get_logger().info(f"Published Pose: x={self.x:.2f} cm, y={self.y:.2f} cm, theta={math.degrees(self.theta):.2f} degrees")

# def main(args=None):
#     rclpy.init(args=args)
#     odometry_node = OdometryNode()
#     try:
#         rclpy.spin(odometry_node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         odometry_node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import Point
import math
# from .live_map import LiveMap


class WheelOdometryNode(Node):
    def __init__(self):
        super().__init__("wheel_odometry_node")

        # Constants for robot geometry and wheel properties
        self.WHEEL_DIAMETER = 0.0259  # meters
        self.WHEEL_CIRCUMFERENCE = math.pi * self.WHEEL_DIAMETER  # meters
        self.WHEEL_BASE = 0.095  # distance between wheel centers in meters

        # Init livemap
        # self.live_map = LiveMap("/home/angelo/ros2_ws/VAR/assets/voetbalveld.jpg")

        # State variables
        self.last_left_wheel_value = None
        self.last_right_wheel_value = None
        self.current_x = 0.0  # x position in centimeters
        self.current_y = 0.0  # y position in centimeters
        self.current_theta = 0.0  # orientation angle

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
        linear_velocity = (left_distance + right_distance) / 2
        angular_velocity = (right_distance - left_distance) / self.WHEEL_BASE

        # Update robot pose
        self.current_theta += angular_velocity

        # Calculate new x, y positions
        dx = linear_velocity * math.cos(self.current_theta)
        dy = linear_velocity * math.sin(self.current_theta)

        self.current_x += dx * 12  # convert to centimeters
        self.current_y += dy * 12  # convert to centimeters

        # Update map
        # self.live_map.update_plot((self.current_x, self.current_y))

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
            f"Robot Position: x={self.current_x:.2f}cm, y={self.current_y:.2f}cm"
        )


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
