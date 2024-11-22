import math


class WheelOdometry:
    def __init__(self):
        # Constants for robot geometry and wheel properties
        self.WHEEL_DIAMETER = 0.0259  # meters
        self.WHEEL_CIRCUMFERENCE = math.pi * self.WHEEL_DIAMETER  # meters
        self.WHEEL_BASE = 0.095  # distance between wheel centers in meters

        # State variables
        self.last_left_wheel_value = None
        self.last_right_wheel_value = None
        self.current_x = 0.0  # x position in centimeters
        self.current_y = 0.0  # y position in centimeters
        self.current_theta = math.pi / 2  # orientation angle

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
            return (0, 0), self.current_theta

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

        # Update last wheel values
        self.last_left_wheel_value = left_wheel_value
        self.last_right_wheel_value = right_wheel_value

        return (self.current_x, self.current_y), 180 * abs(self.current_theta - math.pi)
