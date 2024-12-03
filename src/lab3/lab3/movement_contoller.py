from geometry_msgs.msg import Twist
import math
import numpy as np
from .consts import PATH


class MovementController:
    def __init__(self, config):
        self.config = config
        self.max_angular_speed = config.get("max_angular_speed")  # rad/s

    def get_target_location(self, current_pos):
        distance = float("inf")
        for idx, location in enumerate(PATH):
            dx = location[0] - current_pos[0]
            dy = location[1] - current_pos[1]
            new_distance = math.sqrt(dx * dx + dy * dy)
            if new_distance > distance:
                if idx + self.config.get("look_ahead") < len(PATH):
                    return PATH[idx + self.config.get("look_ahead")]
                assert 1 == 2, "Arrived at the target location (1/2)"
            distance = new_distance
        assert 1 == 2, "Arrived at the target location (2/2)"

    def move_to_target(self, current_pos, pose):
        if current_pos is None:
            cmd = Twist()
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0
            return cmd

        target_location = self.get_target_location(current_pos)

        dx = target_location[0] - current_pos[0]
        dy = target_location[1] - current_pos[1]

        distance = math.sqrt(dx * dx + dy * dy)
        target_angle = np.degrees(math.atan2(dy, -dx))
        use_angle = pose - target_angle  # TODO check this

        if self.config["debug"] > 6:
            print("calculated angle: ", target_angle)
            print(f"pose: {pose}")
            print(f"Use angle: {use_angle}")

        # Normal operation with cautious checking
        cmd = self._normal_movement(distance, use_angle)
        return cmd

    def _normal_movement(self, target_angle: float) -> Twist:
        cmd = Twist()
        if abs(target_angle) > 190:
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = 0.2
        cmd.angular.z = self._get_angular_velocity(target_angle)

        # Ensure we're not exceeding max speeds
        cmd.angular.z = max(
            min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed
        )
        print(f"Linear: {cmd.linear.x}, Angular: {cmd.angular.z}")
        return cmd

    def _get_angular_velocity(self, angle_diff):
        """Get angular velocity based on heading difference"""
        kp = self.config.get("angular_kp")
        return kp * angle_diff
