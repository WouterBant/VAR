from geometry_msgs.msg import Twist
import math
import numpy as np
from typing import Set, Tuple


class MovementController:
    def __init__(self, config):
        self.config = config

        # Controller parameters
        self.max_linear_speed = config.get("max_linear_speed")  # m/s
        self.max_angular_speed = config.get("max_angular_speed")  # rad/s
        self.position_tolerance = config.get("position_tolerance")  # meters

        # Obstacle avoidance parameters
        self.danger_slow_factor = config.get(
            "danger_slow_factor"
        )  # Reduce speed when obstacles detected
        self.escape_turn_speed = config.get(
            "escape_turn_speed"
        )  # rad/s for escape maneuvers

    def move_to_target(self, current_pos, obstacle_detection: Tuple[bool, Set[str]], pose: float):
        """
        Move robot towards target position while avoiding obstacles

        Args:
            current_pos: tuple (x, y) of current estimated position
            target_pos: tuple (x, y) of target position
            obstacle_detection: tuple (isDanger, set of positions["left", "right", "middle"])
        """
        is_danger, danger_positions = obstacle_detection

        # Calculate basic movement parameters
        print(current_pos)
        dx = self.config.get("target_x_location") - current_pos[0]
        dy = self.config.get("target_y_location") - current_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)
        print(dy, dx)
        target_angle = np.degrees(math.atan2(dy, dx))
        print("calculated angle: ", target_angle)
        correct_angle = 180 - target_angle  # angle on left bottom side
        print(f"Correct angle: {correct_angle}")
        # if angle > 90 we are going right else left
        # the higher in case of > 90 the more we are going right
        if correct_angle > 90:
            use_angle = (correct_angle - 90) * -1
        else:
            use_angle = 90 - correct_angle
        print(f"Use angle: {use_angle}")

        # Use angle assumes that the robot is parallel with x=0 axis
        # So incorporate the pose now
        use_angle += pose
        print(f"Pose corrected angle: {use_angle}")

        use_angle /= 180
        # TODO take into current orientation of the robot, currently this want to drive from the front of the target
        # TODO make sure that if no target is detected the robot will not go circles but goes straight

        # Create Twist message
        cmd = Twist()

        # If we're close enough to target, stop
        if distance < self.position_tolerance:
            self._publish_stop()
            return

        # Handle dangerous situations first
        if is_danger:
            cmd = self._handle_danger(danger_positions, target_angle)
        else:
            # Normal operation with cautious checking
            cmd = self._normal_movement(distance, target_angle, danger_positions)
        return cmd

    def _handle_danger(self, danger_positions: Set[str], target_angle: float) -> Twist:
        """Handle dangerous situations with immediate response"""
        cmd = Twist()

        # If danger is directly ahead, prioritize avoiding it
        if "middle" in danger_positions:
            cmd.linear.x = 0.75  # Move forward slowly

            # Choose escape direction based on target angle and available space
            if "left" not in danger_positions and "right" in danger_positions:
                cmd.angular.z = self.escape_turn_speed  # Turn left
            elif "right" not in danger_positions and "left" in danger_positions:
                cmd.angular.z = -self.escape_turn_speed  # Turn right
            elif (
                "left" not in danger_positions
            ):  # Both sides clear, choose based on target
                cmd.angular.z = (
                    self.escape_turn_speed
                    if target_angle > 0
                    else -self.escape_turn_speed
                )
            else:  # Both sides blocked
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

        # Handle side dangers while moving
        elif "left" in danger_positions:
            cmd.linear.x = self.max_linear_speed * self.danger_slow_factor
            cmd.angular.z = -self.escape_turn_speed  # Turn right
        elif "right" in danger_positions:
            cmd.linear.x = self.max_linear_speed * self.danger_slow_factor
            cmd.angular.z = self.escape_turn_speed  # Turn left

        return cmd

    def _normal_movement(
        self, distance: float, target_angle: float, danger_positions: Set[str]
    ) -> Twist:
        """Normal movement with obstacle awareness"""
        cmd = Twist()

        # Move forward with heading adjustment
        cmd.linear.x = self._get_linear_velocity(distance)

        # Reduce forward speed if any dangers detected
        if len(danger_positions) > 0:
            cmd.linear.x *= self.danger_slow_factor
        cmd.angular.z = self._get_angular_velocity(target_angle)

        # Ensure we're not exceeding max speeds
        cmd.linear.x = min(cmd.linear.x, self.max_linear_speed)
        cmd.angular.z = max(
            min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed
        )
        print(f"Linear: {cmd.linear.x}, Angular: {cmd.angular.z}")

        return cmd

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _get_linear_velocity(self, distance):
        """Get linear velocity based on distance to target"""
        # Simple proportional control
        kp = 0.5
        return kp * distance

    def _get_angular_velocity(self, angle_diff):
        """Get angular velocity based on heading difference"""
        # Simple proportional control
        kp = 1.0
        return kp * angle_diff

    def _publish_stop(self):
        """Publish zero velocity command"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
