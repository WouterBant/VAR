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
            pse: float between 0 and 180 of current orientation of the robot (0 means to left 180 to right)
        """
        is_danger, danger_positions = obstacle_detection

        if current_pos is None:  # TODO maybe not but prevents circles
            cmd = Twist()
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0
            return cmd

        # Calculate basic movement parameters
        dx = self.config.get("target_x_location") - current_pos[0]
        dy = self.config.get("target_y_location") - current_pos[1]

        if dy < 0:  # TODO stop if we are over the horizontal line (we dont drive back)
            cmd = Twist()  
            return cmd

        distance = math.sqrt(dx * dx + dy * dy)
        target_angle = np.degrees(math.atan2(dy, -dx))
        use_angle = pose - target_angle

        if self.config["debug"] > 6:
            print("calculated angle: ", target_angle)
            print(f"pose: {pose}")
            print(f"Use angle: {use_angle}")
        
        cmd = Twist()

        # If we're close enough to target, stop
        if distance < self.position_tolerance:
            self._publish_stop()
            return

        # Handle dangerous situations first
        if is_danger:
            cmd = self._handle_danger(danger_positions, use_angle)
        else:
            # Normal operation with cautious checking
            cmd = self._normal_movement(distance, use_angle, danger_positions)
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
        kp = 1/18  # TODO tune this more if needed
        return kp * angle_diff

    def _publish_stop(self):
        """Publish zero velocity command"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
