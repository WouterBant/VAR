from geometry_msgs.msg import Twist
import math
import numpy as np
from .consts import PATH, GRID
import matplotlib.pyplot as plt
from .route_planning import RoutePlanning
from .utils.plot_maze import plot_path

class MovementController:
    def __init__(self, config):
        self.config = config
        self.max_angular_speed = config.get("max_angular_speed")  # rad/s
        self.path = PATH if not config.get("online_path", False) else self.get_path()
        plot_path(GRID, self.path, name="Dijkstra")

    def get_path(self):
        start = (self.config.get("initial_x_location"), self.config.get("initial_y_location"))
        end = (self.config.get("final_x_location"), self.config.get("final_y_location"))
        path = RoutePlanning.dijkstra_with_wall_distance(
            GRID, start, end, wall_weight=150.0
        )
        print(f"Path: {path}")
        return path

    def get_target_location(self, current_pos):
        distance = float("inf")
        index = -1
        for idx, location in enumerate(self.path):
            dx = location[0] - current_pos[0]
            dy = location[1] - current_pos[1]
            new_distance = math.sqrt(dx * dx + dy * dy)
            if new_distance < distance:
                distance = new_distance
                index = idx

        if index + self.config.get("look_ahead") < len(self.path):
            print(f"Length path: {len(self.path)}")
            print(f"Index: {index}")
            return self.path[index + self.config.get("look_ahead")]
        
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

        print(f"Current Location: {current_pos}, Target Location: {target_location}")
        print(f"current pose: {pose}")
        distance = math.sqrt(dx * dx + dy * dy)
        target_angle = np.degrees(math.atan2(dy, -dx))
        print(f"Target angle: {target_angle}")
        use_angle = pose - target_angle  # TODO check this

        if self.config["debug"] > 6:
            print("calculated angle: ", target_angle)
            print(f"pose: {pose}")
            print(f"Use angle: {use_angle}")

        # Normal operation with cautious checking
        cmd = self._normal_movement(use_angle)
        return cmd

    def _normal_movement(self, target_angle: float) -> Twist:
        cmd = Twist()
        if abs(target_angle) > 190:
            cmd.linear.x = 0.1
            target_angle *= -1
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
