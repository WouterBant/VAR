import numpy as np
from scipy.optimize import least_squares
from IPython import get_ipython

try:
    if "get_ipython" in globals() and "IPKernelApp" in get_ipython().config:
        from consts import MARKER_ID_2_LOCATION
    else:
        from .consts import MARKER_ID_2_LOCATION
except AttributeError:
    from .consts import MARKER_ID_2_LOCATION


class Localization:
    def __init__(self, config):
        self.config = config
        self.previous_location = np.array(
            [config.get("initial_x_location"), config.get("initial_y_location")]
        )

    def triangulate(self, marker_detection_results):
        if len(marker_detection_results["marker_ids"]) == 0:
            return self.previous_location

        # Define the residual function with camera position constrained to z=0
        def residuals(camera_position_2d, landmarks, distances):
            # Convert the 2D camera position to a 3D point at z=0
            camera_position = np.array(
                [camera_position_2d[0], camera_position_2d[1], 0]
            )
            estimated_distances = np.linalg.norm(landmarks - camera_position, axis=1)
            return estimated_distances - distances

        # Use the previous location as the initial guess for optimization
        initial_guess = self.previous_location

        # Extract 3D landmark positions and corresponding distances
        assert all(
            marker_id in MARKER_ID_2_LOCATION
            for marker_id in marker_detection_results["marker_ids"]
        ), "Some marker IDs are missing from MARKER_ID_2_LOCATION"

        landmarks = np.array(
            [
                [location.x, location.y, location.z]
                for marker_id in marker_detection_results["marker_ids"]
                for location in [MARKER_ID_2_LOCATION[marker_id]]
            ]
        )
        distances = np.array(marker_detection_results["marker_distances"])

        print(f"Landmarks: {landmarks}")
        print(f"Marker IDs: {marker_detection_results['marker_ids']}")
        print(f"Distances: {distances}")
        print(f"Initial guess: {initial_guess}")

        # Perform least squares optimization to find the best (x, y) position
        result = least_squares(residuals, initial_guess, args=(landmarks, distances))
        print(f"Optimization result: {result}")

        # Update previous location with the new result
        self.previous_location = result.x  # 2D position (x, y)
        return result.x  # Return the estimated 2D camera position (x, y)

    def exact_triangulate(self, marker_detection_results):
        if len(marker_detection_results["marker_ids"]) < 2:
            return self.previous_location

        # Get the marker IDs, 3D landmark positions, and measured distances
        marker_ids = marker_detection_results["marker_ids"]
        distances = np.array(marker_detection_results["marker_distances"])

        # Find the two closest markers
        sorted_indices = np.argsort(distances)[:2]
        closest_marker_ids = [marker_ids[i] for i in sorted_indices]
        closest_landmarks = np.array(
            [
                [
                    MARKER_ID_2_LOCATION[marker_id].x,
                    MARKER_ID_2_LOCATION[marker_id].y,
                    MARKER_ID_2_LOCATION[marker_id].z,
                ]
                for marker_id in closest_marker_ids
            ]
        )
        closest_distances = distances[sorted_indices]

        # Find the intersection points
        estimated_position = self.find_intersection_point(
            closest_distances[0],
            closest_distances[1],
            np.linalg.norm(closest_landmarks[0] - closest_landmarks[1]),
            x1=closest_landmarks[0][0],
            y1=closest_landmarks[0][1],
            x2=closest_landmarks[1][0],
            y2=closest_landmarks[1][1],
        )

        self.previous_location = estimated_position
        return estimated_position

    def find_intersection_point(self, r1, r2, R, x1, y1, x2, y2):
        part1 = 0.5 * np.array([x1 + x2, y1 + y2])
        part2 = (r1**2 - r2**2) / (2 * R**2) * np.array([x2 - x1, y2 - y1])
        rem = (
            0.5
            * np.sqrt(2 * (r1**2 + r2**2) / R**2 - (r1**2 - r2**2) ** 2 / R**4 - 1)
            * np.array([y2 - y1, x1 - x2])
        )
        pos1 = part1 + part2 + rem
        pos2 = part1 + part2 - rem
        return (
            pos1
            if np.linalg.norm(pos1 - self.previous_location)
            < np.linalg.norm(pos2 - self.previous_location)
            else pos2
        )
