import numpy as np
from scipy.optimize import least_squares

# Landmark positions (example coordinates; replace with actual values)
MARKER_ID_2_LOCATION = {
    0: [0, 0, 5],  # Example 3D position for marker 0
    1: [10, 0, 5],  # Example 3D position for marker 1
    2: [0, 10, 5],  # Example 3D position for marker 2
}


class Localization:
    def __init__(self, config):
        self.config = config
        self.previous_location = np.array([0, 0])  # 2D initial guess for (x, y)

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
        # TODO fix this .get should not be necessary
        landmarks = np.array(
            [
                MARKER_ID_2_LOCATION.get(marker_id, [3, 3, 3])
                for marker_id in marker_detection_results["marker_ids"]
            ]
        )
        distances = np.array(marker_detection_results["marker_distances"])

        # Perform least squares optimization to find the best (x, y) position
        result = least_squares(residuals, initial_guess, args=(landmarks, distances))

        # Update previous location with the new result
        self.previous_location = result.x  # 2D position (x, y)
        return result.x  # Return the estimated 2D camera position (x, y)
