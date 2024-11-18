import numpy as np
import math
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
        self.previous_pose = 0

    def triangulate(self, marker_detection_results):
        if len(marker_detection_results["marker_ids"]) == 0:
            return self.previous_location, self.previous_pose
        position = self.triangulate_2d_ls(marker_detection_results)
        poses = self.estimate_pose(marker_detection_results, position)
        print("poses", poses)
        if len(poses) == 0:  # TODO now we have already updated previous position
            return position, self.previous_pose
        average_pose = sum(poses) / len(poses)
        return position, average_pose

    def triangulate_2d_ls(self, marker_detection_results):
        if len(marker_detection_results["marker_ids"]) == 0:
            return self.previous_location

        def residuals(camera_position_2d, landmarks, distances):
            camera_position = np.array([camera_position_2d[0], camera_position_2d[1]])
            estimated_distances = np.linalg.norm(landmarks - camera_position, axis=1)
            return estimated_distances - distances

        initial_guess = self.previous_location

        landmarks = np.array(
            [
                [location.x, location.y]
                for marker_id in marker_detection_results["marker_ids"]
                for location in [MARKER_ID_2_LOCATION[marker_id]]
            ]
        )

        print(marker_detection_results["marker_distances"])
        print([MARKER_ID_2_LOCATION[marker_id].z for marker_id in marker_detection_results["marker_ids"]])
        distances = np.array(
            [
                np.sqrt(distance**2 - MARKER_ID_2_LOCATION[marker_id].z ** 2)
                for marker_id, distance in zip(
                    marker_detection_results["marker_ids"],
                    marker_detection_results["marker_distances"],
                )
            ]
        )

        result = least_squares(
            residuals,
            initial_guess,
            args=(landmarks, distances),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

        self.previous_location = result.x
        return result.x

    def estimate_pose(self, marker_detection_results, location):
        estimated_poses = list()
        for i in range(len(marker_detection_results["marker_ids"])):
            marker_id = marker_detection_results["marker_ids"][i]
            distance = marker_detection_results["marker_distances"][i]
            tvec = marker_detection_results["tvecs"][i]
            # TODO make sure it also works on left side
            marker_location = MARKER_ID_2_LOCATION[marker_id]
            distance_ground = np.sqrt(distance**2 - marker_location.z ** 2)
            delta_x = abs(location[0] - marker_location.x)
            delta_y = abs(location[1] - marker_location.y)
            if marker_location.y > location[1]:
                initial_angle = 90 + math.degrees(math.atan2(delta_x, delta_y))
            else:
                initial_angle = 270 - math.degrees(math.atan2(delta_x, delta_y))
            print(initial_angle, "initial")
            print(tvec, "tvec")
            print(distance_ground, "distance_ground")
            if abs(tvec[0]) > distance_ground:  # TODO this is hacky
                print("skipping invalid marker location")
                continue
            adjustment_angle = math.degrees(math.asin(abs(tvec[0]) / distance_ground))
            print(adjustment_angle, "adjustment")
            if (tvec[0] > 0 and location[0] > 0) or (tvec[0] < 0 and location[0] < 0):
                angle = initial_angle - adjustment_angle
            else:
                angle = initial_angle + adjustment_angle
            print(angle, "correct")
            if location[0] < 0:
                angle *= -1
            if angle < 0:
                angle += 90
            else:
                angle -= 90
            estimated_poses.append(angle)
        return estimated_poses

    def triangulate_3d_ls(self, marker_detection_results):
        def residuals(camera_position_2d, landmarks, distances):
            camera_position = np.array(
                [camera_position_2d[0], camera_position_2d[1], 0]
            )
            estimated_distances = np.linalg.norm(landmarks - camera_position, axis=1)
            return estimated_distances - distances

        initial_guess = self.previous_location

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

        result = least_squares(
            residuals,
            initial_guess,
            args=(landmarks, distances),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

        self.previous_location = result.x

        # now that we have location, estimate pose
        poses = self.estimate_pose(marker_detection_results, result.x)
        print(poses, "poses")

        return result.x

    def triangulate_exact(self, marker_detection_results):
        if len(marker_detection_results["marker_ids"]) < 2:
            return self.previous_location

        # Get the two closest markers based on measured distances
        distances = np.array(marker_detection_results["marker_distances"])
        sorted_indices = np.argsort(distances)
        closest_indices = sorted_indices[:2]

        marker_ids = marker_detection_results["marker_ids"]
        p1 = MARKER_ID_2_LOCATION[marker_ids[closest_indices[0]]]
        p2 = MARKER_ID_2_LOCATION[marker_ids[closest_indices[1]]]

        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        # Adjust distances using Pythagorean theorem to get ground plane distances
        r1 = np.sqrt(distances[closest_indices[0]] ** 2 - p1.z**2)
        r2 = np.sqrt(distances[closest_indices[1]] ** 2 - p2.z**2)

        # Calculate distance between circle centers
        d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Check if circles intersect
        if d > r1 + r2:  # Circles too far apart
            return self.previous_location
        if d < abs(r1 - r2):  # One circle contains the other
            return self.previous_location
        if d == 0 and r1 == r2:  # Circles are identical
            return self.previous_location

        # Calculate intersection points
        # Reference: https://mathworld.wolfram.com/Circle-CircleIntersection.html
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        try:
            h = np.sqrt(r1**2 - a**2)
        except ValueError:  # If r1**2 < a**2 due to numerical errors
            h = 0

        x3 = x1 + a * (x2 - x1) / d
        y3 = y1 + a * (y2 - y1) / d

        intersect_x1 = x3 + h * (y2 - y1) / d
        intersect_y1 = y3 - h * (x2 - x1) / d
        intersect_x2 = x3 - h * (y2 - y1) / d
        intersect_y2 = y3 + h * (x2 - x1) / d

        point1 = np.array([intersect_x1, intersect_y1])
        point2 = np.array([intersect_x2, intersect_y2])

        # If we have more than 2 markers, use them to determine which intersection point is correct
        if len(marker_detection_results["marker_ids"]) > 2:
            # Calculate errors for both potential points using all other markers
            error1 = 0
            error2 = 0

            for i in range(len(marker_ids)):
                if i not in closest_indices:
                    p = MARKER_ID_2_LOCATION[marker_ids[i]]
                    measured_dist = np.sqrt(distances[i] ** 2 - p.z**2)

                    # Calculate errors as difference between measured and computed distances
                    error1 += abs(
                        np.sqrt((p.x - point1[0]) ** 2 + (p.y - point1[1]) ** 2)
                        - measured_dist
                    )
                    error2 += abs(
                        np.sqrt((p.x - point2[0]) ** 2 + (p.y - point2[1]) ** 2)
                        - measured_dist
                    )

            # Choose the point with lower error
            result = point1 if error1 < error2 else point2
        else:
            # If we only have 2 markers, choose the point closer to previous location
            dist1 = np.linalg.norm(point1 - self.previous_location)
            dist2 = np.linalg.norm(point2 - self.previous_location)
            result = point1 if dist1 < dist2 else point2

        self.previous_location = result
        return result

    def find_intersection_point(self, r1, r2, d, x1, y1, x2, y2):
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = np.sqrt(r1**2 - a**2)
        x = (x2 - x1) * a / d + x1 + h / d * np.array([y2 - y1, y1 - y2])
        y = (y2 - y1) * a / d + y1 + h / d * np.array([x1 - x2, x2 - x1])
        pos1 = np.array([x[0], y[0]])
        pos2 = np.array([x[1], y[1]])
        return (
            pos1
            if np.linalg.norm(pos1 - self.previous_location)
            < np.linalg.norm(pos2 - self.previous_location)
            else pos2
        )
