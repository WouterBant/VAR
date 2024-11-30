import numpy as np
import cv2
from typing import List, Tuple
import matplotlib.pyplot as plt


class SfMPipeline:
    def __init__(self, images_paths: List[str], window_size: int = 30):
        """
        Initialize SfM pipeline with image folder and matching window

        Args:
            image_folder (str): Path to folder containing ordered images
            window_size (int): Number of images to consider for matching around each image
        """
        self.window_size = window_size
        self.images = images_paths
        self.sift = cv2.SIFT_create()

    def extract_features(self) -> List[Tuple]:
        """
        Extract SIFT features for all images

        Returns:
            List of (keypoints, descriptors) for each image
        """
        features = []
        for img_path in self.images:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = self.sift.detectAndCompute(img, None)
            features.append((keypoints, descriptors))
        return features

    def match_features(self, features: List[Tuple]) -> List[List[cv2.DMatch]]:
        """
        Match features between adjacent images within the sliding window

        Args:
            features (List[Tuple]): List of (keypoints, descriptors)

        Returns:
            List of matches between adjacent images
        """
        matches = []
        bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        for i in range(len(features)):
            # Define window boundaries
            start = max(0, i - self.window_size // 2)
            end = min(len(features), i + self.window_size // 2 + 1)

            image_matches = []
            for j in range(start, end):
                if i != j:
                    # Match descriptors between images
                    curr_matches = bf_matcher.match(features[i][1], features[j][1])
                    curr_matches = sorted(curr_matches, key=lambda x: x.distance)
                    image_matches.append(
                        {
                            "source_idx": i,
                            "target_idx": j,
                            "matches": curr_matches[:50],  # Top 50 matches
                        }
                    )

            matches.append(image_matches)

        return matches

    def estimate_fundamental_matrices(
        self, features: List[Tuple], matches: List[List[cv2.DMatch]]
    ):
        """
        Estimate fundamental matrices between image pairs

        Args:
            features (List[Tuple]): List of (keypoints, descriptors)
            matches (List[List[cv2.DMatch]]): Matched features

        Returns:
            List of fundamental matrices and inlier matches
        """
        fundamental_matrices = []

        for img_matches in matches:
            for match_set in img_matches:
                src_idx = match_set["source_idx"]
                tgt_idx = match_set["target_idx"]
                matches = match_set["matches"]

                # Extract matching keypoints
                src_kp = features[src_idx][0]
                tgt_kp = features[tgt_idx][0]

                src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches])
                tgt_pts = np.float32([tgt_kp[m.trainIdx].pt for m in matches])

                # Estimate fundamental matrix with RANSAC
                F, mask = cv2.findFundamentalMat(
                    src_pts, tgt_pts, cv2.FM_RANSAC, ransacReprojThreshold=3.0
                )

                inlier_matches = [matches[i] for i in range(len(matches)) if mask[i][0]]

                fundamental_matrices.append(
                    {
                        "source_idx": src_idx,
                        "target_idx": tgt_idx,
                        "fundamental_matrix": F,
                        "inlier_matches": inlier_matches,
                    }
                )

        return fundamental_matrices

    def triangulate_points(
        self, P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray
    ):
        """
        Triangulate 3D points from two camera projection matrices

        Args:
            P1, P2 (np.ndarray): Camera projection matrices
            pts1, pts2 (np.ndarray): Corresponding 2D points

        Returns:
            3D points
        """
        # Homogeneous coordinates
        pts1_homog = cv2.convertPointsToHomogeneous(pts1)
        pts2_homog = cv2.convertPointsToHomogeneous(pts2)

        # Triangulation
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

        return points_3d

    def run_sfm(self):
        """
        Main SfM pipeline execution
        """
        # Extract features
        features = self.extract_features()

        # Match features
        matches = self.match_features(features)

        # Estimate fundamental matrices
        fundamental_matrices = self.estimate_fundamental_matrices(features, matches)

        # TODO: Implement bundle adjustment and camera pose estimation
        print("SfM Pipeline execution complete.")

        return {
            "features": features,
            "matches": matches,
            "fundamental_matrices": fundamental_matrices,
        }

    def visualize_features(self, features):
        """
        Visualize SIFT keypoints for each image
        """
        plt.figure(figsize=(15, 3))

        for i, (keypoints, _) in enumerate(features):
            # Read image
            img = cv2.imread(self.images[i])

            # Draw keypoints
            img_with_keypoints = cv2.drawKeypoints(
                img,
                keypoints,
                None,
                color=(0, 255, 0),  # Green color
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            plt.subplot(1, len(features), i + 1)
            plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
            plt.title(f"Keypoints - Image {i}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_matches(self, features, matches):
        """
        Visualize feature matches between image pairs

        Args:
            features: List of (keypoints, descriptors)
            matches: List of match sets
        """
        plt.figure(figsize=(15, 5))

        for match_set in matches[
            0
        ]:  # Visualize matches for the first image's neighborhood
            src_idx = match_set["source_idx"]
            tgt_idx = match_set["target_idx"]

            # Read images
            img1 = cv2.imread(self.images[src_idx])
            img2 = cv2.imread(self.images[tgt_idx])

            # Get keypoints
            src_kp = features[src_idx][0]
            tgt_kp = features[tgt_idx][0]

            # Draw matches
            matching_img = cv2.drawMatches(
                img1,
                src_kp,
                img2,
                tgt_kp,
                match_set["matches"],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            plt.imshow(cv2.cvtColor(matching_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Matches between Image {src_idx} and {tgt_idx}")
            plt.axis("off")
            plt.show()

    def visualize_fundamental_matrices(self, features, fundamental_matrices):
        """
        Visualize epipolar lines for fundamental matrices

        Args:
            features: List of (keypoints, descriptors)
            fundamental_matrices: List of fundamental matrix data
        """
        plt.figure(figsize=(15, 5))

        for fm_data in fundamental_matrices:
            src_idx = fm_data["source_idx"]
            tgt_idx = fm_data["target_idx"]
            F = fm_data["fundamental_matrix"]
            inlier_matches = fm_data["inlier_matches"]

            # Read images
            img1 = cv2.imread(self.images[src_idx])
            img2 = cv2.imread(self.images[tgt_idx])

            # Get keypoints
            src_kp = features[src_idx][0]
            tgt_kp = features[tgt_idx][0]

            # Extract point coordinates from inlier matches
            src_pts = np.float32([src_kp[m.queryIdx].pt for m in inlier_matches])
            tgt_pts = np.float32([tgt_kp[m.trainIdx].pt for m in inlier_matches])

            # Compute epipolar lines
            lines1 = cv2.computeCorrespondEpilines(tgt_pts.reshape(-1, 1, 2), 2, F)
            lines1 = lines1.reshape(-1, 3)
            lines2 = cv2.computeCorrespondEpilines(src_pts.reshape(-1, 1, 2), 1, F)
            lines2 = lines2.reshape(-1, 3)

            # Draw epipolar lines
            def draw_lines(img, lines, points):
                r, c = img.shape[:2]
                for r, pt in zip(lines, points):
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    x0, y0 = map(int, [0, -r[2] / r[1]])
                    x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
                    img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
                    img = cv2.circle(img, tuple(map(int, pt)), 5, color, -1)
                return img

            img1_lines = draw_lines(img1, lines1, src_pts)
            img2_lines = draw_lines(img2, lines2, tgt_pts)

            # Plot
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img1_lines, cv2.COLOR_BGR2RGB))
            plt.title(f"Epipolar Lines - Image {src_idx}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(img2_lines, cv2.COLOR_BGR2RGB))
            plt.title(f"Epipolar Lines - Image {tgt_idx}")
            plt.axis("off")

            plt.tight_layout()
            plt.show()
