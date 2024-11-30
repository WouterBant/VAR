import cv2
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go


class StructureFromMotion:
    def __init__(self, debug=False):
        self.debug = debug

    def run(self, images):
        print("[INFO] Running SfM pipeline...")

        keypoints, descriptors = self.detect_features(images)
        matches = self.match_features(images, keypoints, descriptors)
        points_3d, camera_poses = self.reconstruct_3d(images, keypoints, matches)

        print("[INFO] SfM pipeline completed.")
        return points_3d, camera_poses

    def detect_features(self, images):
        print("[INFO] Detecting features...")
        keypoints, descriptors = [], []
        sift = cv2.SIFT_create()

        for img_path in images:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            kp, des = sift.detectAndCompute(image, None)
            keypoints.append(kp)
            descriptors.append(des)

            if self.debug:
                img_with_kp = cv2.drawKeypoints(
                    image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
                plt.imshow(img_with_kp, cmap="gray")
                plt.title(f"Detected Features in {img_path}")
                plt.show()

        return keypoints, descriptors

    def match_features(self, images, keypoints, descriptors):
        print("[INFO] Matching features...")
        bf = cv2.BFMatcher()
        matches = []

        for i in range(len(descriptors) - 1):
            raw_matches = bf.knnMatch(descriptors[i], descriptors[i + 1], k=2)

            # Apply Lowe's ratio test
            good_matches = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]
            matches.append(good_matches)

            if self.debug:
                print(
                    f"[DEBUG] Matches between image {i} and {i + 1}: {len(good_matches)}"
                )
                # Visualize matches
                img1 = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(images[i + 1], cv2.IMREAD_GRAYSCALE)
                img_matches = cv2.drawMatches(
                    img1, keypoints[i], img2, keypoints[i + 1], good_matches, None
                )
                plt.imshow(img_matches)
                plt.title(f"Matches between Image {i} and {i + 1}")
                plt.show()

        return matches

    def reconstruct_3d(self, images, keypoints, matches):
        print("[INFO] Reconstructing 3D structure...")
        points_3d = []
        camera_poses = []  # Global camera poses relative to the first frame

        # First camera is at the origin
        camera_poses.append(
            {
                "R": np.eye(3),  # Rotation matrix
                "t": np.zeros((3, 1)),  # Translation vector
                "P": np.hstack((np.eye(3), np.zeros((3, 1)))),  # Projection matrix
            }
        )

        for i in range(len(matches)):
            # Extract matched points
            pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i]])
            pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches[i]])

            # Compute fundamental matrix with RANSAC to filter outliers
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)

            # Filter matches using the mask
            inliers1 = pts1[mask.ravel() == 1]
            inliers2 = pts2[mask.ravel() == 1]

            # More robust essential matrix estimation
            K = np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            )  # Intrinsic camera matrix (adjust as needed)
            E, _ = cv2.findEssentialMat(
                inliers1, inliers2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
            )

            # Recover pose with proper intrinsic matrix
            _, R, t, mask_pose = cv2.recoverPose(E, inliers1, inliers2, K)

            # Compute global camera pose
            prev_pose = camera_poses[-1]
            global_R = prev_pose["R"] @ R
            global_t = prev_pose["R"] @ t + prev_pose["t"]

            # Create projection matrix for this camera
            global_P = np.hstack((global_R, global_t))
            global_P = K @ global_P  # Combine with intrinsic matrix

            camera_poses.append({"R": global_R, "t": global_t, "P": global_P})

            # More robust triangulation
            points_4d_hom = cv2.triangulatePoints(
                camera_poses[i]["P"], global_P, inliers1.T, inliers2.T
            )

            # Convert to 3D points and store
            current_points_3d = points_4d_hom[:3] / points_4d_hom[3]
            points_3d.append(current_points_3d)

            if self.debug:
                print(f"[DEBUG] Global Camera Pose {i + 1}:")
                print(f"Rotation:\n{global_R}")
                print(f"Translation:\n{global_t}")

        return points_3d, camera_poses

    # def reconstruct_3d(self, images, keypoints, matches):
    #     print("[INFO] Reconstructing 3D structure...")
    #     points_3d = []
    #     camera_poses = []  # Global camera poses relative to the first frame

    #     # First camera is at the origin
    #     camera_poses.append((np.eye(3), np.zeros((3, 1))))

    #     # Compute camera poses relative to the first frame
    #     for i in range(len(matches)):
    #         # Extract matched points
    #         pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i]])
    #         pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches[i]])

    #         # Estimate essential matrix
    #         E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=1.0)

    #         # Decompose essential matrix to get camera pose
    #         _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2)

    #         # Compute global camera pose relative to the first frame
    #         prev_R, prev_t = camera_poses[-1]
    #         global_R = prev_R @ R
    #         global_t = prev_R @ t + prev_t

    #         camera_poses.append((global_R, global_t))

    #         # Triangulate points
    #         P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for the first camera
    #         P2 = np.hstack((global_R, global_t))  # Projection matrix for the current camera
    #         points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    #         points_3d.append(points_4d_hom[:3] / points_4d_hom[3])  # Convert to 3D points

    #         if self.debug:
    #             print(f"[DEBUG] Global Camera Pose {i + 1}: Rotation={global_R}, Translation={global_t}")
    #             print(f"[DEBUG] Triangulated Points (Sample): {points_3d[-1][:5]}")

    #     return points_3d, camera_poses

    def visualize_sparse_point_cloud(self, points_3d, camera_poses):
        """
        Create an interactive 3D visualization of the sparse point cloud and camera locations.
        :param points_3d: List of 3D points (one array per image pair).
        :param camera_poses: List of global camera poses (R, t) relative to the first frame.
        """
        print("[INFO] Creating interactive sparse point cloud visualization...")

        # Aggregate all points into a single point cloud
        all_points = np.hstack(points_3d)

        # Create figure
        fig = go.Figure()

        # Add point cloud scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=all_points[0],
                y=all_points[1],
                z=all_points[2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=all_points[2],  # Color by depth
                    colorscale="Viridis",
                    opacity=0.5,
                ),
                name="Sparse Point Cloud",
            )
        )

        # Add camera locations and orientations
        camera_x, camera_y, camera_z = [], [], []
        camera_u, camera_v, camera_w = [], [], []
        camera_colors = []

        for i, (R, t) in enumerate(camera_poses):
            camera_x.append(t[0][0])
            camera_y.append(t[1][0])
            camera_z.append(t[2][0])

            # Camera orientation vector (Z-axis)
            direction_vector = R[:, 2]
            camera_u.append(direction_vector[0])
            camera_v.append(direction_vector[1])
            camera_w.append(direction_vector[2])

            # Color cameras differently
            camera_colors.append("red" if i == 0 else "green")

        # Add camera locations
        fig.add_trace(
            go.Scatter3d(
                x=camera_x,
                y=camera_y,
                z=camera_z,
                mode="markers",
                marker=dict(size=5, color=camera_colors, symbol="diamond"),
                name="Camera Locations",
            )
        )

        # Add camera orientation vectors
        for i in range(len(camera_x)):
            fig.add_trace(
                go.Scatter3d(
                    x=[camera_x[i], camera_x[i] + camera_u[i] * 0.2],
                    y=[camera_y[i], camera_y[i] + camera_v[i] * 0.2],
                    z=[camera_z[i], camera_z[i] + camera_w[i] * 0.2],
                    mode="lines",
                    line=dict(color=camera_colors[i], width=3),
                    name=f"Camera {i+1} Orientation",
                )
            )

        # Customize layout
        fig.update_layout(
            title="Interactive Sparse Point Cloud and Camera Poses",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",  # Preserve the actual aspect ratio
            ),
            height=800,
            width=1000,
            margin=dict(r=20, b=10, l=10, t=40),
        )

        return fig

    # def visualize_sparse_point_cloud(self, points_3d, camera_poses):
    #     """
    #     Visualize the sparse point cloud and camera locations.
    #     :param points_3d: List of 3D points (one array per image pair).
    #     :param camera_poses: List of global camera poses (R, t) relative to the first frame.
    #     """
    #     print("[INFO] Visualizing sparse point cloud...")

    #     fig = plt.figure(figsize=(10, 7))
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Aggregate all points into a single point cloud
    #     all_points = np.hstack(points_3d)

    #     # Plot 3D points
    #     ax.scatter(all_points[0], all_points[1], all_points[2], c='blue', s=1, alpha=0.5)

    #     # Plot camera locations
    #     for i, (R, t) in enumerate(camera_poses):
    #         ax.scatter(t[0], t[1], t[2],
    #                    c='red' if i == 0 else 'green',
    #                    marker='^' if i == 0 else 'o',
    #                    s=50,
    #                    label=f'Camera {i + 1}')

    #         # Plot camera orientation
    #         direction_vector = R[:, 2]  # Z-axis of the camera
    #         ax.quiver(t[0], t[1], t[2],
    #                   direction_vector[0], direction_vector[1], direction_vector[2],
    #                   color='red' if i == 0 else 'green',
    #                   length=0.2,
    #                   normalize=True)

    #     # Set plot labels and legend
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title('Sparse Point Cloud with Camera Positions')
    #     ax.legend()
    #     plt.show()

    # def reconstruct_3d(self, images, keypoints, matches):
    #     print("[INFO] Reconstructing 3D structure...")
    #     points_3d = []
    #     camera_poses = []  # Placeholder for camera poses

    #     # Simplified: Estimate essential matrix and triangulate points
    #     for i in range(len(matches)):
    #         img1 = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
    #         img2 = cv2.imread(images[i + 1], cv2.IMREAD_GRAYSCALE)

    #         # Extract matched points
    #         pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i]])
    #         pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches[i]])

    #         # Estimate essential matrix
    #         E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=1.0)

    #         # Decompose essential matrix to get camera pose
    #         _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2)

    #         # Triangulate points
    #         P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for the first camera
    #         P2 = np.hstack((R, t))  # Projection matrix for the second camera
    #         points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    #         points_3d.append(points_4d_hom[:3] / points_4d_hom[3])  # Convert to 3D points

    #         camera_poses.append((R, t))

    #         if self.debug:
    #             print(f"[DEBUG] Camera Pose {i + 1}: Rotation={R}, Translation={t}")
    #             print(f"[DEBUG] Triangulated Points (Sample): {points_3d[-1][:5]}")

    #     return points_3d, camera_poses

    # def visualize_sparse_point_cloud(self, points_3d, camera_poses):
    #     """
    #     Visualize the sparse point cloud and camera locations.
    #     :param points_3d: List of 3D points (one array per image pair).
    #     :param camera_poses: List of camera poses (R, t) for each pair of images.
    #     """
    #     print("[INFO] Visualizing sparse point cloud...")

    #     fig = plt.figure(figsize=(10, 7))
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Plot 3D points
    #     for points in points_3d:
    #         ax.scatter(points[0], points[1], points[2], c='blue', s=1)

    #     # Plot camera locations
    #     origin = np.array([0, 0, 0]).reshape(3, 1)  # The origin of the first camera
    #     ax.scatter(origin[0], origin[1], origin[2], c='red', marker='o', s=50, label='Camera 1 (Origin)')

    #     # Compute camera positions from poses
    #     camera_positions = [origin]
    #     for R, t in camera_poses:
    #         camera_position = -R.T @ t  # Camera position in world coordinates
    #         camera_positions.append(camera_position)

    #         ax.scatter(camera_position[0], camera_position[1], camera_position[2],
    #                    c='green', marker='^', s=50)

    #     # Plot camera orientations
    #     for i, (R, t) in enumerate(camera_poses):
    #         camera_position = camera_positions[i + 1]
    #         direction_vector = R[:, 2]  # Z-axis of the camera in world coordinates
    #         ax.quiver(camera_position[0], camera_position[1], camera_position[2],
    #                   direction_vector[0], direction_vector[1], direction_vector[2],
    #                   color='red', length=0.2, normalize=True)

    #     # Set plot labels and legend
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title('Sparse Point Cloud with Camera Positions')
    #     ax.legend()
    #     plt.show()
