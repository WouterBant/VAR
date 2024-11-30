import cv2
import numpy as np
from scipy.optimize import least_squares


# class StructureFromMotion:
#     def __init__(self, debug=False):
#         """
#         Initialize the SfM pipeline.
#         :param debug: If True, intermediate steps will be visualized.
#         """
#         self.debug = debug

#     def run(self, images):
#         print("[INFO] Running SfM pipeline...")

#         keypoints, descriptors = self.detect_features(images)
#         matches = self.match_features(images, keypoints, descriptors)
#         points_3d, camera_poses = self.reconstruct_3d(images, keypoints, matches)

#         print("[INFO] SfM pipeline completed.")
#         return points_3d, camera_poses

#     def detect_features(self, images):
#         print("[INFO] Detecting features...")
#         keypoints, descriptors = [], []
#         sift = cv2.SIFT_create()

#         for img_path in images:
#             image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             kp, des = sift.detectAndCompute(image, None)
#             keypoints.append(kp)
#             descriptors.append(des)

#             if self.debug:
#                 img_with_kp = cv2.drawKeypoints(image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#                 plt.imshow(img_with_kp, cmap='gray')
#                 plt.title(f"Detected Features in {img_path}")
#                 plt.show()

#         return keypoints, descriptors

#     def match_features(self, images, keypoints, descriptors):
#         print("[INFO] Matching features...")
#         bf = cv2.BFMatcher()
#         matches = []

#         for i in range(len(descriptors) - 1):
#             raw_matches = bf.knnMatch(descriptors[i], descriptors[i + 1], k=2)

#             # Apply Lowe's ratio test
#             good_matches = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]
#             matches.append(good_matches)

#             if self.debug:
#                 print(f"[DEBUG] Matches between image {i} and {i + 1}: {len(good_matches)}")
#                 # Visualize matches
#                 img1 = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
#                 img2 = cv2.imread(images[i + 1], cv2.IMREAD_GRAYSCALE)
#                 img_matches = cv2.drawMatches(img1, keypoints[i], img2, keypoints[i + 1], good_matches, None)
#                 plt.imshow(img_matches)
#                 plt.title(f"Matches between Image {i} and {i + 1}")
#                 plt.show()

#         return matches

#     def reconstruct_3d(self, images, keypoints, matches):
#         print("[INFO] Reconstructing 3D structure...")
#         points_3d = []
#         camera_poses = []  # Global camera poses relative to the first frame

#         # First camera is at the origin
#         camera_poses.append((np.eye(3), np.zeros((3, 1))))

#         # Compute camera poses relative to the first frame
#         for i in range(len(matches)):
#             # Extract matched points
#             pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i]])
#             pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches[i]])

#             # Estimate essential matrix
#             E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=1.0)

#             # Decompose essential matrix to get camera pose
#             _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2)

#             # Compute global camera pose relative to the first frame
#             prev_R, prev_t = camera_poses[-1]
#             global_R = prev_R @ R
#             global_t = prev_R @ t + prev_t

#             camera_poses.append((global_R, global_t))

#             # Triangulate points
#             P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for the first camera
#             P2 = np.hstack((global_R, global_t))  # Projection matrix for the current camera
#             points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
#             points_3d.append(points_4d_hom[:3] / points_4d_hom[3])  # Convert to 3D points

#             if self.debug:
#                 print(f"[DEBUG] Global Camera Pose {i + 1}: Rotation={global_R}, Translation={global_t}")
#                 print(f"[DEBUG] Triangulated Points (Sample): {points_3d[-1][:5]}")

#         return points_3d, camera_poses

#     def visualize_sparse_point_cloud(self, points_3d, camera_poses):
#         """
#         Create an interactive 3D visualization of the sparse point cloud and camera locations.
#         :param points_3d: List of 3D points (one array per image pair).
#         :param camera_poses: List of global camera poses (R, t) relative to the first frame.
#         """
#         print("[INFO] Creating interactive sparse point cloud visualization...")

#         # Aggregate all points into a single point cloud
#         all_points = np.hstack(points_3d)

#         # Create figure
#         fig = go.Figure()

#         # Add point cloud scatter plot
#         fig.add_trace(go.Scatter3d(
#             x=all_points[0],
#             y=all_points[1],
#             z=all_points[2],
#             mode='markers',
#             marker=dict(
#                 size=2,
#                 color=all_points[2],  # Color by depth
#                 colorscale='Viridis',
#                 opacity=0.5
#             ),
#             name='Sparse Point Cloud'
#         ))

#         # Add camera locations and orientations
#         camera_x, camera_y, camera_z = [], [], []
#         camera_u, camera_v, camera_w = [], [], []
#         camera_colors = []

#         for i, (R, t) in enumerate(camera_poses):
#             camera_x.append(t[0][0])
#             camera_y.append(t[1][0])
#             camera_z.append(t[2][0])

#             # Camera orientation vector (Z-axis)
#             direction_vector = R[:, 2]
#             camera_u.append(direction_vector[0])
#             camera_v.append(direction_vector[1])
#             camera_w.append(direction_vector[2])

#             # Color cameras differently
#             camera_colors.append('red' if i == 0 else 'green')

#         # Add camera locations
#         fig.add_trace(go.Scatter3d(
#             x=camera_x,
#             y=camera_y,
#             z=camera_z,
#             mode='markers',
#             marker=dict(
#                 size=5,
#                 color=camera_colors,
#                 symbol='diamond'
#             ),
#             name='Camera Locations'
#         ))

#         # Add camera orientation vectors
#         for i in range(len(camera_x)):
#             fig.add_trace(go.Scatter3d(
#                 x=[camera_x[i], camera_x[i] + camera_u[i] * 0.2],
#                 y=[camera_y[i], camera_y[i] + camera_v[i] * 0.2],
#                 z=[camera_z[i], camera_z[i] + camera_w[i] * 0.2],
#                 mode='lines',
#                 line=dict(color=camera_colors[i], width=3),
#                 name=f'Camera {i+1} Orientation'
#             ))

#         # Customize layout
#         fig.update_layout(
#             title='Interactive Sparse Point Cloud and Camera Poses',
#             scene=dict(
#                 xaxis_title='X',
#                 yaxis_title='Y',
#                 zaxis_title='Z',
#                 aspectmode='data'  # Preserve the actual aspect ratio
#             ),
#             height=800,
#             width=1000,
#             margin=dict(r=20, b=10, l=10, t=40)
#         )

#         return fig


#     # def visualize_sparse_point_cloud(self, points_3d, camera_poses):
#     #     """
#     #     Visualize the sparse point cloud and camera locations.
#     #     :param points_3d: List of 3D points (one array per image pair).
#     #     :param camera_poses: List of global camera poses (R, t) relative to the first frame.
#     #     """
#     #     print("[INFO] Visualizing sparse point cloud...")

#     #     fig = plt.figure(figsize=(10, 7))
#     #     ax = fig.add_subplot(111, projection='3d')

#     #     # Aggregate all points into a single point cloud
#     #     all_points = np.hstack(points_3d)

#     #     # Plot 3D points
#     #     ax.scatter(all_points[0], all_points[1], all_points[2], c='blue', s=1, alpha=0.5)

#     #     # Plot camera locations
#     #     for i, (R, t) in enumerate(camera_poses):
#     #         ax.scatter(t[0], t[1], t[2],
#     #                    c='red' if i == 0 else 'green',
#     #                    marker='^' if i == 0 else 'o',
#     #                    s=50,
#     #                    label=f'Camera {i + 1}')

#     #         # Plot camera orientation
#     #         direction_vector = R[:, 2]  # Z-axis of the camera
#     #         ax.quiver(t[0], t[1], t[2],
#     #                   direction_vector[0], direction_vector[1], direction_vector[2],
#     #                   color='red' if i == 0 else 'green',
#     #                   length=0.2,
#     #                   normalize=True)

#     #     # Set plot labels and legend
#     #     ax.set_xlabel('X')
#     #     ax.set_ylabel('Y')
#     #     ax.set_zlabel('Z')
#     #     ax.set_title('Sparse Point Cloud with Camera Positions')
#     #     ax.legend()
#     #     plt.show()

#     # def reconstruct_3d(self, images, keypoints, matches):
#     #     print("[INFO] Reconstructing 3D structure...")
#     #     points_3d = []
#     #     camera_poses = []  # Placeholder for camera poses

#     #     # Simplified: Estimate essential matrix and triangulate points
#     #     for i in range(len(matches)):
#     #         img1 = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
#     #         img2 = cv2.imread(images[i + 1], cv2.IMREAD_GRAYSCALE)

#     #         # Extract matched points
#     #         pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i]])
#     #         pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches[i]])

#     #         # Estimate essential matrix
#     #         E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=1.0)

#     #         # Decompose essential matrix to get camera pose
#     #         _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2)

#     #         # Triangulate points
#     #         P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for the first camera
#     #         P2 = np.hstack((R, t))  # Projection matrix for the second camera
#     #         points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
#     #         points_3d.append(points_4d_hom[:3] / points_4d_hom[3])  # Convert to 3D points

#     #         camera_poses.append((R, t))

#     #         if self.debug:
#     #             print(f"[DEBUG] Camera Pose {i + 1}: Rotation={R}, Translation={t}")
#     #             print(f"[DEBUG] Triangulated Points (Sample): {points_3d[-1][:5]}")

#     #     return points_3d, camera_poses

#     # def visualize_sparse_point_cloud(self, points_3d, camera_poses):
#     #     """
#     #     Visualize the sparse point cloud and camera locations.
#     #     :param points_3d: List of 3D points (one array per image pair).
#     #     :param camera_poses: List of camera poses (R, t) for each pair of images.
#     #     """
#     #     print("[INFO] Visualizing sparse point cloud...")

#     #     fig = plt.figure(figsize=(10, 7))
#     #     ax = fig.add_subplot(111, projection='3d')

#     #     # Plot 3D points
#     #     for points in points_3d:
#     #         ax.scatter(points[0], points[1], points[2], c='blue', s=1)

#     #     # Plot camera locations
#     #     origin = np.array([0, 0, 0]).reshape(3, 1)  # The origin of the first camera
#     #     ax.scatter(origin[0], origin[1], origin[2], c='red', marker='o', s=50, label='Camera 1 (Origin)')

#     #     # Compute camera positions from poses
#     #     camera_positions = [origin]
#     #     for R, t in camera_poses:
#     #         camera_position = -R.T @ t  # Camera position in world coordinates
#     #         camera_positions.append(camera_position)

#     #         ax.scatter(camera_position[0], camera_position[1], camera_position[2],
#     #                    c='green', marker='^', s=50)

#     #     # Plot camera orientations
#     #     for i, (R, t) in enumerate(camera_poses):
#     #         camera_position = camera_positions[i + 1]
#     #         direction_vector = R[:, 2]  # Z-axis of the camera in world coordinates
#     #         ax.quiver(camera_position[0], camera_position[1], camera_position[2],
#     #                   direction_vector[0], direction_vector[1], direction_vector[2],
#     #                   color='red', length=0.2, normalize=True)

#     #     # Set plot labels and legend
#     #     ax.set_xlabel('X')
#     #     ax.set_ylabel('Y')
#     #     ax.set_zlabel('Z')
#     #     ax.set_title('Sparse Point Cloud with Camera Positions')
#     #     ax.legend()
#     #     plt.show()


class StructureFromMotion:
    def __init__(self, debug=False):
        """
        Initialize the SfM pipeline with advanced refinement.
        :param debug: If True, intermediate steps will be visualized.
        """
        self.debug = debug
        # Intrinsic camera matrix (assuming normalized coordinates)
        self.K = np.array(
            [
                [1, 0, 0],  # Focal length and principal point
                [0, 1, 0],  # These are placeholder values
                [0, 0, 1],
            ]
        )

    def detect_features(self, images):
        print("[INFO] Detecting features...")
        keypoints, descriptors = [], []
        sift = cv2.SIFT_create()

        for img_path in images:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            kp, des = sift.detectAndCompute(image, None)

            # Normalize keypoints to image coordinates
            normalized_kp = []
            for k in kp:
                x, y = k.pt
                # Convert to normalized coordinates
                x_norm = (x - self.K[0, 2]) / self.K[0, 0]
                y_norm = (y - self.K[1, 2]) / self.K[1, 1]
                k.pt = (x_norm, y_norm)
                normalized_kp.append(k)

            keypoints.append(normalized_kp)
            descriptors.append(des)

        return keypoints, descriptors

    def match_features(self, images, keypoints, descriptors):
        print("[INFO] Matching features...")
        bf = cv2.BFMatcher()
        matches_list = []

        for i in range(len(descriptors) - 1):
            raw_matches = bf.knnMatch(descriptors[i], descriptors[i + 1], k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for m, n in raw_matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            matches_list.append(good_matches)

        return matches_list

    def bundle_adjustment_cost(
        self, params, n_cameras, n_points, camera_indices, point_indices, points_2d
    ):
        """
        Cost function for bundle adjustment.
        :param params: Flattened array of camera parameters and 3D points
        :param n_cameras: Number of cameras
        :param n_points: Number of 3D points
        :param camera_indices: Indices of cameras for each observation
        :param point_indices: Indices of 3D points for each observation
        :param points_2d: 2D point observations
        :return: Reprojection errors
        """
        # Reshape parameters
        camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6 :].reshape((n_points, 3))

        # Compute reprojection errors
        reprojection_errors = []
        for i in range(len(points_2d)):
            cam_idx = camera_indices[i]
            point_idx = point_indices[i]

            # Extract camera parameters
            rvec = camera_params[cam_idx][:3]
            tvec = camera_params[cam_idx][3:]

            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Project 3D point
            point_3d = points_3d[point_idx]
            point_proj = self.K @ (R @ point_3d + tvec)
            point_proj /= point_proj[2]  # Normalize

            # Compute reprojection error
            error = points_2d[i] - point_proj[:2]
            reprojection_errors.append(error)

        return np.array(reprojection_errors).ravel()

    def reconstruct_3d(self, images, keypoints, matches):
        print("[INFO] Reconstructing 3D structure...")
        points_3d = []
        camera_poses = []  # Global camera poses relative to the first frame

        # First camera is at the origin
        camera_poses.append((np.eye(3), np.zeros((3, 1))))

        # Compute camera poses relative to the first frame
        for i in range(len(matches)):
            # Extract matched points
            pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i]])
            pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches[i]])

            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(
                pts1,
                pts2,
                focal=1.0,
                pp=(0, 0),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )

            # Decompose essential matrix to get camera pose
            _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2)

            # Compute global camera pose relative to the first frame
            prev_R, prev_t = camera_poses[-1]
            global_R = prev_R @ R
            global_t = prev_R @ t + prev_t

            camera_poses.append((global_R, global_t))

            # Triangulate points
            P1 = np.hstack(
                (np.eye(3), np.zeros((3, 1)))
            )  # Projection matrix for the first camera
            P2 = np.hstack(
                (global_R, global_t)
            )  # Projection matrix for the current camera
            points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            points_3d.append(
                points_4d_hom[:3] / points_4d_hom[3]
            )  # Convert to 3D points

            if self.debug:
                print(
                    f"[DEBUG] Global Camera Pose {i + 1}: Rotation={global_R}, Translation={global_t}"
                )
                print(f"[DEBUG] Triangulated Points (Sample): {points_3d[-1][:5]}")

        return points_3d, camera_poses

    def subsample_points(self, points_3d, max_points=10000):
        """
        Subsample 3D points to reduce memory usage.
        :param points_3d: Original 3D points
        :param max_points: Maximum number of points to use
        :return: Subsampled points
        """
        # Flatten and consolidate 3D points
        if isinstance(points_3d, list):
            if isinstance(points_3d[0], np.ndarray):
                # If points_3d is a list of arrays, concatenate
                points_flat = np.concatenate(points_3d, axis=1).T
            else:
                # If points_3d is a list of lists, convert to numpy array
                points_flat = np.array([np.array(pts).flatten() for pts in points_3d])
        else:
            # If already a numpy array, ensure correct shape
            points_flat = np.atleast_2d(points_3d)

        # Reshape to ensure 2D array with shape (n_points, 3)
        points_flat = points_flat.reshape(-1, 3)

        # If too many points, randomly subsample
        if len(points_flat) > max_points:
            # Use numpy's random choice for efficient subsampling
            indices = np.random.choice(len(points_flat), size=max_points, replace=False)
            points_flat = points_flat[indices]

        return points_flat

    def bundle_adjustment_cost_chunked(
        self,
        camera_params,
        points_3d,
        camera_indices,
        point_indices,
        points_2d,
        chunk_size=1000,
    ):
        """
        Compute reprojection errors in chunks to manage memory.
        :param camera_params: Camera parameters
        :param points_3d: 3D points
        :param camera_indices: Camera indices for each observation
        :param point_indices: Point indices for each observation
        :param points_2d: 2D point observations
        :param chunk_size: Number of points to process in each chunk
        :return: Reprojection errors
        """
        reprojection_errors = []

        # Process points in chunks
        for start in range(0, len(points_3d), chunk_size):
            end = min(start + chunk_size, len(points_3d))
            chunk_points = points_3d[start:end]

            # Find observations for this chunk of points
            chunk_mask = (point_indices >= start) & (point_indices < end)
            chunk_obs_indices = np.where(chunk_mask)[0]

            if len(chunk_obs_indices) == 0:
                continue

            # Adjust point and camera indices for this chunk
            local_point_indices = point_indices[chunk_mask] - start
            local_camera_indices = camera_indices[chunk_mask]
            local_points_2d = points_2d[chunk_mask]

            # Compute chunk reprojection errors
            for i, (point_idx, cam_idx, obs_2d) in enumerate(
                zip(local_point_indices, local_camera_indices, local_points_2d)
            ):
                # Extract camera parameters
                rvec = camera_params[cam_idx][:3]
                tvec = camera_params[cam_idx][3:]

                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                # Project 3D point
                point_3d = chunk_points[point_idx]
                point_proj = self.K @ (R @ point_3d + tvec)
                point_proj /= point_proj[2]  # Normalize

                # Compute reprojection error
                error = obs_2d - point_proj[:2]
                reprojection_errors.append(error)

        return np.array(reprojection_errors).ravel()

    def refine_reconstruction(
        self, points_3d, camera_poses, keypoints, matches, max_points=10000
    ):
        """
        Refine 3D points and camera poses using memory-efficient bundle adjustment.
        :param points_3d: Initial 3D point estimates
        :param camera_poses: Initial camera poses
        :param keypoints: Feature keypoints
        :param matches: Feature matches
        :param max_points: Maximum number of points to use
        :return: Refined 3D points and camera poses
        """
        print("[INFO] Performing memory-efficient bundle adjustment...")

        # Subsample points to manage memory
        points_3d_flat = self.subsample_points(points_3d, max_points)

        # Prepare camera parameters
        camera_params = []
        for R, t in camera_poses:
            # Convert rotation matrix to rotation vector
            rvec, _ = cv2.Rodrigues(R)
            camera_params.append(np.concatenate([rvec.ravel(), t.ravel()]))

        # Prepare data for optimization
        camera_indices = []
        point_indices = []
        points_2d = []

        # Collect all observations
        for i, match_set in enumerate(matches):
            for match in match_set:
                camera_indices.append(i)
                camera_indices.append(i + 1)

                point_indices.append(len(point_indices) // 2)
                point_indices.append(len(point_indices) // 2)

                # Get 2D points for this match
                pt1 = keypoints[i][match.queryIdx].pt
                pt2 = keypoints[i + 1][match.trainIdx].pt
                points_2d.extend([pt1, pt2])

        # Perform bundle adjustment with custom cost function
        result = least_squares(
            lambda x: self.bundle_adjustment_cost_chunked(
                x[: len(camera_params) * 6].reshape((len(camera_params), 6)),
                points_3d_flat,
                np.array(camera_indices),
                np.array(point_indices),
                np.array(points_2d),
            ),
            np.concatenate([np.array(camera_params).ravel(), points_3d_flat.ravel()]),
            method="trf",
            ftol=1e-4,
            xtol=1e-4,
        )

        # Extract refined parameters
        refined_camera_params = result.x[: len(camera_params) * 6].reshape(
            (len(camera_params), 6)
        )
        refined_points_3d = result.x[len(camera_params) * 6 :].reshape((-1, 3))

        # Convert camera parameters back to rotation matrices and translations
        refined_camera_poses = []
        for params in refined_camera_params:
            R, _ = cv2.Rodrigues(params[:3])
            t = params[3:].reshape((3, 1))
            refined_camera_poses.append((R, t))

        return refined_points_3d, refined_camera_poses

    def run(self, images):
        print("[INFO] Running SfM pipeline...")

        keypoints, descriptors = self.detect_features(images)
        matches = self.match_features(images, keypoints, descriptors)

        # Initial reconstruction
        initial_points_3d, initial_camera_poses = self.reconstruct_3d(
            images, keypoints, matches
        )
        return initial_points_3d, initial_camera_poses
        # Refine reconstruction
        # refined_points_3d, refined_camera_poses = self.refine_reconstruction(
        #     initial_points_3d,
        #     initial_camera_poses,
        #     keypoints,
        #     matches
        # )

        # print("[INFO] SfM pipeline completed.")
        # return refined_points_3d, refined_camera_poses
