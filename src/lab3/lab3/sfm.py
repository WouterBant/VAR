import cv2
import numpy as np


class SfM:
    def __init__(self, image_paths):  # TODO add config file
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher()
        self.image_paths = image_paths
        
    def detect_features(self):
        self.keypoints = []
        self.descriptors = []
        
        for image_path in self.image_paths:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            kp, des = self.detector.detectAndCompute(img, None)
            self.keypoints.append(kp)
            self.descriptors.append(des)
        
        return self.keypoints, self.descriptors
    
    def match_features(self):
        self.matches = []
        for i in range(len(self.descriptors) - 1):
            # Match features between consecutive frames
            matches = self.matcher.knnMatch(
                self.descriptors[i], 
                self.descriptors[i+1], 
                k=2
            )
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            self.matches.append(good_matches)
        
        return self.matches
    
    def estimate_camera_motion(self):
        camera_matrices = []
        
        for i in range(len(self.matches)):
            # Get matched keypoints
            img1_kp = np.float32([
                self.keypoints[i][m.queryIdx].pt 
                for m in self.matches[i]
            ])
            img2_kp = np.float32([
                self.keypoints[i+1][m.trainIdx].pt 
                for m in self.matches[i]
            ])
            
            # Estimate fundamental matrix
            F, mask = cv2.findFundamentalMat(
                img1_kp, img2_kp, 
                cv2.FM_RANSAC
            )
            
            # Recover pose
            _, R, t, _ = cv2.recoverPose(
                F, img1_kp, img2_kp
            )
            
            camera_matrices.append((R, t))
        
        return camera_matrices
    
    def triangulate_points(self, camera_matrices):
        # Initial camera matrices
        P0 = np.eye(3, 4)  # First camera at origin
        point_clouds = []
        
        for i, (R, t) in enumerate(camera_matrices):
            # Construct camera matrix
            P1 = np.hstack((R, t))
            
            # Get matched points
            img1_kp = np.float32([
                self.keypoints[i][m.queryIdx].pt 
                for m in self.matches[i]
            ])
            img2_kp = np.float32([
                self.keypoints[i+1][m.trainIdx].pt 
                for m in self.matches[i]
            ])
            
            # Triangulate points
            points_4d = cv2.triangulatePoints(
                P0, P1, 
                img1_kp.T, img2_kp.T
            )
            
            # Convert to 3D points
            points_3d = points_4d[:3, :] / points_4d[3, :]
            point_clouds.append(points_3d.T)
        
        return point_clouds
    
    def filter_point_cloud(self, point_clouds):
        # Combine all point clouds
        all_points = np.vstack(point_clouds)
        
        # Compute center and standard deviation
        center = np.mean(all_points, axis=0)
        std = np.std(all_points, axis=0)
        
        # Filter points within 2 standard deviations
        mask = np.all(np.abs(all_points - center) < 2 * std, axis=1)
        filtered_points = all_points[mask]
        
        return filtered_points
    
    def visualize_point_cloud(self, points, output_path='maze_point_cloud.ply'):
        with open(output_path, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            # Write points
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        print(f"Point cloud saved to {output_path}")
    
    def run_sfm(self):
        self.detect_features()
        self.match_features()
        camera_matrices = self.estimate_camera_motion()
        point_clouds = self.triangulate_points(camera_matrices)
        point_clouds = self.filter_point_cloud(point_clouds)  # TODO make it work without filtering
        self.visualize_point_cloud(point_clouds)
        return point_clouds