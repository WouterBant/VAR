import numpy as np
import plotly.graph_objects as go
from scipy.spatial import KDTree


def compute_camera_positions(cameras):
    """
    Compute the camera positions in world coordinates.

    :param cameras: List of cameras, each with 'rotation' and 'translation'.
    :return: List of camera positions in world coordinates.
    """
    camera_positions = []
    for cam in cameras:
        R = np.array(cam["rotation"])  # Rotation matrix (3x3)
        t = np.array(cam["translation"]).reshape(3, 1)  # Translation vector (3x1)
        position = -np.dot(R.T, t).flatten()  # Transform to world coordinates
        camera_positions.append(position)
    return camera_positions


def interpolate_wall_points(points, neighbor_threshold=120, radius=0.7):
    """
    Interpolate missing wall points based on the existing point cloud.

    :param points: List of points, each with 'coordinates' (X, Y, Z) and optionally 'rgb'.
    :param neighbor_threshold: Minimum number of neighbors for a point to be displayed.
    :param radius: Radius within which neighbors are counted.
    :return: List of original points with interpolated wall points added.
    """
    # Extract point coordinates
    point_coords = np.array([point["coordinates"] for point in points])

    # Use KDTree for efficient neighbor search
    tree = KDTree(point_coords)

    # Find points with insufficient neighbors
    filtered_indices = [
        i
        for i, coord in enumerate(point_coords)
        if len(tree.query_ball_point(coord, radius)) < neighbor_threshold
    ]

    # Interpolate missing points
    new_points = []
    for i in filtered_indices:
        point = points[i]
        neighbors = tree.query_ball_point(point["coordinates"], radius)

        # Calculate the average of neighboring point coordinates
        neighbor_coords = point_coords[neighbors]
        new_coord = np.mean(neighbor_coords, axis=0)

        # Create a new point with the interpolated coordinates and the same color
        new_point = {"coordinates": new_coord, "rgb": point.get("rgb", (0, 0, 0))}
        new_points.append(new_point)

    # Combine original and new points
    new_points = []
    all_points = points + new_points
    return all_points


def plot_sparse_point_cloud3(
    points, cameras, neighbor_threshold=120, radius=0.7, use_point_colors=False
):
    """
    Visualize the sparse point cloud and camera locations using Plotly,
    displaying only points with a minimum number of neighbors and
    interpolating missing wall points.

    :param points: List of points, each with 'coordinates' (X, Y, Z) and optionally 'rgb'.
    :param cameras: List of cameras, each with 'rotation' (R) and 'translation' (t).
    :param neighbor_threshold: Minimum number of neighbors for a point to be displayed.
    :param radius: Radius within which neighbors are counted.
    :param use_point_colors: Boolean to use point colors if available.
    """
    # Filter points based on neighbor count
    filtered_points = interpolate_wall_points(points, neighbor_threshold, radius)

    # Extract point coordinates
    point_x, point_y, point_z = [], [], []
    for point in filtered_points:
        point_x.append(point["coordinates"][0])
        point_y.append(point["coordinates"][1])
        point_z.append(point["coordinates"][2])

    # Determine point colors
    if use_point_colors and all("rgb" in point for point in filtered_points):
        # Convert RGB values to color strings
        point_colors = [
            f'rgb({point["rgb"][0]},{point["rgb"][1]},{point["rgb"][2]})'
            for point in filtered_points
        ]
    else:
        # Default to blue if no colors or option not selected
        point_colors = "blue"

    # Compute camera positions in world coordinates
    camera_positions = compute_camera_positions(cameras)
    camera_positions = np.array(camera_positions)
    cam_x, cam_y, cam_z = (
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
    )

    # Create the point cloud trace
    points_trace = go.Scatter3d(
        x=point_x,
        y=point_y,
        z=point_z,
        mode="markers",
        marker=dict(size=2, color=point_colors),
        name="Filtered Points",
    )

    # Create the camera trace
    cameras_trace = go.Scatter3d(
        x=cam_x,
        y=cam_y,
        z=cam_z,
        mode="markers",
        marker=dict(size=6, color="green", symbol="square"),
        name="Cameras",
    )

    # Combine traces and plot
    fig = go.Figure(data=[points_trace, cameras_trace])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[-20, 20]),  # Fixed X-axis range
            yaxis=dict(title="Y", range=[-20, 20]),  # Fixed Y-axis range
            zaxis=dict(title="Z", range=[-20, 20]),  # Fixed Z-axis range
        ),
        title="Filtered Sparse Point Cloud and Camera Locations",
        legend=dict(x=0.8, y=0.9),
    )
    fig.show()
