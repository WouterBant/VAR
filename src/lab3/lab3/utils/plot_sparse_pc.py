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


def plot_sparse_point_cloud(
    points,
    cameras,
    neighbor_threshold=120,
    radius=0.7,
    use_point_colors=False,
    interpolation_iterations=0,
):
    """
    Visualize the sparse point cloud and camera locations using Plotly,
    displaying only points with a minimum number of neighbors.

    :param points: List of points, each with 'coordinates' (X, Y, Z) and optionally 'rgb'.
    :param cameras: List of cameras, each with 'rotation' (R) and 'translation' (t).
    :param neighbor_threshold: Minimum number of neighbors for a point to be displayed.
    :param radius: Radius within which neighbors are counted.
    :param use_point_colors: Boolean to use point colors if available.
    :param interpolation_iterations: Number of times to repeat the interpolation process.
    """
    # Extract point coordinates
    point_coords = np.array([point["coordinates"] for point in points])

    # Use KDTree for efficient neighbor search
    tree = KDTree(point_coords)
    filtered_indices = [
        i
        for i, coord in enumerate(point_coords)
        if len(tree.query_ball_point(coord, radius)) >= neighbor_threshold
    ]

    # Filter points based on neighbor count
    filtered_points = point_coords[filtered_indices]
    if filtered_points.size > 0:
        point_x, point_y, point_z = (
            filtered_points[:, 0],
            filtered_points[:, 1],
            filtered_points[:, 2],
        )
    else:
        point_x, point_y, point_z = [], [], []

    # Determine point colors
    if use_point_colors and all("rgb" in points[i] for i in filtered_indices):
        # Convert RGB values to color strings
        point_colors = [
            f'rgb({points[i]["rgb"][0]},{points[i]["rgb"][1]},{points[i]["rgb"][2]})'
            for i in filtered_indices
        ]
    else:
        # Default to blue if no colors or option not selected
        point_colors = "blue"

    new_points = []
    new_colors = []

    added_points = 0
    for _ in range(interpolation_iterations):
        for i in filtered_indices:
            point = points[i]
            neighbors = tree.query_ball_point(point["coordinates"], 3 * radius)

            # Calculate the average of neighboring point coordinates
            neighbor_coords = point_coords[neighbors]
            new_coord = np.mean(neighbor_coords, axis=0)
            if interpolation_iterations > 1:
                new_coord += np.random.normal(0, 0.05, 3)

            # Radius check: ensure the new point isn't too close to many existing points
            if len(tree.query_ball_point(new_coord, 0.5 * radius)) < 10:
                new_points.append(new_coord)
                added_points += 1
                if added_points % 300 == 0:
                    print(f"Iteration {_ + 1}: Added {added_points} points")

                if use_point_colors:
                    new_rgb = np.mean([points[j]["rgb"] for j in neighbors], axis=0)
                    new_colors.append(f"rgb({new_rgb[0]},{new_rgb[1]},{new_rgb[2]})")

    # Update points and colors
    if new_points:
        new_points = np.array(new_points)
        point_coords = np.vstack([point_coords, new_points])
        point_x = np.hstack([point_x, new_points[:, 0]])
        point_y = np.hstack([point_y, new_points[:, 1]])
        point_z = np.hstack([point_z, new_points[:, 2]])
        if use_point_colors:
            point_colors.extend(new_colors)

    # Rebuild KDTree with new points
    tree = KDTree(point_coords)

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
