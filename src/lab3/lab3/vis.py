import plotly.graph_objects as go
from scipy.spatial import KDTree
import numpy as np
import heapq
import matplotlib.pyplot as plt


class BundleFileReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.num_cameras = 0
        self.num_points = 0
        self.cameras = []  # To store camera parameters
        self.points = []  # To store 3D points

    def read_file(self):
        """
        Reads the bundle file and extracts camera parameters and 3D points.
        """
        with open(self.filepath, "r") as file:
            lines = file.readlines()

        # Parse header
        assert lines[0].startswith("# Bundle file"), "Invalid bundle file format"
        self.num_cameras, self.num_points = map(int, lines[1].split())

        # Read camera data
        index = 2  # Start after header
        for _ in range(self.num_cameras):
            focal_length = list(map(float, lines[index].split()))
            rotation = [list(map(float, lines[index + i].split())) for i in range(1, 4)]
            translation = list(map(float, lines[index + 4].split()))
            self.cameras.append(
                {
                    "focal_length": focal_length[0],
                    "rotation": rotation,
                    "translation": translation,
                }
            )
            index += 5  # Move to the next camera block

        # Read points data
        for _ in range(self.num_points):
            # Parse 3D coordinates
            coords = list(map(float, lines[index].split()))
            index += 1

            # Parse RGB color
            rgb = list(map(int, lines[index].split()))
            index += 1

            # Parse visibility data
            visibility_data = lines[index].split()
            num_views = int(visibility_data[0])
            self.points.append(
                {"rgb": rgb, "num_views": num_views, "coordinates": coords}
            )
            index += 1

    def print_camera_info(self):
        """
        Print camera information to verify the reading process.
        """
        for i, cam in enumerate(self.cameras):
            print(f"Camera {i + 1}:")
            print(f"  Focal Length: {cam['focal_length']}")
            print(f"  Rotation Matrix: {cam['rotation']}")
            print(f"  Translation Vector: {cam['translation']}")
            print()

    def print_points_info(self, num_points=5):
        """
        Print information about the 3D points.
        :param num_points: Number of points to print (default: 5).
        """
        for i, point in enumerate(self.points[:num_points]):
            print(f"Point {i + 1}:")
            print(f"  RGB: {point['rgb']}")
            print(f"  Coordinates: {point['coordinates']}")
            print(f"  Visible in {point['num_views']} views")
            print()


def plot_sparse_point_cloud(points, cameras):
    """
    Visualize the sparse point cloud and camera locations using Plotly.

    :param points: List of points, each with 'coordinates' (X, Y, Z).
    :param cameras: List of cameras, each with 'translation' (X, Y, Z).
    """
    # Extract point coordinates
    point_coords = [point["coordinates"] for point in points]
    point_x, point_y, point_z = zip(*point_coords)

    # Extract camera locations
    camera_coords = [cam["translation"] for cam in cameras]
    cam_x, cam_y, cam_z = zip(*camera_coords)

    # Create the point cloud trace
    points_trace = go.Scatter3d(
        x=point_x,
        y=point_y,
        z=point_z,
        mode="markers",
        marker=dict(size=2, color="blue"),
        name="Points",
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
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        title="Sparse Point Cloud and Camera Locations",
        legend=dict(x=0.8, y=0.9),
    )
    fig.show()


def plot_sparse_point_cloud2(points, cameras, neighbor_threshold=100, radius=1.0):
    """
    Visualize the sparse point cloud and camera locations using Plotly,
    displaying only points with a minimum number of neighbors.

    :param points: List of points, each with 'coordinates' (X, Y, Z).
    :param cameras: List of cameras, each with 'translation' (X, Y, Z).
    :param neighbor_threshold: Minimum number of neighbors for a point to be displayed.
    :param radius: Radius within which neighbors are counted.
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

    # Extract camera locations
    camera_coords = [cam["translation"] for cam in cameras]
    cam_x, cam_y, cam_z = zip(*camera_coords)

    # Create the point cloud trace
    points_trace = go.Scatter3d(
        x=point_x,
        y=point_y,
        z=point_z,
        mode="markers",
        marker=dict(size=2, color="blue"),
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
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        title="Filtered Sparse Point Cloud and Camera Locations",
        legend=dict(x=0.8, y=0.9),
    )
    fig.show()


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


def plot_sparse_point_cloud3(points, cameras, neighbor_threshold=100, radius=1.0):
    """
    Visualize the sparse point cloud and camera locations using Plotly,
    displaying only points with a minimum number of neighbors.

    :param points: List of points, each with 'coordinates' (X, Y, Z).
    :param cameras: List of cameras, each with 'rotation' (R) and 'translation' (t).
    :param neighbor_threshold: Minimum number of neighbors for a point to be displayed.
    :param radius: Radius within which neighbors are counted.
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
        marker=dict(size=2, color="blue"),
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


def create_2d_grid(points, radius=0.7, neighbor_threshold=120):
    """
    Creates a 2D grid representation of the maze from 3D points.

    :param points: List of 3D points, each with 'coordinates' (X, Y, Z).
    :param grid_size: Resolution of the grid (e.g., 0.5 meters per cell).
    :param grid_extent: Extent of the grid in each direction (default: -20 to 20).
    :return: 2D NumPy array representing the maze.
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
            filtered_points[:, 2],
            filtered_points[:, 1],
        )
    else:
        point_x, point_y, point_z = [], [], []

    # Create 2D grid
    max_x, max_y = max(point_x), max(point_y)
    min_x, min_y = min(point_x), min(point_y)
    factor = 10
    grid = np.zeros(
        (int((max_x - min_x) * factor) + 1, int((max_y - min_y) * factor) + 1)
    )
    for x, y in zip(point_x, point_y):
        x, y = int((x - min_x) * factor), int((y - min_y) * factor)
        grid[x, y] = 1
    return grid


def display_grid(grid):
    """
    Displays the grid using matplotlib.

    :param grid: 2D NumPy array representing the maze.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap="Greys", origin="lower")
    plt.title("2D Maze Grid")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


def dijkstra(grid, start, end):
    """
    Implements Dijkstra's algorithm for pathfinding on a 2D grid.

    :param grid: 2D NumPy array representing the maze (0: free, 1: obstacle).
    :param start: Start coordinate as (x, y).
    :param end: End coordinate as (x, y).
    :return: List of coordinates forming the shortest path.
    """
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Grid dimensions
    rows, cols = grid.shape

    # Distance array: Initialize with infinity
    dist = np.full_like(grid, float("inf"), dtype=float)
    dist[start[1], start[0]] = 0  # Start distance is 0

    # Previous array to reconstruct path
    prev = np.full_like(grid, None, dtype=object)

    # Min-heap priority queue (distance, (x, y))
    pq = [(0, start)]  # Starting point

    while pq:
        current_dist, (x, y) = heapq.heappop(pq)

        # If we reach the destination
        if (x, y) == end:
            break

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < cols and 0 <= ny < rows and grid[ny, nx] == 0
            ):  # Valid and free space
                new_dist = current_dist + 1  # Assumes uniform cost (1 per move)
                if new_dist < dist[ny, nx]:
                    dist[ny, nx] = new_dist
                    prev[ny, nx] = (x, y)
                    heapq.heappush(pq, (new_dist, (nx, ny)))

    # Reconstruct the path from end to start
    path = []
    curr = end
    while curr != start:
        path.append(curr)
        curr = prev[curr[1], curr[0]]
    path.append(start)
    path.reverse()

    return path


def plot_path(grid, path):
    """
    Plot the grid with the shortest path overlaid using matplotlib.

    :param grid: 2D NumPy array representing the maze.
    :param path: List of coordinates representing the shortest path.
    """
    # Create a plot
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap="Greys", origin="lower")  # Maze in black and white
    path_x, path_y = zip(*path)
    plt.plot(
        path_x, path_y, color="red", marker="o", markersize=5, label="Shortest Path"
    )
    plt.title("Shortest Path using Dijkstra's Algorithm")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()


def find_path_through_intermediate(grid, start, intermediate, end):
    """
    Find a path that goes from the start to the intermediate point, then to the end.

    :param grid: 2D NumPy array representing the maze.
    :param start: Start coordinate as (x, y).
    :param intermediate: Intermediate coordinate as (x, y).
    :param end: End coordinate as (x, y).
    :return: List of coordinates forming the complete path.
    """
    # Find the path from start to intermediate
    path_start_to_intermediate = dijkstra(grid, start, intermediate)

    # Find the path from intermediate to end
    path_intermediate_to_end = dijkstra(grid, intermediate, end)

    # Combine the paths (remove the intermediate point from the second path)
    complete_path = path_start_to_intermediate[:-1] + path_intermediate_to_end

    return complete_path


if __name__ == "__main__":
    # Example usage
    reader = BundleFileReader("bundle.rd.out")
    reader.read_file()
    # reader.print_camera_info()
    # reader.print_points_info()
    # plot_sparse_point_cloud3(reader.points, reader.cameras, neighbor_threshold=120, radius=0.7)
    maze_grid = create_2d_grid(reader.points)

    # Define start and end coordinates in grid space (x, y)
    start = (130, 150)  # Example: Start at (10, 10)
    intermediate = (75, 60)
    end = (25, 175)  # Example: End at (15, 15)

    # Run Dijkstra's algorithm
    path = find_path_through_intermediate(maze_grid, start, intermediate, end)

    # Plot the grid with the shortest path
    plot_path(maze_grid, path)

    # Display the grid
    # display_grid(maze_grid)
