import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


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
        point_x, point_y, _ = (
            filtered_points[:, 0],
            filtered_points[:, 2],  # NOTE the two here
            filtered_points[:, 1],
        )
    else:
        point_x, point_y = [], []

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

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap="Greys", origin="lower")
    plt.title("2D Maze Grid")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


def plot_path(grid, path, name="Dijkstra"):
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
    plt.title(f"Shortest Path using {name} Algorithm")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()
