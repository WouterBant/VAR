from utils.filereader import BundleFileReader
from utils.plot_maze import create_2d_grid, display_grid, plot_path
from utils.plot_sparse_pc import plot_sparse_point_cloud
from route_planning import RoutePlanning


def get_visualization_type():
    """Prompt user to choose visualization type."""
    print("\nChoose Visualization Type:")
    options = {
        "1": "sparse_pc",
        "2": "maze",
        "3": "dijkstra_default",
        "4": "dijkstra_int_stop",
        "5": "dijkstra_wall_avoidance",
    }

    for key, value in options.items():
        print(f"{key}. {value}")

    while True:
        choice = input("Enter the number of your chosen visualization type: ")
        if choice in options:
            return options[choice]
        print("Invalid choice. Please try again.")


def get_filter_option():
    """Prompt user about outlier filtering for sparse point cloud."""
    while True:
        choice = input("Do you want to filter outliers? (y/n): ").lower()
        if choice in ["y", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False
        print("Invalid choice. Please enter 'y' or 'n'.")


def get_use_point_colors():
    """Prompt user about using point colors for sparse point cloud."""
    while True:
        choice = input("Do you want to use point colors? (y/n): ").lower()
        if choice in ["y", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False
        print("Invalid choice. Please enter 'y' or 'n'.")


def get_interpolation_iterations():
    """Prompt user for number of interpolation iterations."""
    while True:
        try:
            choice = input("Enter the number of interpolation iterations (default 0): ")
            if not choice:
                return 0
            if int(choice) >= 0:
                return int(choice)
            print("Invalid choice. Please enter a non-negative integer.")
        except ValueError:
            print("Invalid choice. Please enter a non-negative integer.")


def get_data_file():
    """Prompt user for data file path."""
    while True:
        file_path = input(
            "Enter the path to the data file (default: bundle.rd.out): "
        ).strip()
        if not file_path:
            file_path = "bundle.rd.out"

        try:
            reader = BundleFileReader(file_path)
            reader.read_file()
            return reader
        except Exception as e:
            print(f"Error reading file: {e}")
            print("Please check the file path and try again.")


def main():
    # Get data file
    reader = get_data_file()

    # Define some predefined points
    start = (130, 150)
    intermediate = (75, 60)
    end = (25, 175)

    # Get visualization type
    viz_type = get_visualization_type()

    # Additional prompt for sparse point cloud
    if viz_type == "sparse_pc":
        filter_outliers = get_filter_option()
        use_point_colors = get_use_point_colors()
        interpolation_iterations = get_interpolation_iterations()
        neighbor_threshold = 120 if filter_outliers else 0
        plot_sparse_point_cloud(
            points=reader.points,
            cameras=reader.cameras,
            neighbor_threshold=neighbor_threshold,
            use_point_colors=use_point_colors,
            interpolation_iterations=interpolation_iterations,
        )

    elif viz_type == "maze":
        maze_grid = create_2d_grid(reader.points)  # filter out outliers
        display_grid(maze_grid)

    elif viz_type == "dijkstra_default":
        maze_grid = create_2d_grid(reader.points)  # filter out outliers
        path = RoutePlanning.dijkstra(maze_grid, start, end)
        plot_path(maze_grid, path, viz_type)

    elif viz_type == "dijkstra_int_stop":
        maze_grid = create_2d_grid(reader.points)  # filter out outliers
        path = RoutePlanning.dijkstra_through_intermediate(
            maze_grid, start, intermediate, end, avoid_wall=False
        )
        plot_path(maze_grid, path, viz_type)

    elif viz_type == "dijkstra_wall_avoidance":
        maze_grid = create_2d_grid(reader.points)
        path = RoutePlanning.dijkstra_through_intermediate(
            maze_grid, start, intermediate, end, avoid_wall=True
        )
        plot_path(maze_grid, path, viz_type)


if __name__ == "__main__":
    main()
