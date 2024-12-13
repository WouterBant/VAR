from utils.filereader import BundleFileReader
from utils.plot_maze import create_2d_grid, display_grid, plot_path
from utils.plot_sparse_pc import plot_sparse_point_cloud
from route_planning import RoutePlanning
import numpy as np

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
    start = (140, 150)
    intermediate = (100, 25)
    end = (30, 175)

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
        # maze_grid = create_2d_grid(reader.points)
        array = [[0 for _ in range(425)] for _ in range(425)]
        for i in range(0, 425):
            array[420-5][i] = 1
            array[421-5][i] = 1
            array[422-5][i] = 1
            array[423-5][i] = 1
            array[424-5][i] = 1
        for i in range(85, 425):
            array[i][0+5] = 1
            array[i][1+5] = 1
            array[i][2+5] = 1
            array[i][3+5] = 1
            array[i][4+5] = 1
        for i in range(170, 340):
            array[i][420-5] = 1
            array[i][421-5] = 1
            array[i][422-5] = 1
            array[i][423-5] = 1
            array[i][424-5] = 1
        for i in range(0, 255):
            array[0+5][i] = 1
            array[1+5][i] = 1
            array[2+5][i] = 1
            array[3+5][i] = 1
            array[4+5][i] = 1
        for i in range(85, 255):
            array[340][i] = 1
            array[341][i] = 1
            array[342][i] = 1
            array[343][i] = 1
            array[344][i] = 1
        for i in range(170, 256):
            array[255][i] = 1
            array[256][i] = 1
            array[257][i] = 1
            array[258][i] = 1
            array[259][i] = 1
        for i in range(255, 341):
            array[i][255] = 1
            array[i][256] = 1
            array[i][257] = 1
            array[i][258] = 1
            array[i][259] = 1
            array[i][85] = 1
            array[i][86] = 1
            array[i][87] = 1
            array[i][88] = 1
            array[i][89] = 1
        for i in range(85, 255):
            array[i][170] = 1
            array[i][171] = 1
            array[i][172] = 1
            array[i][173] = 1
            array[i][174] = 1
        for i in range(85, 170):
            array[i][85] = 1
            array[i][86] = 1
            array[i][87] = 1
            array[i][88] = 1
            array[i][89] = 1
        for i in range(0,85):
            array[i][255] = 1
            array[i][256] = 1
            array[i][257] = 1
            array[i][258] = 1
            array[i][259] = 1
        for i in range(255, 340):
            array[85][i] = 1
            array[86][i] = 1
            array[87][i] = 1
            array[88][i] = 1
            array[89][i] = 1
        for i in range(85, 255):
            array[i][340-5] = 1
            array[i][341-5] = 1
            array[i][342-5] = 1
            array[i][343-5] = 1
            array[i][344-5] = 1
        for i in range(340, 425):
            array[170][i] = 1
            array[171][i] = 1
            array[172][i] = 1
            array[173][i] = 1
            array[174][i] = 1
        # flip the array horizontally
        for i in range(425):
            array[i] = array[i][::-1]
        maze_grid = np.array(array)
        start = (0, 400)
        intermediate = (100, 340)
        end = (424, 40)
        path = RoutePlanning.dijkstra_through_intermediate(
            maze_grid, start, intermediate, end, avoid_wall=True
        )
        # np.save("dijkstra_wall_avoidance_path.npy", path)
        plot_path(maze_grid, path, viz_type)


if __name__ == "__main__":
    main()
