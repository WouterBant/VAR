import matplotlib.pyplot as plt

# import os
import yaml
import cv2


class LiveMap:
    def __init__(self, image_path, config_path):
        # Load config
        self.load_config(config_path)

        # Set up the map with a desired size
        self.map_size = (620, 920)  # Map size (width x height in cm)

        # Create a figure and axes for plotting
        aspect_ratio = self.map_size[1] / self.map_size[0]  # height / width
        self.fig, self.ax = plt.subplots(
            figsize=(6, 6 * aspect_ratio)
        )  # Adjust aspect ratio
        self.ax.set_xlim(
            -self.map_size[0] // 2, self.map_size[0] // 2
        )  # X: -300 to +300
        self.ax.set_ylim(
            -self.map_size[1] // 2, self.map_size[1] // 2
        )  # Y: -450 to +450
        self.ax.set_aspect("equal")  # Maintain equal aspect ratio

        # Add a grid for reference
        self.ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Label axes
        self.ax.set_xlabel("X Position (cm)")
        self.ax.set_ylabel("Y Position (cm)")

        # # Load and resize background image
        self.background_image = cv2.imread(image_path)

        # Convert image from BGR (OpenCV) to RGB (matplotlib)
        self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2RGB)

        # Resize the image to match the map size (this will adjust the dimensions)
        self.background_image = cv2.resize(
            self.background_image, (self.map_size[1], self.map_size[0])
        )

        # # Display the background image
        self.ax.imshow(
            self.background_image,
            extent=(
                -self.map_size[0] // 2,
                self.map_size[0] // 2,
                -self.map_size[1] // 2,
                self.map_size[1] // 2,
            ),
        )

        # Add a red dot to represent the robot's position
        (self.robot_marker,) = self.ax.plot([0], [0], "ro", markersize=10)
        target_x, target_y = (
            self.config["target_x_location"],
            self.config["target_y_location"],
        )
        (self.target_marker,) = self.ax.plot(
            [target_x], [target_y], "bo", markersize=10
        )

        # Turn on interactive mode so the plot updates
        plt.ion()
        plt.show()
        plt.pause(0.1)

    def update_plot(self, location):
        """
        Update the robot's location on the live map.
        :param location: tuple (x, y) representing the robot's current position
        """
        x, y = location
        # Update the robot's position on the map
        self.robot_marker.set_data([x], [y])
        self.ax.draw_artist(self.ax.patch)  # Redraw the background
        self.ax.draw_artist(self.robot_marker)  # Redraw the robot marker
        self.fig.canvas.draw()  # Draw the figure
        self.fig.canvas.flush_events()  # Refresh the canvas to show updates

    def load_config(self, config_path):
        # config_path = os.path.join(
        #     os.path.dirname(__file__),
        #     "..",
        #     "..",
        #     "..",
        #     "..",
        #     "..",
        #     "configs",
        #     "lab2",
        #     "config.yaml",
        # )
        # config_path = "/home/angelo/ros2_ws/VAR/configs/lab2/config.yaml"
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def show(self):
        """
        Display the live map.
        This can be called to initialize the first view of the map.
        """
        plt.ion()  # Turn on interactive mode to allow live updates
        plt.show()

    def save_map(self, filename="live_map.png"):
        """
        Save the current state of the live map to an image file.
        :param filename: Name of the output file
        """
        self.fig.savefig(filename)
