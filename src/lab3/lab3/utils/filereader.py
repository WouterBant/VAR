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
