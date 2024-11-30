import heapq
import numpy as np
import scipy.ndimage as ndimage


class RoutePlanning:
    @staticmethod
    def calculate_wall_distance_cost(grid):
        """
        Precompute distance to the nearest wall for each cell.

        :param grid: 2D NumPy array representing the maze (0: free, 1: obstacle).
        :return: 2D NumPy array of distances to the nearest wall.
        """
        # Create a binary image where walls are white (True)
        binary_grid = grid.astype(bool)

        # Compute distance transform
        # This calculates the distance from each cell to the nearest wall
        wall_distance = ndimage.distance_transform_edt(~binary_grid)

        return wall_distance

    @staticmethod
    def dijkstra_with_wall_distance(
        grid, start, end, wall_weight=50.0
    ):  # NOTE if resizing the maze this should also be changed
        """
        Implements Dijkstra's algorithm with wall distance penalty.

        :param grid: 2D NumPy array representing the maze (0: free, 1: obstacle).
        :param start: Start coordinate as (x, y).
        :param end: End coordinate as (x, y).
        :param wall_weight: Multiplier for wall distance cost.
        :return: List of coordinates forming the shortest path.
        """
        # Precompute wall distances
        wall_distances = RoutePlanning.calculate_wall_distance_cost(grid)

        # Directions: cardinal + diagonal moves
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),  # cardinal moves
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),  # diagonal moves
        ]

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

                # Calculate move cost (diagonal moves cost √2, cardinal moves cost 1)
                move_cost = np.sqrt(2) if dx != 0 and dy != 0 else 1

                if 0 <= nx < cols and 0 <= ny < rows and grid[ny, nx] == 0:
                    # Get wall distance cost for this cell
                    wall_distance_cost = wall_weight * (
                        1 / (wall_distances[ny, nx] + 1)
                    )

                    # Total cost: base move cost + wall distance penalty
                    new_dist = current_dist + move_cost + wall_distance_cost

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

    @staticmethod
    def dijkstra(grid, start, end):
        """
        Implements Dijkstra's algorithm for pathfinding on a 2D grid with cardinal and diagonal moves.

        :param grid: 2D NumPy array representing the maze (0: free, 1: obstacle).
        :param start: Start coordinate as (x, y).
        :param end: End coordinate as (x, y).
        :return: List of coordinates forming the shortest path.
        """
        # Directions: cardinal + diagonal moves
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),  # cardinal moves
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),  # diagonal moves
        ]

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

                # Calculate move cost (diagonal moves cost √2, cardinal moves cost 1)
                move_cost = np.sqrt(2) if dx != 0 and dy != 0 else 1

                if (
                    0 <= nx < cols and 0 <= ny < rows and grid[ny, nx] == 0
                ):  # Valid and free space
                    new_dist = current_dist + move_cost
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

    @staticmethod
    def dijkstra_through_intermediate(grid, start, intermediate, end, avoid_wall=True):
        """
        Find a path that goes from the start to the intermediate point, then to the end.

        :param grid: 2D NumPy array representing the maze.
        :param start: Start coordinate as (x, y).
        :param intermediate: Intermediate coordinate as (x, y).
        :param end: End coordinate as (x, y).
        :return: List of coordinates forming the complete path.
        """
        if avoid_wall:
            path_start_to_intermediate = RoutePlanning.dijkstra_with_wall_distance(
                grid, start, intermediate
            )
            path_intermediate_to_end = RoutePlanning.dijkstra_with_wall_distance(
                grid, intermediate, end
            )
        else:
            path_start_to_intermediate = RoutePlanning.dijkstra(
                grid, start, intermediate
            )
            path_intermediate_to_end = RoutePlanning.dijkstra(grid, intermediate, end)

        complete_path = path_start_to_intermediate[:-1] + path_intermediate_to_end

        return complete_path
