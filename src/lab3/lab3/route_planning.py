import heapq


class RoutePlanning:
    @staticmethod
    def dijkstra(graph, start, end):
        distances = {node: float("infinity") for node in graph}
        distances[start] = 0
        priority_queue = [(0, start)]  # (distance, node)

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            # Relax edges
            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances[end]

    @staticmethod
    def a_star(graph, start, end):
        open_list = []
        closed_list = set()
        g = {node: float("infinity") for node in graph}  # Distance from start to node
        g[start] = 0
        f = {
            node: float("infinity") for node in graph
        }  # Distance from start to end passing through node
        f[start] = RoutePlanning.heuristic(start, end)

        heapq.heappush(open_list, (f[start], start))  # (f, node)

        while open_list:
            current_f, current_node = heapq.heappop(open_list)

            if current_node == end:
                return g[end]

            closed_list.add(current_node)

            # Explore neighbors
            for neighbor, weight in graph[current_node].items():
                if neighbor in closed_list:
                    continue

                tentative_g = g[current_node] + weight

                if tentative_g < g[neighbor]:
                    g[neighbor] = tentative_g
                    f[neighbor] = g[neighbor] + RoutePlanning.heuristic(neighbor, end)
                    heapq.heappush(open_list, (f[neighbor], neighbor))

        return float("infinity")  # If no path exists

    @staticmethod
    def heuristic(node, end):
        # Manhattan distance heuristic for grid-based problems
        return abs(node[0] - end[0]) + abs(node[1] - end[1])
