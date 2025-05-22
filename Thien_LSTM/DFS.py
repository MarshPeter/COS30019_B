from Graph import Graph
from Time import calculate_time

class DepthFirst:
    def __init__(self, graph: Graph, flow: dict):
        self.graph = graph
        self.flow = flow

    def dfs(self):
        stack = [(self.graph.origin, [self.graph.origin], 0)]
        visited = set()
        found_destinations = {}

        while stack:
            node, path, cost_so_far = stack.pop()

            if node in visited:
                continue
            visited.add(node)

            if self.graph.is_goal(node):
                if node not in found_destinations:
                    found_destinations[node] = (path, len(visited))
                if len(found_destinations) == len(self.graph.goals):
                    return found_destinations.items()

            for neighbor, _ in sorted(self.graph.get_edges(node), reverse=True):
                if neighbor not in visited:
                    pos_a = self.graph.get_position(node)
                    pos_b = self.graph.get_position(neighbor)
                    cost = calculate_time(pos_a, pos_b, self.flow.get(neighbor, 500))
                    stack.append((neighbor, path + [neighbor], cost_so_far + cost))

        return None, float('inf')
