from Graph import Graph
from Time import get_travel_time 

class DepthFirst:
    def __init__(self, graph: Graph):
        self.graph = graph

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
                    cost = get_travel_time(node, neighbor)
                    stack.append((neighbor, path + [neighbor], cost_so_far + cost))

        return None, float('inf')