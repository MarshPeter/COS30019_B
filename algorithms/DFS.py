from .Graph import Graph
from .Time import calculate_time

class DepthFirst:
    def __init__(self, graph: Graph):
        self.graph = graph

    def dfs(self):
        stack = [(self.graph.origin, [self.graph.origin])]
        visited = set()
        found_destinations = {}

        while stack:
            node, path = stack.pop()

            if node in visited:
                continue

            visited.add(node)

            if self.graph.is_goal(node):
                if node not in found_destinations:
                    total_time = 0
                    for i in range(len(path) - 1):
                        current = path[i]
                        next = path[i + 1]
                        pos_current = self.graph.get_position(current)
                        pos_next = self.graph.get_position(next)
                    
                        # Look up time from graph
                        for neighbor, weight in self.graph.get_edges(current):
                            if neighbor == next:
                                time = calculate_time(pos_current, pos_next, weight)
                                total_time += time
                                break
                        
                    found_destinations[node] = (path, total_time)
                if len(found_destinations) == len(self.graph.goals):
                    return list(found_destinations.items())

            for neighbor, _ in sorted(self.graph.get_edges(node), reverse=True):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        return None, float('inf')
