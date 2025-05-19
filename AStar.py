import heapq
import math
from Graph import Graph
from Time import get_travel_time  

class AStar:
    def __init__(self, graph: Graph):
        self.graph = graph

    def heuristic(self, node, goal):
        x1, y1 = self.graph.node_positions[node]
        x2, y2 = self.graph.node_positions[goal]
        return math.hypot(x2 - x1, y2 - y1)

    def astar(self):
        frontier = []
        heapq.heappush(frontier, (0, 0, self.graph.origin, [self.graph.origin]))
        explored = {}
        found_destinations = {}

        while frontier:
            f, g, node, path = heapq.heappop(frontier)

            if node in explored and explored[node] <= g:
                continue
            explored[node] = g

            if self.graph.is_goal(node):
                if node not in found_destinations:
                    found_destinations[node] = (path, len(explored))
                if len(found_destinations) == len(self.graph.goals):
                    return found_destinations.items()

            for neighbor, _ in self.graph.get_edges(node):
                cost = get_travel_time(node, neighbor)
                new_g = g + cost
                h = min([self.heuristic(neighbor, goal) for goal in self.graph.goals])
                new_f = new_g + h
                heapq.heappush(frontier, (new_f, new_g, neighbor, path + [neighbor]))

        return None, float('inf')