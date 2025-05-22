import heapq
from geopy.distance import geodesic
from Graph import Graph
from Time import calculate_time  

class AStar:
    def __init__(self, graph, flow):
        self.graph = graph
        self.flow = flow

    def heuristic(self, node, goal):
        coord1 = self.graph.get_position(node)
        coord2 = self.graph.get_position(goal)
        return geodesic(coord1, coord2).kilometers

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
                pos_a = self.graph.get_position(node)
                pos_b = self.graph.get_position(neighbor)
                cost = calculate_time(pos_a, pos_b, self.flow.get(neighbor, 500))
                new_g = g + cost
                h = min([self.heuristic(neighbor, goal) for goal in self.graph.goals])
                new_f = new_g + h
                heapq.heappush(frontier, (new_f, new_g, neighbor, path + [neighbor]))

        return None, float('inf')
