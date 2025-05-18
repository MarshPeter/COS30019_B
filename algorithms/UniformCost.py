import heapq
from .Graph import Graph
from .Time import calculate_time

# currently in classes, maybe not useful
class UniformCost:

    def __init__(self, graph: Graph):
        self.graph = graph

    def uniform_cost_search(self):
        frontier = [] 
        # this adds to the frontier such that the next pop will be the min cost
        # order of list is: [cost, origin, path]
        heapq.heappush(frontier, [0, self.graph.origin, [self.graph.origin]]) 

        # keep track of nodes we have checked
        explored = {}

        # dictionary of all destinations + lowest costs we have found
        # in form of {destination: (path, cost)}
        found_destinations = {}

        while frontier:
            # removes item with smallest cost, reorganizes so that smallest cost is next again
            cost, node, path = heapq.heappop(frontier)

            # we found a destination
            if self.graph.is_goal(node):
                # add a new destination if not seen before
                if node not in found_destinations:
                    found_destinations[node] = (path, cost)
                # if we have found all destinations, then return
                if len(found_destinations) == len(self.graph.goals):
                    return list(found_destinations.items()) # return items for now, maybe change later

            # if the current path to a node is more than what we have found previously
            # discard this path, it will not be optimal
            if node in explored and explored[node] <= cost:
                continue

            # add or update cost of path to node
            explored[node] = cost

            # returns List(Tuple(node_neighbour_number, edge_cost)) for the current node
            edges = self.graph.get_edges(node)

            # add all edges to the heap in form: [cost, origin, path]
            for edge in edges:
                neighbor_cost = calculate_time(scat_a=self.graph.get_position(node), scat_b=self.graph.get_position(edge[0]), flow_at_b=edge[1])

                cumulative = cost + neighbor_cost
                new_path = path + [edge[0]]
                heapq.heappush(frontier, (cumulative, edge[0], new_path))

        # If there is no valid path
        return list(found_destinations.items())
