from .Graph import Graph
from collections import deque
from .Time import calculate_time

class BFS:
    """
        Solves graphs using a Best First Search approach
    """
    def __init__(self, graph: Graph):
        self.graph = graph

    def breadth_first_search(self):
        # contains node we have already looked at to prevent cycles
        explored = set()
        
        # frontier contains a tuple of a node and the path from the origin to that node
        # first node added is the origin
        frontier = deque([(self.graph.origin, [self.graph.origin])]) 

        # contains all the destinations that have been found to return them all at once
        found_destinations = {}

        while frontier:
            # get the next element from the deque
            node, path = frontier.popleft()

            # if current node is a destination
            if self.graph.is_goal(node):
                # add it to found_destinations
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

                # if we have all destinations end the algorithm early
                if len(found_destinations) == len(self.graph.goals):
                    break # found all destinations

            if node not in explored:
                # we will now explore this node, so we don't want to do it again
                explored.add(node)
                # for neighbour in the list of neighbours...
                for neighbor, _ in self.graph.get_edges(node): # neighbor_node, cost (cost ignored)
                    if neighbor not in explored:
                        # add it to the end of the deque
                        frontier.append((neighbor, path + [neighbor]))

        # display all paths       
        return list(found_destinations.items())
