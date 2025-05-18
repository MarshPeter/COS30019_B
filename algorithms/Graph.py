from collections import defaultdict

class Graph:
    """
        Contains all relevant information of a graph that is required by all possible solutions
        member variables: 
    """
    def __init__(self):
        self.origin = None # will become a number
        self.goals = None # will be a list of numbers
        self.edges = defaultdict(list) # will be set up such that: {node: [(neighbor1, flow), neighbor2, flow)...]} (NOTE: flow is from node to neighbor)
        self.node_positions = defaultdict(tuple) # {node_number: (x, y)}

    # sets the position of a number {node_number: (x, y)} NOTE: x is LAT, y is LON
    def add_node(self, node, pos_x, pos_y):
        self.node_positions[node] = (pos_x, pos_y)

    # sets the origin node
    def set_origin(self, origin):
        self.origin = origin

    # sets goals to a list of goals 
    def set_goals(self, goals):
        self.goals = goals

    # adds an edge to the graph {node: [(neighbor1, flow), (neighbor2, flow)...]}
    def add_neighbor(self, node, neighbor_node, cost):
        self.edges[node].append((neighbor_node, cost))

    # returns true if the parameter node number is in the goals list
    def is_goal(self, node):
        return node in self.goals

    # return all neighbors and costs of a node
    def get_edges(self, node):
        return self.edges[node]

    # returns a tuple where the position is (pos_x, pos_y)
    def get_position(self, node):
        return self.node_positions[node]
