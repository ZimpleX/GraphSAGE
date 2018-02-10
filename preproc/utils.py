"""
We can generate some special graphs here for debugging purpose"
* Disconnected graphs
"""
import math
import networkx as nx
import matplotlib.pyplot as plt


class gen_special_graph(object):

    def __init__(self, num_nodes, avg_degree):
        self.num_nodes = num_nodes
        self.avg_degree = avg_degree
        self.gen_map = {'barbell': self.gen_barbell}

    def draw_graph(self, G):
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        plt.show()

    def gen_barbell(self):
        return nx.barbell_graph(int(math.ceil(self.num_nodes/2)),
                                int(math.floor(self.num_nodes/2)))
