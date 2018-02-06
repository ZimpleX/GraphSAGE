from __future__ import print_function

import numpy as np
import random
import json

import networkx as nx
from networkx.readwrite import json_graph

def _resample_G(G, max_degree):
    """
    resize G to have a max degree.
    Assume nodes of G is relabels to be 0,1,2,...,|V|-1
    """
    adj_list = []
    nodes = G.nodes()
    for nodeid in nodes:
        neighbors = G.neighbors(nodeid)
        if len(neighbors) == 0:
            adj_list.append([])
        elif len(neighbors) > max_degree:
            adj_list.append(np.random.choice(neighbors,max_degree,replace=False))
        else:
            adj_list.append(np.random.choice(neighbors,max_degree,replace=True))
    G = nx.DiGraph()    # DiGraph cuz the resampling process does not maintain the undirected property
    for nodeid in nodes:
        G.add_node(nodeid)
        edges = [(nodeid,i) for i in adj_list[nodeid]]
        G.add_edges_from(edges)
    return G

 

def load_data_simple(prefix, max_degree):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    mapping = {v:i for i,v in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
    print('now resampling to max degree of {}'.format(max_degree))
    G = _resample_G(G, max_degree)

    return G


           
