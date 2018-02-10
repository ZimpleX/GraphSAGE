from __future__ import print_function

import numpy as np
import random
import json
import re
import time

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
    for nodeid in nodes:
        edges = [(nodeid,i) for i in adj_list[nodeid]]
        G.add_edges_from(edges)
    return G

 
def _read_from_txt(data_file):
    G = nx.DiGraph()
    with open(data_file, 'r') as inline:
        for line in inline:
            if line[0] == '#':
                continue
            s,t = [v.strip('\n').strip('\t') for v in re.split(' |\t',line)]
            G.add_edge(s,t)
    return G

def load_data_simple(data_file, max_degree):
    _relabel = True
    ts = time.time()
    if data_file.split('.')[-1] == 'json':
        G_data = json.load(open(data_file))
        G = json_graph.node_link_graph(G_data)
    elif data_file.split('.')[-1] == 'txt':
        G = _read_from_txt(data_file)
    elif data_file.split('.')[-1] == 'edgelist':    # written by nx.write_edgelist()
        _relabel = False
        G = nx.read_edgelist(data_file, nodetype=int)
    else:
        raise Exception('unsupported input format')
    te = time.time()
    print('done reading from input file in {:6.2f}s'.format(te-ts))
    import pdb; pdb.set_trace()
    if _relabel:
        ts = time.time()
        mapping = {v:i for i,v in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        te = time.time()
        print('done relabeling nodes in {:6.2f}s'.format(te-ts))
    
    if max_degree > 0:
        ts = time.time()
        G = _resample_G(G, max_degree)
        te = time.time()
        print('done resampling to max degree of {} in {:6.2f}s'.format(max_degree,te-ts))

    return G


def rewrite_edgelist(data_file):
    G = load_data_simple(data_file, -1)
    nx.write_edgelist(G, '{}.edgelist'.format(data_file))
