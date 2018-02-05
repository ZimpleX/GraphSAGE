from __future__ import print_function
import partition as p
import networkx as nx

import time
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import graphsage.utils as gsut
import argparse

def parse_args():
    parser = argparse.ArgumentParser('testing various partitioning algorithms')
    parser.add_argument('-p', '--data_path', type=str, default='../data.ignore/',
        required=False, help='path to append to your dataset')
    parser.add_argument('-d', '--dataset', type=str, 
        required=True, help='name of the data set. e.g., reddit')
    parser.add_argument('-s', '--partition_size', type=int,
        required=True, help='partition size')
    parser.add_argument('--divide_step', type=int,
        required=False, help='used in the greedy divide-conquer algo')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ts = time.time()
    data_dir = '{}/{}/{}'.format(args.data_path,args.dataset,args.dataset)
    G, feats, id_map, walks, class_map = gsut.load_data(data_dir)
    # relabel nodes to be indexed by integer
    mapping = {v:i for i,v in enumerate(G.nodes())}
    G = nx.relabel_nodes(G,mapping)
    te = time.time()
    print('[{}] data loading time: {:6.2f}s'.format('TIME',te-ts))

    ts = time.time()
    partition_baseline = p.partition_random(G, args.partition_size)
    partition_baseline.partitioning()
    adj_len_per_part, avg_deg_per_part = partition_baseline.evaluating()
    print('[{}] average adj list length: {}, average deg: {}'.format('RAND', adj_len_per_part, avg_deg_per_part))
    te = time.time()
    print('[{}] random partitioning time: {:6.2f}s'.format('TIME',te-ts))

    ts = time.time()
    partition_dc = p.partition_divide_conquer(G, args.partition_size, k=args.divide_step)
    partition_dc.partitioning()
    adj_len_per_part, avg_deg_per_part = partition_dc.evaluating()
    print('[{}] average adj list length: {}, average deg: {}'.format('GREEDY', adj_len_per_part, avg_deg_per_part))
    te = time.time()
    print('[{}] greedy partitioning time: {:6.2f}s'.format('TIME',te-ts))
