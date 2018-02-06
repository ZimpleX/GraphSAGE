from __future__ import print_function
import partition as p
import networkx as nx

import time
import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir)
#import graphsage.utils as gsut

import data_format as df
import argparse

def parse_args():
    parser = argparse.ArgumentParser('testing various partitioning algorithms')
    parser.add_argument('-p', '--data_path', type=str, default='../data.ignore/',
        required=False, help='path to append to your dataset')
    parser.add_argument('-d', '--dataset', type=str, 
        required=True, help='name of the data set. e.g., reddit')
    parser.add_argument('-s', '--partition_size', type=int,
        required=True, help='partition size')
    parser.add_argument('-r', '--resize_degree', type=int, default=128,
        required=False, help='resample graph to this max degree as preprocessing')
    parser.add_argument('--divide_step', type=int, default=2,
        required=False, help='used in the greedy divide-conquer algo')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ts = time.time()
    data_dir = '{}/{}/{}'.format(args.data_path,args.dataset,args.dataset)
    G = df.load_data_simple(data_dir,args.resize_degree)
    te = time.time()
    print('[{}] data loading time: {:6.2f}s'.format('TIME',te-ts))

    ts = time.time()
    partition_baseline = p.partition_random(G, args.partition_size)
    partition_baseline.partitioning()
    adj_deg_per_node, avg_sample_per_node = partition_baseline.evaluating()
    print('[{}] average deg: {}, average samples: {}'.format('RAND', adj_deg_per_node, avg_sample_per_node))
    te = time.time()
    print('[{}] random partitioning time: {:6.2f}s'.format('TIME',te-ts))

    ts = time.time()
    partition_dc = p.partition_divide_conquer(G, args.partition_size, k=args.divide_step)
    partition_dc.partitioning()
    adj_deg_per_node, avg_sample_per_node = partition_dc.evaluating()
    print('[{}] average deg: {}, average samples: {}'.format('GREEDY', adj_deg_per_node, avg_sample_per_node))
    te = time.time()
    print('[{}] greedy partitioning time: {:6.2f}s'.format('TIME',te-ts))
