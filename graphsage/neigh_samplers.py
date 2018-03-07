from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer
import numpy as np

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        #import pdb; pdb.set_trace()
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists

    def sample_at_batching(self, inputs, adj):
        """ [z]
        similar to the _call function, this function do the sampling when choosing batch,
        i.e., this function samples on the np array (adj), instead of the tf array.

        INPUT:
            inputs      tuple(<list: of root nodes>, <int: number of samples per root>)
            adj         np array
        """
        ids, num_samples = inputs
        adj_roots_T = adj[ids,:].T
        np.random.shuffle(adj_roots_T)
        adj_roots = adj_roots_T.T
        return adj_roots[:,0:num_samples]
