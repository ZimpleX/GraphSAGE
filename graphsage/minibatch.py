from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)


####################
# FOR UNSUPERVISED #
####################
class EdgeMinibatchIterator(object):
    
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """
    def __init__(self, G, id2idx, 
            placeholders, context_pairs=None, batch_size=100, max_degree=25,
            n2v_retrain=False, fixed_n2v=False,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0

        self.nodes = np.random.permutation(G.nodes())
        # [z]: self.adj is resampled to have uniform degree.
        #   self.deg, however, is the degree before resampling.
        self.adj, self.deg = self.construct_adj()
        # [z]: test_adj is just the graph resampled.
        self.test_adj = self.construct_test_adj()
        if context_pairs is None:
            edges = G.edges()
        else:
            edges = context_pairs
        self.train_edges = self.edges = np.random.permutation(edges)
        if not n2v_retrain:
            self.train_edges = self._remove_isolated(self.train_edges)
            self.val_edges = [e for e in G.edges() if G[e[0]][e[1]]['train_removed']]
        else:
            if fixed_n2v:
                self.train_edges = self.val_edges = self._n2v_prune(self.edges)
            else:
                self.train_edges = self.val_edges = self.edges

        print(len([n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]), 'train nodes')
        print(len([n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]), 'test nodes')
        self.val_set_size = len(self.val_edges)

    def _n2v_prune(self, edges):
        is_val = lambda n : self.G.node[n]["val"] or self.G.node[n]["test"]
        return [e for e in edges if not is_val(e[1])]

    def _remove_isolated(self, edge_list):
        new_edge_list = []
        missing = 0
        for n1, n2 in edge_list:
            if not n1 in self.G.node or not n2 in self.G.node:
                missing += 1
                continue
            if (self.deg[self.id2idx[n1]] == 0 or self.deg[self.id2idx[n2]] == 0) \
                    and (not self.G.node[n1]['test'] or self.G.node[n1]['val']) \
                    and (not self.G.node[n2]['test'] or self.G.node[n2]['val']):
                continue
            else:
                new_edge_list.append((n1,n2))
        print("Unexpected missing:", missing)
        return new_edge_list

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size, 
            len(edge_list))]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(self.val_edges), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(node_list), val_edges

    def label_val(self):
        train_edges = []
        val_edges = []
        for n1, n2 in self.G.edges():
            if (self.G.node[n1]['val'] or self.G.node[n1]['test'] 
                    or self.G.node[n2]['val'] or self.G.node[n2]['test']):
                val_edges.append((n1,n2))
            else:
                train_edges.append((n1,n2))
        return train_edges, val_edges

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0


############################
# FOR SUPERVISED MINIBATCH #
############################
class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """

    # [z]: NOTE: downsampled adj list
    def __init__(self, G, id2idx, layer_infos,
            placeholders, placeholder_nr, label_map, num_classes, 
            batch_size=100, max_degree=25, **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.layer_infos = layer_infos
        self.placeholders = placeholders
        self.placeholder_nr = placeholder_nr
        self.batch_size = batch_size
        self.batch_nodes = None
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes
        # self.deg: array of size |V|
        # self.adj: array of R^{|V|xself.max_degree}
        #           after the self.construct_adj() method
        # [z]: note, you should not have this adj storing id, you should have it storing idx
        # [z]: i.e., you should not use id2idx onwards
        self.adj, self.deg = self.construct_adj()   # resample graph to uniform deg.
        self.test_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        """
        resample graph to be of uniform degree --> 
        this enables you to represent adj matrix as a 2D array
        """
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes=None, val=False):
        if batch_nodes is not None:
            batch1id = batch_nodes
        else:
            batch1id = self.batch_nodes
        batch1 = [self.id2idx[n] for n in batch1id]
              
        labels = np.vstack([self._make_label_vec(node) for node in batch1id])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch1)})
        feed_dict.update({self.placeholders['batch']: batch1})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels

    def batch_feed_dict_nodereuse(self, batch_hop_1, batch_hop_2,
                              batch_adj_0_1, batch_adj_1_2, val=False):
        batch1_hop1_id = batch_hop_1
        batch1_hop1 = [self.id2idx[n] for n in batch1_hop1_id]
        batch1_hop2_id = batch_hop_2
        batch1_hop2 = [self.id2idx[n] for n in batch1_hop2_id]
        feed_dict_nr = dict()
        feed_dict_nr.update({self.placeholder_nr['batch_hop_1']: batch1_hop1})
        feed_dict_nr.update({self.placeholder_nr['batch_hop_2']: batch1_hop2})
        feed_dict_nr.update({self.placeholder_nr['num_hop_1']: len(batch1_hop1)})
        feed_dict_nr.update({self.placeholder_nr['num_hop_2']: len(batch1_hop2)})
        feed_dict_nr.update({self.placeholder_nr['batch_adj_0_1']: batch_adj_0_1.flatten()})
        feed_dict_nr.update({self.placeholder_nr['batch_adj_1_2']: batch_adj_1_2.flatten()})

        return feed_dict_nr

    def node_val_feed_dict(self, size=None, test=False):
        """
        Caller: supervised_train/evaluate()
        """
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(batch_nodes=val_nodes)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        """
        Caller: supervised_train/evaluate()
            - for non sigmoid activation
        """
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size, 
            len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(batch_nodes=val_node_subset)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        """
        Caller: supervised_train/train()
        IMPORTANT: here return also the adj matrix of the layers,
        as well as the support vectors for each layer
        """
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        self.batch_nodes = self.train_nodes[start_idx : end_idx]
        #################################
        # sample support neighbors here #
        #################################
        # for l in layers:
        #np.random.choice(self.adj[r], sample_size[l], replace=False)
        return self.batch_feed_dict()

    def next_sample_subgraph_feed_dict(self):
        batch_size = self.batch_nodes.shape[0]
        s1 = self.layer_infos[1].num_samples    # note here s1 is referring to layer 2 in forward prop
        s2 = self.layer_infos[0].num_samples    # s2 is referring to layer 1 in forward prop
        # here you have the actual sampled nodes
        _sampler = self.layer_infos[0].neigh_sampler
        l1_samples_mat = _sampler.sample_at_batching([self.batch_nodes,s1], self.adj)
        # unique samples for the 1-hop neighbors
        l1_samples = np.sort(np.unique(l1_samples_mat)).astype(np.int)
        # unique samples for the 2-hop neighbors
        l2_samples_mat = _sampler.sample_at_batching([l1_samples,s2], self.adj)
        l2_samples = np.sort(np.unique(l2_samples_mat)).astype(np.int)
        adj_0_1 = np.zeros((self.batch_nodes.shape[0], l1_samples.shape[0]))
        adj_1_2 = np.zeros((l1_samples.shape[0], l2_samples.shape[0]))
        l1_index = {si:i for i,si in enumerate(l1_samples)}
        l2_index = {si:i for i,si in enumerate(l2_samples)}
        idx01_map = np.vectorize(lambda x: l1_index[x])
        idx12_map = np.vectorize(lambda x: l2_index[x])

        adj01_idx_ax1 = idx01_map(l1_samples_mat)
        adj01_idx_ax0 = (np.arange(batch_size)*np.ones((s1,batch_size)).astype(np.int)).T
        adj_0_1[adj01_idx_ax0.flatten().astype(np.int),adj01_idx_ax1.flatten()] = 1.
        norm = adj_0_1.sum(axis=1).reshape(-1,1)
        adj_0_1 = adj_0_1/norm
        adj12_idx_ax1 = idx12_map(l2_samples_mat)
        adj12_idx_ax0 = (np.arange(l1_samples.shape[0])*np.ones((s2,l1_samples.shape[0])).astype(np.int)).T
        adj_1_2[adj12_idx_ax0.flatten().astype(np.int),adj12_idx_ax1.flatten()] = 1.
        norm = adj_1_2.sum(axis=1).reshape(-1,1)
        adj_1_2 = adj_1_2/norm
        return self.batch_feed_dict_nodereuse(l1_samples, l2_samples, adj_0_1.astype(np.float32), adj_1_2.astype(np.float32))
        ####################
        # you probably don't want to use the provided neighbor sampler,
        # cuz that will incur much overhead in tf session.
        # uniform neighbor sampler, then flatten and set
        #batch_hop_1 = 
        #batch_hop_2 = 
        #batch_adj_0_1 = 
        #batch_adj_1_2 = 
        #return self.batch_feed_dict_nodereuse(batch_hop_1, batch_hop_2, batch_adj_0_1, batch_adj_1_2)
        

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        return self.batch_feed_dict(batch_nodes=val_nodes), (iter_num+1)*size >= len(node_list), val_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0
