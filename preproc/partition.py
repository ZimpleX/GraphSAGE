import numpy as np
import networkx as nx
import math
from operator import itemgetter

class partition(object):
    def __init__(self, G, part_size, seed=0):
        """
        ATTRIBUTES:
            G               a networkx graph
            part_size       integer specifying the size of each partition
            seed            integer as a seed to generate random numbers
            partition       2D array of shape Np x self.part_size, where Np=|V|/self.part_size
                            means the number of partitions. 
        """
        self.G = G
        self.part_size = part_size
        num_nodes = self.G.number_of_nodes()
        padded_num_nodes = int(np.ceil(1.*num_nodes/part_size)*part_size)
        for i in range(padded_num_nodes-num_nodes):
            self.G.add_node(-i-1)
        self.partition = np.zeros(padded_num_nodes).reshape(-1, part_size)
        self.seed = seed

    def partitioning(self):
        """
        Partition algorithm. To be overwritten by the subclass functions
        """
        pass
    
    def _sampler(self, v, num_samples=10):
        """
        uniform sampler for node v.
        """
        neighbors = self.G.neighbors(v)
        if len(neighbors) == 0:
            return []
        elif len(neighbors) > num_samples:
            return np.random.choice(neighbors, num_samples, replace=False)
        else:
            return np.random.choice(neighbors, num_samples, replace=True)

    def evaluating(self, num_samples=10, num_runs=1):
        """
        OUTPUT:
            the average length of adj_list for each partition, and
            the average degree after our partitioning
        """
        avg_adj_len_list = []
        avg_sample_len_list = []
        for r in range(num_runs):
            l = [reduce(lambda x,y: x|y, [set(self.G.neighbors(v)) for v in pi]) 
                    for pi in self.partition]
            l_num = np.array([len(li) for li in l])
            avg_adj_length = np.average(l_num)      # average adjacency list length for each partition
            avg_adj_len_list.append(avg_adj_length)
            n = [reduce(lambda x,y: x|y, [set(self._sampler(v)) for v in pi])
                    for pi in self.partition]
            n_num = np.array([len(ni) for ni in n])
            avg_sample_len_list.append(n_num)
        return np.average(avg_adj_len_list)/self.part_size, \
                np.average(avg_sample_len_list)/self.part_size



class partition_random(partition):
    """
    random partitioning scheme
        
    """
    def __init__(self, G, part_size, seed=0):
        super(partition_random, self).__init__(G, part_size, seed=seed)
    
    def partitioning(self):
        """
        partition the vertices by random selection. set self.partition correspondingly. 
        """
        V = np.array(self.G.nodes())
        V_perm = np.random.permutation(V)
        self.partition = V_perm.reshape(len(V)/self.part_size, self.part_size)


class partition_divide_conquer(partition):
    """
    partitioning with greedy and divide-conquer.

    ATTRIBUTES:
        divide_step:        divide the problem into how many subproblems
    """
    def __init__(self, G, part_size, seed=0, k=2):
        super(partition_divide_conquer, self).__init__(G, part_size, seed=seed)
        # you divide each partition into k subpartitions in each recursion.
        self.divide_step = k
        self.status_hash = np.zeros((self.G.number_of_nodes(),self.divide_step))

    def _to_kary_representation(self, ip):
        """
        use this function to divide V into chunks of size:
        self.divide_step to the power of n, so that we can
        proceed with the divide and conquer.
        """
        parts = []
        residue = ip
        while residue > self.divide_step:
            part = self.divide_step**np.floor(math.log(residue, self.divide_step))
            parts.append(int(part))
            residue -= part
        if residue > 0:
            parts.append(int(residue))
        return np.array(parts)

    def partitioning(self):
        V = np.array(self.G.nodes())
        V_perm = np.random.permutation(V)
        parts_idx = self._to_kary_representation(len(V)/self.part_size)
        start_idx_V_perm = 0
        for i,pidx in enumerate(parts_idx):
            end_idx_V_perm = start_idx_V_perm+pidx*self.part_size
            cur_partitions = V_perm[start_idx_V_perm:end_idx_V_perm].reshape(1,-1)
            # sort by degree here: https://groups.google.com/forum/#!topic/networkx-discuss/Bai-YcHQdqg
            #sorted_vertex_deg = sorted(self.G.degree_iter(cur_partitions[0]),key=itemgetter(1))#sorted(_deg_list,key=itemgetter(1))
            #sorted_vertex = [si[0] for si in sorted_vertex_deg]
            #cur_partitions = np.array(sorted_vertex).reshape(cur_partitions.shape)
            ##################################################
            start_idx_V_perm = end_idx_V_perm
            while cur_partitions.shape[1] > self.part_size:
                # _divide_step: in case of the last residue. e.g., when self.divide_step=4,
                # then if there are 3*part_size elements remaining, we need to reset _divide_step to 3
                _divide_step = (i==len(parts_idx)-1) and parts_idx[i] or self.divide_step
                next_partitions = np.zeros((cur_partitions.shape[0]*_divide_step,cur_partitions.shape[1]/_divide_step))
                next_partition_size = cur_partitions.shape[1]/_divide_step
                self.status_hash[...] = 0.
                # can do parallel
                for k,parti in enumerate(cur_partitions):
                    _temp_part = [[] for j in range(_divide_step)]
                    _temp_part_counter = np.array([0]*_divide_step)
                    for ii,vi in enumerate(parti):
                        orig_status_table = self.status_hash[self.G.neighbors(vi)][:,:_divide_step]
                        orig_non_zeros = np.not_equal(0,orig_status_table).sum(axis=0)
                        after_non_zeros = np.not_equal(0,orig_status_table+1).sum(axis=0)
                        avg_fill = ii/_divide_step
                        penalty = (after_non_zeros-orig_non_zeros) \
                                + (_temp_part_counter-avg_fill)/2. \
                                + (next_partition_size+1e-10-_temp_part_counter)**-1
                        dest_partition = np.argmin(penalty)
                        for nbor_i in self.G.neighbors(vi):
                            self.status_hash[nbor_i][dest_partition] += 1
                        _temp_part_counter[dest_partition] += 1
                        _temp_part[dest_partition].append(vi)
                    next_partitions[k*_divide_step:(k+1)*_divide_step] = _temp_part
                cur_partitions = next_partitions
            _ie = sum(parts_idx[:i+1])
            _is = _ie - parts_idx[i]
            self.partition[_is:_ie] = cur_partitions



class partition_GOrder(partition):
    def __init__(self):
        pass
