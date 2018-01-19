import numpy as np
import sys

"""
The feature vector file (*.npy) for a graph G=(V,E) contains a matrix of dimension R^{|V|xD},
where D is the length of the feature vector for a vertex. 

For a given graph topology, if we want to vary the feature vector length for each vertex,
use this script to generate the corresponding *.npy file.
"""

original_file = 'ppi-feats.npy'
original_feat = np.load(original_file)

num_nodes = original_feat.shape[0]

gen_feats = np.random.rand(num_nodes,int(sys.argv[1]))
out_file = 'ppi-feats{}.npy'.format(sys.argv[1])

np.save(out_file,gen_feats)
