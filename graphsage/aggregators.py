import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def poincare_distance(self, x_vec, y_vec):
        def norm_squared(x):
            return np.square(np.linalg.norm(x))

        x_vec = np.array(x_vec)
        y_vec = np.array(y_vec)
        return np.arccosh(1 + 2 * (norm_squared(x_vec - y_vec)/(norm_squared(x_vec) * norm_squared(y_vec))))

    def forward(self, nodes, to_neighs, num_sample=10, ordering_embeddings = None):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        unique_nodes_reversed = {i:n for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1

        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        if ordering_embeddings is not None:
            # for all rows in mask
            for i in range(0, len(nodes)):
                # for all cols in mask
                for j in range(0, mask.shape[1]):
                    if mask[i][j] > 0:
                        original_node = nodes[i]
                        neighbor_node = unique_nodes_reversed[j] #unique_nodes.keys()[unique_nodes.values().index(j)]
                        dist = self.poincare_distance(ordering_embeddings[str(original_node)], ordering_embeddings[str(neighbor_node)])
                        mask[i][j] * (1/(0.001 + dist))

            #print("num nodes is ", len(nodes))
            #print("0th node is ", nodes[0])
            #print(mask, mask.shape)
            #print("mask 0, 0is", mask[0][1])
            #print("uinque nodes are", unique_nodes)
            #print("corresponding to index", unique_nodes.keys()[unique_nodes.values().index(600)])
            #print(embed_matrix, embed_matrix.shape)
            #print("--------------")

        to_feats = mask.mm(embed_matrix)
        return to_feats
