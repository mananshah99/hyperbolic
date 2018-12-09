import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import scipy.sparse as sp
import time
import random
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict

from encoders import Encoder
from aggregators import MeanAggregator

import snap

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def embed(self, nodes):
        embeds = self.enc(nodes)
        return embeds.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_embeddings(filename, HEADER):
    fin = open(filename, 'r')
    if HEADER:
        node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        if HEADER:
            assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    if HEADER:
        assert len(vectors) == node_num
    return vectors

def load_edgelist(name, edgelist_path, label_path, embedding_path, embedding_header):
    
    if name == "email":
        edgelist_path = "../data/email/email-Eu-core.txt"
        label_path = "../data/email/email-Eu-core-department-labels.txt"

    if name == "chg-miner":
        edgelist_path = "../data/chg-miner/chg-miner-graph.txt"
        label_path = "../data/chg-miner/chg-miner-labels.txt"

    # graph
    graph = snap.LoadEdgeList(snap.PNGraph, edgelist_path, 0, 1, ' ')
    edges = []
    for e in graph.Edges():
        edges.append([e.GetSrcNId(), e.GetDstNId()])
    edges = np.array(edges)

    # labels
    labels_raw = []
    with open(label_path, 'r') as f:
        labels_raw = [l.split(' ')[1].strip('\n') for l in f.readlines()]
    
    labels = np.empty((len(labels_raw), 1), dtype=np.int64)
    for i in range(0, len(labels_raw)):
        labels[i] = int(labels_raw[i])

    # symmetric adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                       shape=(graph.GetNodes(), graph.GetNodes()), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # adj matrix to adj list
    adj_list = defaultdict(set)
    for e in edges:
        adj_list[e[0]].add(e[1])
        adj_list[e[1]].add(e[0])

    # features (zeros)
    if embedding_path is None:
        features = sp.csr_matrix(np.random.normal(size=(graph.GetNodes(), 1000)), dtype=np.float32).todense()
    else:
        embeddings = load_embeddings(embedding_path, embedding_header)
        features = np.zeros(shape=(graph.GetNodes(), len(embeddings[embeddings.keys()[0]])))
        for e in embeddings.keys():
            features[int(e)] = embeddings[e]
        
    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    
    print('Features shape is ' + str(features.shape))
    print('Adjacency shape is ' + str(adj.shape))
    print('Labels shape is ' + str(labels.shape))
    return features, labels, adj_list, adj.shape[0]

def run_edgelist(name="email",
                    edgelist_path    = "../data/email/email-Eu-core.txt", 
                    label_path       = "../data/email/email-Eu-core-department-labels.txt",
                    embedding_path   = "../poincare/embeddings/poincare_email_noburn.txt", # used to initialize + for distances
                    embedding_header = False):

    feat_data, labels, adj_lists, num_nodes = load_edgelist(name, edgelist_path, label_path, embedding_path, embedding_header)
    features = nn.Embedding(num_nodes, feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    # network
    node_ordering_embeddings = load_embeddings(embedding_path, embedding_header)

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, feat_data.shape[1], 128, adj_lists, agg1, gcn=True, cuda=False, ordering_embeddings = None)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False, ordering_embeddings = node_ordering_embeddings)

    # make sure we don't sample -- but change this later?
    enc1.num_sample = None
    enc2.num_sample = None

    graphsage = SupervisedGraphSage(max(labels)[0]+1, enc2)
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:10]
    val = rand_indices[10:11]
    train = list(rand_indices[11:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=1)
    times = []
    # embeds = None
    for batch in range(1000):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        # embeds = graphsage.embed(batch_nodes).detach().numpy()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print batch, loss.data[0]
        
    val_output = graphsage.forward(test) 
    print "Test F1:", f1_score(labels[test], val_output.data.numpy().argmax(axis=1), average="macro")
    print "Test Accuracy:", accuracy_score(labels[test], val_output.data.numpy().argmax(axis=1))
    print "Average batch time:", np.mean(times)   
    
    out = open('embeddings/' + 'graphsage_' + edgelist_path.split('/')[-1], 'wb+')
    embeddings = graphsage.embed(np.arange(num_nodes)).detach().numpy()
    for i in range(0, embeddings.shape[0]):
        s = str(int(i)) + ' '
        s += ' '.join([str(x) for x in embeddings[i]])
        s += '\n'
        out.write(s)

    out.close()
    
if __name__ == "__main__":
    run_edgelist()
