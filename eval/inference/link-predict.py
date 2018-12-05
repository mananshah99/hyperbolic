# Significantly inspired by (and derived from): https://github.com/lucashu1/link-prediction
from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.manifold import spectral_embedding
from sklearn.linear_model import LogisticRegression
import time
import os
import tensorflow as tf
from predict_utils import mask_test_edges
import pickle
from copy import deepcopy

EMBEDDINGS_FILE = '../../baseline/embeddings/node2vec_email.txt'
GRAPH_FILE = '../../data/email/email-Eu-core.txt'
HEADER = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input: positive test/val edges, negative test/val edges, edge score matrix
# Output: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=False):

    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(1) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(0) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    # roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    # return roc_score, roc_curve_tuple, ap_score
    return roc_score, ap_score

# Return a list of tuples (node1, node2) for networkx link prediction evaluation
def get_ebunch(train_test_split):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split
 
    test_edges_list = test_edges.tolist() # convert to nested list
    test_edges_list = [tuple(node_pair) for node_pair in test_edges_list] # convert node-pairs to tuples
    test_edges_false_list = test_edges_false.tolist()
    test_edges_false_list = [tuple(node_pair) for node_pair in test_edges_false_list]
    return (test_edges_list + test_edges_false_list)

# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def adamic_adar_scores(g_train, train_test_split):
    if g_train.is_directed(): # Only works for undirected graphs
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    
    aa_scores = {}

    # Calculate scores
    aa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.adamic_adar_index(g_train, ebunch=get_ebunch(train_test_split)): # (u, v) = node indices, p = Adamic-Adar index
        aa_matrix[u][v] = p
        aa_matrix[v][u] = p # make sure it's symmetric
    aa_matrix = aa_matrix / aa_matrix.max() # Normalize matrix

    runtime = time.time() - start_time
    aa_roc, aa_ap = get_roc_score(test_edges, test_edges_false, aa_matrix)

    aa_scores['test_roc'] = aa_roc
    # aa_scores['test_roc_curve'] = aa_roc_curve
    aa_scores['test_ap'] = aa_ap
    aa_scores['runtime'] = runtime
    return aa_scores


# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def jaccard_coefficient_scores(g_train, train_test_split):
    if g_train.is_directed(): # Jaccard coef only works for undirected graphs
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    jc_scores = {}

    # Calculate scores
    jc_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.jaccard_coefficient(g_train, ebunch=get_ebunch(train_test_split)): # (u, v) = node indices, p = Jaccard coefficient
        jc_matrix[u][v] = p
        jc_matrix[v][u] = p # make sure it's symmetric
    jc_matrix = jc_matrix / jc_matrix.max() # Normalize matrix

    runtime = time.time() - start_time
    jc_roc, jc_ap = get_roc_score(test_edges, test_edges_false, jc_matrix)

    jc_scores['test_roc'] = jc_roc
    # jc_scores['test_roc_curve'] = jc_roc_curve
    jc_scores['test_ap'] = jc_ap
    jc_scores['runtime'] = runtime
    return jc_scores


# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def preferential_attachment_scores(g_train, train_test_split):
    if g_train.is_directed(): # Only defined for undirected graphs
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    pa_scores = {}

    # Calculate scores
    pa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.preferential_attachment(g_train, ebunch=get_ebunch(train_test_split)): # (u, v) = node indices, p = Jaccard coefficient
        pa_matrix[u][v] = p
        pa_matrix[v][u] = p # make sure it's symmetric
    pa_matrix = pa_matrix / pa_matrix.max() # Normalize matrix

    runtime = time.time() - start_time
    pa_roc, pa_ap = get_roc_score(test_edges, test_edges_false, pa_matrix)

    pa_scores['test_roc'] = pa_roc
    # pa_scores['test_roc_curve'] = pa_roc_curve
    pa_scores['test_ap'] = pa_ap
    pa_scores['runtime'] = runtime
    return pa_scores


# Input: train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def spectral_clustering_scores(train_test_split, random_state=0):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    sc_scores = {}

    # Perform spectral clustering link prediction
    spectral_emb = spectral_embedding(adj_train, n_components=16, random_state=random_state)
    sc_score_matrix = np.dot(spectral_emb, spectral_emb.T)

    runtime = time.time() - start_time
    sc_test_roc, sc_test_ap = get_roc_score(test_edges, test_edges_false, sc_score_matrix, apply_sigmoid=True)
    sc_val_roc, sc_val_ap = get_roc_score(val_edges, val_edges_false, sc_score_matrix, apply_sigmoid=True)

    # Record scores
    sc_scores['test_roc'] = sc_test_roc
    # sc_scores['test_roc_curve'] = sc_test_roc_curve
    sc_scores['test_ap'] = sc_test_ap

    sc_scores['val_roc'] = sc_val_roc
    # sc_scores['val_roc_curve'] = sc_val_roc_curve
    sc_scores['val_ap'] = sc_val_ap

    sc_scores['runtime'] = runtime
    return sc_scores


def load_embeddings(filename):
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
        vectors[vec[0]] = np.array([float(x) for x in vec[1:]])
    fin.close()
    if HEADER:
        assert len(vectors) == node_num
    return vectors

# Input: NetworkX training graph, train_test_split (from mask_test_edges), n2v hyperparameters
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
# Whether to use bootstrapped edge embeddings + LogReg (like in node2vec paper), 
# or simple dot-product (like in GAE paper) for edge scoring
def embedding_scores(g_train, train_test_split, embedding_file, edge_score_mode = "edge-emb", verbose=1):
    if g_train.is_directed():
        DIRECTED = True

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split

    start_time = time.time()

    emb_mappings = load_embeddings(embedding_file)

    # Create node embeddings matrix (rows = nodes, columns = embedding features)
    emb_list = []
    for node_index in range(0, adj_train.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if edge_score_mode == "edge-emb":
        
        def get_edge_embeddings(edge_list):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(train_edges)
        neg_train_edge_embs = get_edge_embeddings(train_edges_false)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges)
            neg_val_edge_embs = get_edge_embeddings(val_edges_false)
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            emb_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # emb_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            emb_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            emb_val_roc = None
            emb_val_roc_curve = None
            emb_val_ap = None
        
        emb_test_roc = roc_auc_score(test_edge_labels, test_preds)
        # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        emb_test_ap = average_precision_score(test_edge_labels, test_preds)

    # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    elif edge_score_mode == "dot-product":
        score_matrix = np.dot(emb_matrix, emb_matrix.T)
        runtime = time.time() - start_time

        # Val set scores
        if len(val_edges) > 0:
            emb_val_roc, emb_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
        else:
            emb_val_roc = None
            emb_val_roc_curve = None
            emb_val_ap = None
        
        # Test set scores
        emb_test_roc, emb_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)

    else:
        print "Invalid edge_score_mode! Either use edge-emb or dot-product."

    # Record scores
    emb_scores = {}

    emb_scores['test_roc'] = emb_test_roc
    # emb_scores['test_roc_curve'] = emb_test_roc_curve
    emb_scores['test_ap'] = emb_test_ap

    emb_scores['val_roc'] = emb_val_roc
    # emb_scores['val_roc_curve'] = emb_val_roc_curve
    emb_scores['val_ap'] = emb_val_ap

    emb_scores['runtime'] = runtime

    return emb_scores

# Input: adjacency matrix (in sparse format), features_matrix (normal format), test_frac, val_frac, verbose
# Returns: Dictionary of results (ROC AUC, ROC Curve, AP, Runtime) for each link prediction method
def calculate_all_scores(adj_sparse, embedding_file, directed=False, \
        test_frac=.3, val_frac=.1, random_state=0, verbose=1, \
        train_test_split_file=None):
    np.random.seed(random_state)

    # Prepare LP scores dictionary
    lp_scores = {}

    ### ---------- PREPROCESSING ---------- ###
    train_test_split = None
    try:
        with open(train_test_split_file, 'rb') as f:
            train_test_split = pickle.load(f)
    except:
        print 'Generating train-test split...'
        if directed == False:
            train_test_split = mask_test_edges(adj_sparse, test_frac=test_frac, val_frac=val_frac, verbose=verbose)
        else:
            train_test_split = mask_test_edges_directed(adj_sparse, test_frac=test_frac, val_frac=val_frac, verbose=verbose)
    
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack tuple

    # g_train: new graph object with only non-hidden edges
    if directed == True:
        g_train = nx.DiGraph(adj_train)
    else:
        g_train = nx.Graph(adj_train)

    # Inspect train/test split
    if verbose >= 1:
        print "Total nodes:", adj_sparse.shape[0]
        print "Total edges:", int(adj_sparse.nnz/2) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
        print "Training edges (positive):", len(train_edges)
        print "Training edges (negative):", len(train_edges_false)
        print "Validation edges (positive):", len(val_edges)
        print "Validation edges (negative):", len(val_edges_false)
        print "Test edges (positive):", len(test_edges)
        print "Test edges (negative):", len(test_edges_false)
        print ''
        print "------------------------------------------------------"

    ### ---------- LINK PREDICTION BASELINES ---------- ###
    # Adamic-Adar
    aa_scores = adamic_adar_scores(g_train, train_test_split)
    lp_scores['aa'] = aa_scores
    if verbose >= 1:
        print ''
        print 'Adamic-Adar Test ROC score: ', str(aa_scores['test_roc'])
        print 'Adamic-Adar Test AP score: ', str(aa_scores['test_ap'])

    # Jaccard Coefficient
    jc_scores = jaccard_coefficient_scores(g_train, train_test_split)
    lp_scores['jc'] = jc_scores
    if verbose >= 1:
        print ''
        print 'Jaccard Coefficient Test ROC score: ', str(jc_scores['test_roc'])
        print 'Jaccard Coefficient Test AP score: ', str(jc_scores['test_ap'])

    # Preferential Attachment
    pa_scores = preferential_attachment_scores(g_train, train_test_split)
    lp_scores['pa'] = pa_scores
    if verbose >= 1:
        print ''
        print 'Preferential Attachment Test ROC score: ', str(pa_scores['test_roc'])
        print 'Preferential Attachment Test AP score: ', str(pa_scores['test_ap'])

    # Spectral Clustering
    sc_scores = spectral_clustering_scores(train_test_split)
    lp_scores['sc'] = sc_scores
    if verbose >= 1:
        print ''
        print 'Spectral Clustering Validation ROC score: ', str(sc_scores['val_roc'])
        print 'Spectral Clustering Validation AP score: ', str(sc_scores['val_ap'])
        print 'Spectral Clustering Test ROC score: ', str(sc_scores['test_roc'])
        print 'Spectral Clustering Test AP score: ', str(sc_scores['test_ap'])

    ### ---------- EDGE EMBEDDING ---------- ###
    embedding_edge_emb_scores = embedding_scores(g_train, train_test_split, embedding_file, "edge-emb", verbose)
    lp_scores['embedding_edge_emb'] = embedding_edge_emb_scores

    if verbose >= 1:
        print ''
        print 'Embedding (Edge Embeddings) Validation ROC score: ', str(embedding_edge_emb_scores['val_roc'])
        print 'Embedding (Edge Embeddings) Validation AP score: ', str(embedding_edge_emb_scores['val_ap'])
        print 'Embedding (Edge Embeddings) Test ROC score: ', str(embedding_edge_emb_scores['test_roc'])
        print 'Embedding (Edge Embeddings) Test AP score: ', str(embedding_edge_emb_scores['test_ap'])

    embedding_dot_product_scores = embedding_scores(g_train, train_test_split, embedding_file, "dot-product", verbose)
    lp_scores['embedding_dot_product'] = embedding_dot_product_scores
    if verbose >= 1:
        print ''
        print 'Embedding (Dot Product) Validation ROC score: ', str(embedding_dot_product_scores['val_roc'])
        print 'Embedding (Dot Product) Validation AP score: ', str(embedding_dot_product_scores['val_ap'])
        print 'Embedding (Dot Product) Test ROC score: ', str(embedding_dot_product_scores['test_roc'])
        print 'Embedding (Dot Product) Test AP score: ', str(embedding_dot_product_scores['test_ap'])
   
    return lp_scores

G = nx.read_edgelist(GRAPH_FILE)
A = nx.to_scipy_sparse_matrix(G)
calculate_all_scores(A, EMBEDDINGS_FILE)
