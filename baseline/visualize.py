# run and then enter tensorboard --logdir=vis to visualize
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import os

LOG_DIR = 'vis'

edgelist = '../data/chg-miner/chg-miner-graph.txt' #'../data/email/email-Eu-core.txt'
labels_path    = '../data/chg-miner/chg-miner-labels.txt' #'../data/email/email-Eu-core-department-labels.txt'
embedding_path = 'embeddings/node2vec_chg_miner.txt'

labels = []
with open(labels_path, 'r') as f:
    for line in f:
        lbl = int(line.split(' ')[1])
        labels.append(lbl)

from openne.lap import LaplacianEigenmaps
from openne.graph import Graph

g = Graph()
g.read_edgelist(filename = edgelist, weighted=False, directed=True)
print('The loaded graph has # nodes = ' + str(g.node_size))

embeddings = np.zeros((g.G.number_of_nodes(), 128))

if embedding_path == '':
    print('No embedding path found')
    # use LaplacianEigenmaps
    model = LaplacianEigenmaps(g)
    vectors = model.vectors

    for i, embedding in vectors.items():
        embeddings[int(i), :] = embedding

else:
    # load from file
    f = open(embedding_path, 'r')
    lines = f.readlines()[1:]
    for l in lines:
        l = l.split(' ')
        node_id = l[0]
        node_embedding = [float(x) for x in l[1:]]
        embeddings[int(node_id), :] = node_embedding

# save embeddings and labels
emb_df = pd.DataFrame(embeddings)
emb_df.to_csv(LOG_DIR + '/embeddings.tsv', sep='\t', header=False, index=False)

lab_df = pd.Series(labels, name='label')
lab_df.to_frame().to_csv(LOG_DIR + '/node_labels.tsv', index=False, header=False)

# save tf variable
embeddings_var = tf.Variable(embeddings, name='embeddings')
sess = tf.Session()

saver = tf.train.Saver([embeddings_var])
sess.run(embeddings_var.initializer)
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)

# configure tf projector
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = 'embeddings'
embedding.metadata_path = 'node_labels.tsv'

projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
