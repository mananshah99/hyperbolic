# code to combine network embeddings
import os
import sys
import numpy as np

EMBEDDING_FILES = ['../../graphsage/embeddings/graphsage_email-Eu-core.txt',
				   '../../poincare/embeddings/poincare_email_noburn.txt']

EMBEDDING_HEADERS = [False, False]

OUTPUT_FILE = '../../poincare/embeddings/graphsage_poincare_email.txt'

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
        vectors[vec[0]] = np.array([float(x) for x in vec[1:]])
    fin.close()
    if HEADER:
        assert len(vectors) == node_num
    return vectors

def save_embeddings(embedding_map, filename):
	f = open(filename, 'wb+')
	for node in embedding_map.keys():
		s = str(node) + ' '
		s += ' '.join([str(x) for x in embedding_map[node]])
		s += '\n'
		f.write(s)

# modes:
# supported - append
# to support - average, weighted (?)
def combine(embedding_files, embedding_headers, mode = 'append'):
	overall_map = {}
	for file, header in zip(embedding_files, embedding_headers):
		embedding_for_file = load_embeddings(file, header)
		if mode == 'append':
			for node in embedding_for_file:
				if node in overall_map:
					overall_map[node] = np.append(overall_map[node], embedding_for_file[node])
				else:
					overall_map[node] = embedding_for_file[node]
		else:
			print 'not supported'

	return overall_map

combined = combine(EMBEDDING_FILES, EMBEDDING_HEADERS)
save_embeddings(combined, OUTPUT_FILE)