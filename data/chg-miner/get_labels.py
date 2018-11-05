import os
import sys
import snap

drugs_set = set()
genes_set = set()
def get_consistent_ids():
    with open('ChG-Miner_miner-chem-gene.tsv', 'r') as f:
        for line in f:
            nodes = [x.strip('\n') for x in line.split('\t')]
            drugs_set.add(nodes[0])
            genes_set.add(nodes[1])

def write_labels(drugs_set, genes_set, mapping, labels_file):
    with open(labels_file, 'wb+') as f:
        for drug in drugs_set:
            f.write(str(mapping[drug]) + ' ' + '0' + '\n')
        for gene in genes_set:
            f.write(str(mapping[gene]) + ' ' + '1' + '\n')

def write_mapping(drugs_set, genes_set, mapping, mapping_file):
    with open('ChG-Miner_miner-chem-gene.tsv', 'r') as f:
        with open(mapping_file, 'wb+') as g:
            for line in f:
                nodes = [x.strip('\n') for x in line.split('\t')]
                mapped_nodes = [mapping[i] for i in nodes]
                g.write(str(mapped_nodes[0]) + ' ' + str(mapped_nodes[1]) + '\n')


get_consistent_ids()

mapping = {}
c = 0
for i in drugs_set:
    mapping[i] = c
    c += 1
for i in genes_set:
    mapping[i] = c
    c += 1

write_labels(drugs_set, genes_set, mapping, 'chg-miner-labels.txt')
write_mapping(drugs_set, genes_set, mapping, 'chg-miner-graph.txt')
