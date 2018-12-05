import argparse
import snap
from bisect import bisect_left
from functools import reduce
from itertools import product
import math, random, sys
import networkx as nx
import numpy as np
import random
import math 

parser = argparse.ArgumentParser(description='Graph Generator')
parser.add_argument('type', type=str, help='Type of graph (see https://snap.stanford.edu/snappy/doc/reference/generators.html)')

args = parser.parse_args()

Graph = None
nodes = 1000

def full():
	G = snap.GenFull(snap.PUNGraph, nodes)
	snap.SaveEdgeList(G, args.type + '.txt')

def circle():
	G = snap.GenCircle(snap.PUNGraph, nodes, nodes/10)
	snap.SaveEdgeList(G, args.type + '.txt')

def prefattach():
	G = snap.GenPrefAttach(nodes, nodes/10, snap.TRnd())
	snap.SaveEdgeList(G, args.type + '.txt')

def dgmhierarchy():
	G = nx.dorogovtsev_goltsev_mendes_graph(6)
	nx.write_edgelist(G, args.type + '.txt')

def florentinefamilies():
	G = nx.florentine_families_graph()
	G = nx.convert_node_labels_to_integers(G)
	nx.write_edgelist(G, args.type + '.txt')

def __hyperbolic_distance(x1, y1, x2, y2):
	return np.arccosh(np.cosh(y1) * np.cosh(x2 - x1) * np.cosh(y2) - np.sinh(y1) * np.sinh(y2))

def __euclidean_distance(x1, y1, x2, y2):
	return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def waxman_graph(n, alpha=0.4, beta=0.1, L=None, domain=(0, 0, 10, 10), distance_fn = __euclidean_distance):
    # build graph of n nodes with random positions in the unit square
    G = nx.Graph()
    G.add_nodes_from(range(n))
    (xmin,ymin,xmax,ymax)=domain
    for n in G:
        G.node[n]['pos']=(xmin + ((xmax-xmin)*random.random()),
                          ymin + ((ymax-ymin)*random.random()))
    if L is None:
        # find maximum distance L between two nodes
        l = 0
        pos = list(nx.get_node_attributes(G,'pos').values())
        while pos:
            x1,y1 = pos.pop()
            for x2,y2 in pos:
            	#TODO: change this distance function
                r2 = distance_fn(x1, y1, x2, y2)**2
                if r2 > l:
                    l = r2
        l=math.sqrt(l)
    else:
        # user specified maximum distance
        l = L

    nodes_=list(G.nodes())
    if L is None:
        # Waxman-1 model
        # try all pairs, connect randomly based on euclidean distance
        while nodes_:
            u = nodes_.pop()
            x1,y1 = G.node[u]['pos']
            for v in nodes_:
                x2,y2 = G.node[v]['pos']
                #TODO: change this distance function
                r = distance_fn(x1, y1, x2, y2)
                if random.random() < alpha*math.exp(-r/(beta*l)):
                    G.add_edge(u,v)
    else:
        # Waxman-2 model
        # try all pairs, connect randomly based on randomly chosen l
        while nodes_:
            u = nodes_.pop()
            for v in nodes_:
                r = random.random()*l
                if random.random() < alpha*math.exp(-r/(beta*l)):
                    G.add_edge(u,v)
    return G

def waxman_euclidean():
	G = waxman_graph(nodes, distance_fn = __euclidean_distance)
	nx.write_edgelist(G, args.type + '.txt')

def waxman_hyperbolic():
	G = waxman_graph(nodes, distance_fn = __hyperbolic_distance)
	nx.write_edgelist(G, args.type + '.txt')

def geo_prefattach():
	Rnd = snap.TRnd();
	G = snap.GenGeoPrefAttach(100, 10, 0.25, Rnd)
	snap.SaveEdgeList(G, args.type + '.txt')

# MAIN

switcher = {
	"full" : full,
	"circle" : circle,
	"prefattach" : prefattach,
	"geo_prefattach" : geo_prefattach,
	"dgmhierarchy" : dgmhierarchy,
	"florentinefamilies" : florentinefamilies,
	"waxman_euclidean" : waxman_euclidean,
	"waxman_hyperbolic" : waxman_euclidean
}
switcher.get(args.type, None)()