import networkx as nx

G = nx.read_edgelist('karate.edgelist', nodetype=int, create_using=nx.DiGraph())

for node in G.nodes():
	print(node)

