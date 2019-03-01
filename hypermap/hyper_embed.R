library(igraph)

load_graph <- function(filename) {
	edgelist <- read.table(filename, sep = "", header = F)
	return(edgelist)
}

convert_graph <- function(graph_df) {
	e <- c()
	for(i in 1:nrow(graph_df)) {
		row <- graph_df[i,]
		e <- c(e, row[[1]] + 1)
		e <- c(e, row[[2]] + 1)
	}
	return(graph(edges = e, n = max(graph_df, na.rm = T) + 1, directed = F))
}

embed_graph <- function(graph) {
	return(labne_hm(net = graph, gma = 2.3, Temp = 0.15, k.speedup = 10, w = 2*pi))
}