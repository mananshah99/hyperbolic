//##############################################################################
//         Copyright (C) 2014 David Coudert <david.coudert@inria.fr>
//
//     Distributed under the terms of the GNU General Public License (GPL)
//                        http://www.gnu.org/licenses/
//##############################################################################

/*

  NOTE:
  - Connectivity: We assume that the input graph is connected and preferably
    bi-connected
  - The graph must be stored in the edgelist format. We assume that vertex
    labels are integers.


  BUG FIX:

  - Aurélien Lancin (Oct. 2014): fix consistency error with openmp. The bounds
    were not propagated properly among the threads and so the final result was
    sometimes incorrect.

 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <stdint.h>    // to use uint16_t, uint32_t, uint64_t, etc.
#include <inttypes.h>  // to use: uint64_t t; printf("%" PRIu64 "\n", t);
#include <omp.h>       // to use openmp

// To control memory usage
#define uintN_t uint32_t // already graphs with up to 4G nodes
#define uintD_t uint16_t // assume diameter at most 2^16-1. Change if needed


/*******************************************************************************/
/* Utility functions                                                           */
/*******************************************************************************/

#define _MAX_(_x_,_y_) ((_x_)>(_y_)?(_x_):(_y_))
#define _MIN_(_x_,_y_) ((_x_)<(_y_)?(_x_):(_y_))
#define _ITR_(_x_,_y_) ((_x_ - 1) * (_y_ - 2))
#define _ITR2_(_x_) ((_x_ - 1) * _x_)
#define MAX_LINE_LENGTH 1000


// Raise an error and exit the program
void raise_error(char *message){
  fprintf(stderr,"%s\n",message);
  exit(1);
}

// Raise an error if ptr is NULL
void test_ptr(void *ptr, char *message){
  if (ptr == NULL )
    raise_error(message);
}

// Return the average of the values in tab
float _avg_(float *tab, int size){
  float s = 0;
  int i;
  if(size<1) raise_error("_avg_: not enough values");
  for(i=0;i<size;i++) s += tab[i];
  return(s/size);
}


// Shuffle array of type uintNt*
void _shuffle_array_uintN_t(uintN_t *tab, uintN_t size){
  uintN_t i,j,tmp;
  for(i=size-1;i>0;i--){
    j = rand()%(i+1);
    if(i!=j){
      tmp = tab[i];
      tab[i] = tab[j];
      tab[j] = tmp;
    }
  }
}



// print some data
void print_result(double time0, double time1, float delta){
  double time = time1 - time0;
  fprintf(stdout, "delta = [%.1f]\n", delta);
  fprintf(stdout, "Computation time: [%dh %dm %ds]\t[%.3fs]\n",
	  (int) time / 3600, ((int) time % 3600) / 60, ((int) time % 3600) % 60, time);
}


/*******************************************************************************/
/* Pairs tools                                                                 */
/*******************************************************************************/

// type pair_t: pair of vertices with distance
typedef struct _pair_t_ {
  uintN_t x;
  uintN_t y;
  uintD_t d;
  struct _pair_t_ *next; // to make lists
} pair_t;

pair_t* pair_new(uintN_t x, uintN_t y, uintD_t value, pair_t *next){
  pair_t *p;
  test_ptr( (p=(pair_t*)malloc(sizeof(pair_t))), "pair_new: malloc() error");
  p->x = _MIN_(x,y);
  p->y = _MAX_(x,y);
  p->d = value;
  p->next = next;
  return p;
}

void pair_free_list(pair_t *p){
  pair_t *q;
  while(p) {
    q = p;
    p = p->next;
    free(q);
  }
}


// Comparison function for qsort(). 
// We want decreasing order of distances and increasing lexicographic order when
// same distance
static int pairs_compare_distance(void const *a, void const *b){
  int res;
  struct _pair_t_ const *pa = a;
  struct _pair_t_ const *pb = b;
  
  res = pb->d - pa->d;
  if (res == 0)
    {
      res = pa->x - pb->x;
      if (res == 0)
	{
	  res = pa->y - pb->y;
	}
    }
  return res;
}

// sort pairs by decreasing distances
// void pair_array_sort_distance(pair_t *p, uint64_t nb_pairs){
//   qsort(p, nb_pairs, sizeof(pair_t), pairs_compare_distance);
// }
#define pair_array_sort_distance(_p_, _nb_pairs_) qsort(_p_, _nb_pairs_, sizeof(pair_t), pairs_compare_distance)


/*******************************************************************************/
/* Graph tools                                                                 */
/*******************************************************************************/

typedef struct _node_t_ {
  uintN_t id;             // index of the node in [0..N-1]
  uintN_t true_id;        // id of the node in input file, assumed < 2^sizeof(intN_t)
  struct _node_t_ *next;  // to make a list
} node_t;

typedef struct _edge_t_ {
  uintN_t u;              // end node with smallest index
  uintN_t v;              // end node with largest index
  struct _edge_t_ *next;
} edge_t;

typedef struct _graph_t_ {
  uintN_t n;                    // number of nodes
  uintN_t m;                    // number of edges
  node_t *nodes;                // list of nodes
  edge_t *edges;                // list of edges
  uintN_t *degree;              // degree of the vertices. Array index by nodes id
  edge_t ***incident_edges;     // arrays of incident edges of nodes
  node_t **int_to_nodes;        // mapping node id to node
  uintN_t *true_id_to_int;      // mapping id in input file to ids in [0..n-1]
  uintN_t _size_true_id_to_int; // largest index in input file
} graph_t;



graph_t* graph_new(){
  graph_t *g;

  test_ptr( (g = (graph_t*)malloc(sizeof(graph_t))), "graph_new: malloc() error");
  g->n = 0;
  g->m = 0;
  g->nodes = NULL;
  g->edges = NULL;
  g->degree = NULL;
  g->int_to_nodes = NULL;
  g->true_id_to_int = NULL;
  g->_size_true_id_to_int = 0;
  g->incident_edges = NULL;
  return(g);
}


void graph_free(graph_t *g){
  edge_t *e;
  node_t *u;
  uintN_t i;
  if (g!=NULL) {
    while(g->edges!=NULL) {
      e = g->edges;
      g->edges = e->next;
      free(e);
    }
    while(g->nodes!=NULL) {
      u = g->nodes;
      g->nodes = u->next;
      free(u);
    }
    if (g->int_to_nodes!=NULL)
      free(g->int_to_nodes);
    if(g->true_id_to_int!=NULL)
      free(g->true_id_to_int);
    if (g->incident_edges!=NULL){
      for(i=0;i<g->n;i++)
	if (g->incident_edges[i])
	  free(g->incident_edges[i]);
      free(g->incident_edges);
    }
    if (g->degree)
      free(g->degree);
    free(g);
  }
}


// add node to graph if not already in
void _graph_add_node_(graph_t *g, uintN_t u){
  node_t *v;

  if(g->true_id_to_int[u]==(uintN_t)-1){
    test_ptr( (v = (node_t*)malloc(sizeof(node_t))), "_graph_add_node_: malloc() error");
    v->true_id = u;
    v->id = g->n;
    g->true_id_to_int[u] = v->id;
    g->n += 1;
    v->next = g->nodes;
    g->nodes = v;
  }
}



// return true if the graph has edge u,v
int graph_has_edge(graph_t *g, uintN_t u, uintN_t v) {
  edge_t *e;
  uintN_t a = _MIN_(u,v);
  uintN_t b = _MAX_(u,v);
  
  e = g->edges;
  while(e){
    if( (e->u==a)&&(e->v==b) )
      return 1;
    e = e->next;
  }
  return 0;
}

// add edge to the graph. Multiple edges are allowed here
void graph_add_edge(graph_t *g, uintN_t u, uintN_t v){
  edge_t *e;

  test_ptr( (e = (edge_t*)malloc(sizeof(edge_t))), "graph_add_edge: malloc() error");
  e->u = _MIN_(u,v);
  e->v = _MAX_(u,v);
  e->next = g->edges;
  g->edges = e;
  g->m += 1;
}


// consolidate the graph from lists of edges
graph_t* graph_consolidate(pair_t *p){
  edge_t *e;
  node_t *v;
  uintN_t i, *tmp;
  uintN_t true_id_max = 0;
  pair_t *q;
  graph_t *g = graph_new();

  // We create a list of nodes and mappings  
  // we first search for the largest index
  q = p;
  while(q){
    true_id_max = _MAX_(true_id_max, q->y);
    q = q->next;
  }
  // we prepare mapping true_id_to_int
  test_ptr( (g->true_id_to_int = (uintN_t*)malloc(sizeof(uintN_t)*(true_id_max+1))), "graph_create_mapping: malloc() error");
  memset(g->true_id_to_int, (char)-1, sizeof(uintN_t)*(true_id_max+1));
  g->_size_true_id_to_int = true_id_max+1;
  // we create the list of nodes
  q = p;
  while(q) {
    _graph_add_node_(g, q->x);
    _graph_add_node_(g, q->y);
    q = q->next;
  }
  // we now make the mapping int_to_nodes
  test_ptr( (g->int_to_nodes = (node_t**)malloc(sizeof(node_t*)*g->n)), "graph_create_mapping: malloc() error");
  v = g->nodes;
  while(v){
    g->int_to_nodes[v->id] = v;
    v = v->next;
  }

  // We build set of edges with end nodes in 0..N-1
  q = p;
  while(q) {
    graph_add_edge(g, g->true_id_to_int[q->x], g->true_id_to_int[q->y]); 
    q = q-> next;
  }

  // We compute degree of nodes
  test_ptr( (g->degree=(uintN_t*)malloc(sizeof(uintN_t)*g->n)), "graph_consolidate: malloc() error");
  for(i=0;i<g->n;i++)
    g->degree[i] = 0;
  e = g->edges;
  while(e) {
    g->degree[e->u] += 1;
    g->degree[e->v] += 1;
    e = e->next;
  }

  // We construct list of incident edges
  test_ptr( (tmp = (uintN_t*)malloc(sizeof(uintN_t)*g->n)), "graph_consolidate: malloc() error");
  memset(tmp, (char)0, sizeof(uintN_t)*g->n);
  test_ptr( (g->incident_edges = (edge_t***)malloc(sizeof(edge_t**)*g->n)), "graph_consolidate: malloc() error");
  for(i=0;i<g->n;i++)
    test_ptr((g->incident_edges[i] = (edge_t**)malloc(sizeof(edge_t*)*g->degree[i])), "graph_consolidate: malloc() error");

  e = g->edges;
  while(e) {
    g->incident_edges[e->u][tmp[e->u]] = e;
    g->incident_edges[e->v][tmp[e->v]] = e;
    tmp[e->u] += 1;
    tmp[e->v] += 1;
    e = e->next;
  }

  free(tmp);
  return g;
}


// Read edgelist format
graph_t *graph_read_edgelist(FILE *f, char verbose){
  char line[MAX_LINE_LENGTH], *ll;
  uintN_t u, v;
  pair_t *p = NULL;
  graph_t *g;
  double t = omp_get_wtime();

  if(verbose) fprintf(stderr,"read graph\n");
  // read data from file
  while (fgets(line, MAX_LINE_LENGTH, f)){
    // remove useless spaces or tabs at the beginning of the line
    ll = line;
    while((ll[0]==' ')||(ll[0]=='\t')) ll = ll+1;

    switch (ll[0]){
    case '#': // read comments and do nothing
      break;
    case 'c': // read comments and do nothing
      break;
    case '\n': // read empty line and do nothing
      break;
    default: // read edge {u,v}
      if (sscanf(ll, "%u %u\n", &u, &v) != 2)    // try with space delimiter
	if (sscanf(ll, "%u\t%u\n", &u, &v) != 2) // try with tab delimiter
	  {fprintf(stderr,"Last line: %s",line);
	    raise_error("graph_read_edgelist: sscanf error while reading <u> <v>");}
      if(u!=v)
	p = pair_new(_MIN_(u,v), _MAX_(u,v), 0, p);
    }
  }

  g = graph_consolidate(p);

  pair_free_list(p);
  if(verbose) fprintf(stderr,"end read\t[%.4fs]\n",omp_get_wtime()-t);
  return g;
}


void graph_print_edgelist(graph_t *g, FILE *f, int true_id){
  edge_t *e;
  uintN_t a,b;
  if(true_id)
    for(e=g->edges;e!=NULL;e=e->next){
      a = g->int_to_nodes[e->u]->true_id;
      b = g->int_to_nodes[e->v]->true_id;
      fprintf(f,"%u %u\n", _MIN_(a,b), _MAX_(a,b));
    }
  else
    for(e=g->edges;e!=NULL;e=e->next)
      fprintf(f,"%u %u\n",e->u,e->v);
}


void graph_test(FILE *f){
  graph_t *g;
  g = graph_read_edgelist(f, 1);
  fprintf(stdout,"N = %d\tM = %d\n",g->n, g->m);
  graph_print_edgelist(g,stdout,1);
  graph_print_edgelist(g,stdout,0);
  graph_free(g);
}





/*******************************************************************************/
/* Distances                                                                   */
/*******************************************************************************/

void distances_and_far_apart_pairs(graph_t *g, uintD_t **distances, char **far_apart_pairs, char verbose){
  uintN_t *seen;
  uintN_t *waiting_list;
  uintN_t waiting_beginning, waiting_end;
  uintN_t source, u, v, i;
  uintD_t *distance_s;
  double t = omp_get_wtime();

  if(verbose) fprintf(stderr,"start distances_and_far_apart_pairs\n");

  if(!g) raise_error("distances_and_far_apart_pairs: input graph is NULL");
  if(g->n==0) raise_error("distances_and_far_apart_pairs: empty graph");
  
  test_ptr( (seen=(uintN_t*)malloc(sizeof(uintN_t)*g->n)), "distances_and_far_apart_pairs: malloc() error");
  test_ptr( (waiting_list=(uintN_t*)malloc(sizeof(uintN_t)*g->n)), "distances_and_far_apart_pairs: malloc() error");

  memset(seen, (char)-1, sizeof(uintN_t)*g->n);

  for(i=0;i<g->n;i++) {
    memset(far_apart_pairs[i], (char)-1, sizeof(char)*g->n);
    far_apart_pairs[i][i] = 0;
  }

  for(source=0;source<g->n;source++) {

    // the source is seen
    seen[source] = source;

    // Its parameters can already be set
    distance_s = distances[source];
    memset(distance_s, (char)-1, sizeof(uintD_t)*g->n);
    distance_s[source] = 0;

    // and added to the queue
    waiting_list[0] = source;
    waiting_beginning = 0;
    waiting_end = 0;

    // For as long as there are vertices left to explore
    while(waiting_beginning <= waiting_end) {
      
      // We pick the first one
      v = waiting_list[waiting_beginning];

      // Iterating over all the outneighbors u of v
      for(i=0;i<g->degree[v];i++) {
	u = g->incident_edges[v][i]->u;
	if(u==v) u = g->incident_edges[v][i]->v;

	// If we notice one of these neighbors is not seen yet, we set
	// its parameters and add it to the queue to be explored later.
	if (seen[u]!=source) {
	  distance_s[u] = distance_s[v]+1;
	  // v is on the path from source to u
	  far_apart_pairs[source][v] = 0;
	  far_apart_pairs[v][source] = 0;
	  seen[u] = source;
	  waiting_end += 1;
	  waiting_list[waiting_end] = u;
	}
	else if(distance_s[u] == distance_s[v]+1) {
	  // v is on the path from source to u
	  far_apart_pairs[source][v] = 0;
	  far_apart_pairs[v][source] = 0;
	}
      }
      waiting_beginning += 1;
    }
  }

  free(seen);
  free(waiting_list);
  if(verbose) fprintf(stderr,"end _distances\t[%.4fs]\n",omp_get_wtime()-t);
}


// Display some statistics on distances and far_apart pairs
void distances_and_far_apart_pairs_distr(graph_t *g, uintD_t **distances, char **far_apart_pairs) {
  uintN_t *cpt, *cpt_FA;
  uintN_t i,j;

  test_ptr( (cpt = (uintN_t*)malloc(2*sizeof(uintN_t)*g->n)), "distances_and_far_apart_pairs_distr: malloc() error");
  memset(cpt, (char)0, 2*sizeof(uintN_t)*g->n);
  cpt_FA = cpt+g->n;

  for(i=0;i<g->n;i++)
    for(j=i+1;j<g->n;j++){
      cpt[distances[i][j]] += 1;
      if(far_apart_pairs[i][j])
	cpt_FA[distances[i][j]] += 1;
    }

  fprintf(stdout,"Distances distribution: ");
  for(i=0;i<g->n;i++)
    if(cpt[i])
      fprintf(stdout,"(%d, %d) ",i,cpt[i]);

  fprintf(stdout,"\nDistant pairs distribution: ");
  for(i=0;i<g->n;i++)
    if(cpt_FA[i])
      fprintf(stdout,"(%d, %d) ",i,cpt_FA[i]);
  fprintf(stdout,"\n");

  free(cpt);
}


// Compute Distances from source using BFS
void single_source_distances_BFS(graph_t *g, uintN_t source, uintN_t *distance_s,
			     char *seen, uintN_t *waiting_list) {
  uintN_t waiting_beginning, waiting_end;
  uintN_t u, v, i;

  memset(seen, 0, g->n);
  memset(distance_s, (char)-1, sizeof(uintN_t)*g->n);

  // the source is seen
  seen[source] = 1;
  distance_s[source] = 0;

  // and added to the queue
  waiting_list[0] = source;
  waiting_beginning = 0;
  waiting_end = 0;

  // For as long as there are vertices left to explore
  while(waiting_beginning <= waiting_end) {
      
    // We pick the first one
    v = waiting_list[waiting_beginning];

    // Iterating over all the outneighbors u of v
    for(i=0;i<g->degree[v];i++) {
      u = g->incident_edges[v][i]->u;
      if(u==v) u = g->incident_edges[v][i]->v;

      // If we notice one of these neighbors is not seen yet, we set
      // its parameters and add it to the queue to be explored later.
      if (!seen[u]) {
	distance_s[u] = distance_s[v]+1;
	// v is on the path source to u
	seen[u] = 1;
	waiting_end += 1;
	waiting_list[waiting_end] = u;
      }
    }
    waiting_beginning += 1;
  }
}

// Return 1 if the graph is connected and 0 otherwise
int graph_is_connected(graph_t *g){
  uintN_t *waiting_list, *distances;
  char *seen;
  uintN_t i,res;

  test_ptr( (distances = (uintN_t*)malloc(g->n*sizeof(uintN_t))), "graph_is_connected: malloc error");
  test_ptr( (waiting_list = (uintN_t*)malloc(g->n*sizeof(uintN_t))), "graph_is_connected: malloc error");
  test_ptr( (seen = (char*)malloc(g->n*sizeof(char))), "graph_is_connected: malloc error");

  single_source_distances_BFS(g, 0, distances, seen, waiting_list);
  for(i=0,res=1;i<g->n;i++)
    if(distances[i]>g->n){
      res = 0;
      break;
    }
  free(seen);
  free(distances);
  free(waiting_list);
  return res;
}


/*******************************************************************************/
/* Data for hyperbolicity algorithms                                           */
/*******************************************************************************/

// type data_t: all necessary data
typedef struct _data_t_ {
  uintN_t n;         // number of nodes
  uint64_t nb_pairs; // number of distant pairs <= n(n-1)/2
  pair_t *pair;      // array of nb_pairs pairs
  uintD_t **dist;    // distance matrix
} data_t;


data_t* data_new(uintN_t n, uint64_t nb_pairs, uintD_t **distances) {
  data_t *data;
  test_ptr((data = (data_t*) malloc(sizeof(data_t))), "data_new: malloc error");
  data->n = n;
  data->nb_pairs = nb_pairs;
  test_ptr((data->pair = (pair_t*) malloc(sizeof(pair_t)*nb_pairs)), "data_new: malloc error");
  data->dist = distances;
  return data;
}

void data_free(data_t *data) {
  free(data->pair);
  free(data->dist[0]);
  free(data->dist);
  free(data);
}


// build list of far-apart pairs sorted by decreasing distances
data_t* data_make(graph_t *g, int verbose) {
  data_t *data;
  uintN_t i,j;
  uint64_t k;
  uintD_t **distances;
  char **far_apart_pairs;

  test_ptr( (distances = (uintD_t**)malloc(g->n*sizeof(uintD_t*))), "data_make: malloc error");
  test_ptr( (distances[0] = (uintD_t*)malloc(g->n*g->n*sizeof(uintD_t))), "data_make: malloc error");
  test_ptr( (far_apart_pairs = (char**)malloc(g->n*sizeof(char*))), "data_make: malloc error");
  test_ptr( (far_apart_pairs[0] = (char*)malloc(g->n*g->n*sizeof(char))), "data_make: malloc error");
  for(i=1;i<g->n;i++) {
    distances[i] = distances[i-1]+g->n;
    far_apart_pairs[i] = far_apart_pairs[i-1]+g->n;
  }

  distances_and_far_apart_pairs(g, distances, far_apart_pairs, verbose);
  if(verbose)
    distances_and_far_apart_pairs_distr(g, distances, far_apart_pairs);

  for(i=0,k=0;i<g->n;i++)
    for(j=i+1;j<g->n;j++)
      if(far_apart_pairs[i][j]) 
	k += 1;

  data = data_new(g->n, k, distances);

  for(i=0,k=0;i<g->n;i++)
    for(j=i+1;j<g->n;j++)
      if(far_apart_pairs[i][j]) {
	data->pair[k].x = i;
	data->pair[k].y = j;
	data->pair[k].d = data->dist[i][j];
	k += 1;
      }

  pair_array_sort_distance(data->pair, data->nb_pairs);

  free(far_apart_pairs[0]);
  free(far_apart_pairs);
  return data;
}

// Print data.
void print_data(data_t *data, graph_t *g){
  int i;
  fprintf(stdout, "BEGIN\nN\t%d\n", data->n);
  fprintf(stdout, "number of pairs: [%" PRIu64 "]\n", data->nb_pairs);
  for (i = 0; i < data->nb_pairs; i++)
    fprintf(stdout, "%d %d %d\n", g->int_to_nodes[data->pair[i].x]->true_id, g->int_to_nodes[data->pair[i].y]->true_id, data->pair[i].d);
  fprintf(stdout, "END\n");
}




/*******************************************************************************/
/* Hyperbolicity                                                               */
/*******************************************************************************/

// Naive algorithm in O(n^4). No parallelism. Readable. Very slow.
float hyperbolicity_naive_single_thread(data_t *data, graph_t *g)
{
  pair_t *pair = data->pair;
  pair_t pi, pj;
  uintD_t **dist = data->dist;
  uintD_t *distx, *disty;
  uintD_t delta = 0;
  int32_t S2, S3, tmp, delta_dxy, z, r, y;
  int64_t i = 0, j = 0, nb_pairs;
  
  nb_pairs = data->nb_pairs;

  for (i = 1; i < nb_pairs; i++) {
    pi = *(pair+i);

    y = pi.y;
    distx = dist[pi.x];
    disty = dist[y];
    delta_dxy = delta - distx[y];

    for (j = 0; j < i; j++) {
      pj = *(pair+j);
      z = pj.x;
      r = pj.y;
      S2 = distx[z] + disty[r];
      S3 = distx[r] + disty[z];
      tmp = pj.d - _MAX_( S2, S3 );
	  
      if (tmp > delta_dxy) {
	delta = tmp + distx[y];

#ifdef MORE
	if (S2 >= S3) {
	  fprintf(stdout, "LB: [%.1f]\t[%u,%u,%u,%u]\t[%u,%u,%u,%u,%u,%u]\t4-tuples pos: [%" PRIu64 "]\n",
		  delta / 2.0,
		  g->int_to_nodes[pi.x]->true_id, g->int_to_nodes[pi.y]->true_id,
		  g->int_to_nodes[z]->true_id, g->int_to_nodes[r]->true_id,
		  pi.d, pj.d, distx[z], disty[r], distx[r], disty[z], (_ITR2_(i) / 2) + j + 1);
	}
	else {
	  fprintf(stdout, "LB: [%.1f]\t[%u,%u,%u,%u]\t[%u,%u,%u,%u,%u,%u]\t4-tuples pos: [%" PRIu64 "]\n",
		  delta / 2.0,
		  g->int_to_nodes[pi.x]->true_id, g->int_to_nodes[pi.y]->true_id,
		  g->int_to_nodes[z]->true_id, g->int_to_nodes[r]->true_id,
		  pi.d, pj.d, distx[z], disty[r], distx[r], disty[z], (_ITR2_(i) / 2) + j + 1);
	}
#endif

	if (delta < pi.d) {
	  delta_dxy = tmp;
	}
      }
    }
  }
  
  fprintf(stdout, "Visited 4-tuples: [%" PRIu64 "]\n", _ITR2_(data->nb_pairs) / 2);

  return (delta / 2.0);
}


// Naive algorithm in O(n^4) with parallelism. Readable. Very slow.
float hyperbolicity_naive_with_openmp(data_t *data, graph_t *g)
{
  pair_t *pair = data->pair;
  pair_t pi, pj;
  uintD_t **dist = data->dist;
  uintD_t *distx, *disty;
  uintD_t delta = 0;
  int32_t S2, S3, tmp, delta_dxy, z, r, y;
  int64_t i = 0, j = 0, nb_pairs;

  nb_pairs = data->nb_pairs;

#pragma omp parallel shared(delta, dist) firstprivate (nb_pairs,pair,i,j) private(delta_dxy,y,pi,distx,disty,r,z,S2,S3,tmp,pj)
  {
#pragma omp for nowait
    for (i = 1; i < nb_pairs; i++) {
      pi = *(pair+i);
	
      y = pi.y;
      distx = dist[pi.x];
      disty = dist[y];
      delta_dxy = delta - distx[y];
	
      for (j = 0; j < i; j++) {
	pj = *(pair+j);
	z = pj.x;
	r = pj.y;
	S2 = distx[z] + disty[r];
	S3 = distx[r] + disty[z];
	tmp = pj.d - _MAX_( S2, S3 );
	    
	if (tmp > delta_dxy) {
#pragma omp critical
	  {
	    delta = tmp + distx[y];
		  
#ifdef MORE
	    if (S2 >= S3) {
	      fprintf(stdout, "LB: [%.1f]\t[%u,%u,%u,%u]\t[%u,%u,%u,%u,%u,%u]\t4-tuples pos: [%" PRIu64 "]\n",
		      delta / 2.0,
		      g->int_to_nodes[pi.x]->true_id, g->int_to_nodes[pi.y]->true_id,
		      g->int_to_nodes[z]->true_id, g->int_to_nodes[r]->true_id,
		      pi.d, pj.d, distx[z], disty[r], distx[r], disty[z], (_ITR2_(i) / 2) + j + 1);
	    }
	    else
	      {
		fprintf(stdout, "LB: [%.1f]\t[%u,%u,%u,%u]\t[%u,%u,%u,%u,%u,%u]\t4-tuples pos: [%" PRIu64 "]\n",
			delta / 2.0,
			g->int_to_nodes[pi.x]->true_id, g->int_to_nodes[pi.y]->true_id,
			g->int_to_nodes[z]->true_id, g->int_to_nodes[r]->true_id,
			pi.d, pj.d, distx[z], disty[r], distx[r], disty[z], (_ITR2_(i) / 2) + j + 1);
	      }
#endif
	    
	    if (delta < pi.d) {
	      delta_dxy = tmp;
	    }

#pragma omp flush(delta)
	  }
	}
      }
    }
  }

  fprintf(stdout, "Visited 4-tuples: [%" PRIu64 "]\n", _ITR2_(data->nb_pairs) / 2);

  return (delta / 2.0);
}



// Optimized version of the implementation of the algorithm. No parallelism.
// Hard to read due to local optimizations.
float hyperbolicity_single_thread(data_t *data, graph_t *g) {
#ifdef COUNT
  int64_t cpt = 0;
#endif

  pair_t *pair = data->pair;
  pair_t pi, pj;
  uint16_t **dist = data->dist;
  uintD_t *distx, *disty;
  uintD_t delta = 0, UB = (uintD_t) -1;
  int32_t S2, S3, tmp, delta_dxy, z, r, y;
  int64_t i = 0, j = 0, nb_pairs, last_i = 0, retains = 0;
  
  nb_pairs = data->nb_pairs;
  
  for (i = 1; i < nb_pairs; i++) {
    pi = *(pair+i);
      
    if (pi.d < UB) {
      UB = pi.d;
	  
#ifdef MORE
      fprintf(stdout, "UB: [%.1f]\t4-tuples pos: [%" PRIu64 "]\n", UB / 2.0, (_ITR2_(i) / 2));
#endif
    }
      
    if (delta >= pi.d) {
    
      // weird observation, using i = nb_pairs as end condition for the first
      // loop and break for the second is sometimes more efficient than using
      // two breaks O_o
      last_i = i;
      i = nb_pairs;
      //break;
    }
    else {
      y = pi.y;
      distx = dist[pi.x];
      disty = dist[y];
      delta_dxy = delta - distx[y];
	  
      for (j = 0; j < i; j++) {
#ifdef COUNT
	cpt++;
#endif
	      
	pj = *(pair+j);
	z = pj.x;
	r = pj.y;
	S2 = distx[z] + disty[r];
	S3 = distx[r] + disty[z];
	tmp = pj.d - _MAX_( S2, S3 );
	      
	if (tmp > delta_dxy) {
	  delta = tmp + distx[y];
	  
#ifdef MORE
	  if (S2 >= S3)	{
	    fprintf(stdout, "LB: [%.1f]\t[%u,%u,%u,%u]\t[%u,%u,%u,%u,%u,%u]\t4-tuples pos: [%" PRIu64 "]\n",
		    delta / 2.0,
		    g->int_to_nodes[pi.x]->true_id, g->int_to_nodes[pi.y]->true_id,
		    g->int_to_nodes[z]->true_id, g->int_to_nodes[r]->true_id,
		    pi.d, pj.d, distx[z], disty[r], distx[r], disty[z], (_ITR2_(i) / 2) + j + 1);
	  }
	  else {
	    fprintf(stdout, "LB: [%.1f]\t[%u,%u,%u,%u]\t[%u,%u,%u,%u,%u,%u]\t4-tuples pos: [%" PRIu64 "]\n",
		    delta / 2.0,
		    g->int_to_nodes[pi.x]->true_id, g->int_to_nodes[pi.y]->true_id,
		    g->int_to_nodes[z]->true_id, g->int_to_nodes[r]->true_id,
		    pi.d, pj.d, distx[z], disty[r], distx[r], disty[z], (_ITR2_(i) / 2) + j + 1);
	  }
#endif

	  if (delta >= pj.d) {
	    retains = i - j - 1;
	    break;
	  }
	  else {
	    delta_dxy = tmp;
	  }
	}
      }
    }
  }

#ifdef COUNT
  fprintf(stdout, "Visited 4-tuples: [%" PRIu64 "]\tcpt: [%" PRIu64 "]\n", (_ITR2_(last_i) / 2) - retains, cpt);
#else
  fprintf(stdout, "Visited 4-tuples: [%" PRIu64 "]\n",  (_ITR2_(last_i) / 2) - retains);
#endif

  return (delta / 2.0);
}


float hyperbolicity_with_openmp(data_t *data, graph_t *g){
#ifdef COUNT
  uint64_t cpt = 0;
  uint64_t cpt_total = 0;
#endif

  pair_t *pair = data->pair;
  pair_t pi, pj;
  uintD_t **dist = data->dist;
  uintD_t *distx, *disty;
  uintD_t delta = 0, UB = (uintD_t) -1;
  uint32_t S2, S3, z, r, dxy;
  uint64_t nb_pairs;
  int64_t i = 0, j = 0;
  int32_t tmp;
  
  nb_pairs = data->nb_pairs;

#ifdef COUNT
#pragma omp parallel shared(delta,dist) firstprivate (pair, nb_pairs,cpt,UB) private(i,j,dxy,r,z,S2,S3,tmp,pj,pi,distx,disty)
#else
#pragma omp parallel shared(delta,dist) firstprivate (pair, nb_pairs,UB) private(i,j,dxy,r,z,S2,S3,tmp,pj,pi,distx,disty)
#endif
  {
#pragma omp for schedule(dynamic)nowait
    // Iterate over the contiguous arrays of pairs sorted by decreasing distance orders
    for (i = 1; i < nb_pairs; i++) {

      pi = *(pair + i);

      // Print each time the current distance between pairs decreased by
      // one. This distance is an upper bound of the hyperbolicity.
      if (pi.d < UB) {
	UB = pi.d;
#ifdef MORE
	fprintf(stdout, "UB: [%.1f]\t4-tuples pos: [%" PRIu64 "] %" PRIu64 "\n", UB / 2.0, (_ITR2_(i) / 2), i);
#endif
      }

      // If the lower bound is equal or larger than the current distance between
      // the pair, then the lower bound cannot increase and the hyperbolicity is
      // found. The loop is broken by setting i to the last value. Note that the
      // break primitive cannot be used because of the omp.
      if (delta >= pi.d) {	
	i = nb_pairs;
      }
      else {
	// Get distances from the first pair of vertices x,y.
	distx = dist[pi.x];
	disty = dist[pi.y];
	//delta_dxy = delta - distx[pi.y];
	dxy = pi.d;

	for (j = 0; j < i; j++) {

// Fine grained iteration counter, normally the loop indexes i,j are enough to
// get the number of iterations done.
#ifdef MORE
	  cpt++;
#endif

	  // Get distances from the second pair of vertices z,r.
	  pj = *(pair + j);
	  z = pj.x;
	  r = pj.y;
	  S2 = distx[z] + disty[r];
	  S3 = distx[r] + disty[z];

	  tmp = dxy + pj.d - _MAX_( S2, S3 );
	  // Using this tmp and delta_dxy allow to save the cost of an operation.
	  // tmp = pj.d - _MAX_( S2, S3 );

	  // If a new value of the lower bound is found.
	  if (tmp > delta) {
	    // To be executed by only one thread at a time.
#pragma omp critical
	    {
	      // Compare tmp and delta_dxy again because delta_dxy
	      // could have been changed by another thread after
	      // the comparison outside the critical section/
	      if (tmp > delta) {
		// Compute the new value of the hyperbolicity.
		//delta = tmp + distx[pi.y];
		delta = tmp;
#pragma omp flush(delta)
		/* If the new value lower bound is equal or larger than the
		   current distance between the pairs of the second loop, the
		   lower bound cannot increase, break the loop and start again
		   with pairs with at larger distances than delta. */
		if (delta >= pj.d) {
		  j = i;
		}

		/* Print the new lower bound, its quadruples and associated
		   distances and its position in the iteration. */
#ifdef MORE
		if (S2 >= S3) {
		  fprintf(stdout, "LB: [%.1f]\t[%u,%u,%u,%u]\t[%u,%u,%u,%u,%u,%u]\t4-tuples pos: [%" PRIu64 "]\n",
			  delta / 2.0,
			  g->int_to_nodes[pi.x]->true_id, g->int_to_nodes[pi.y]->true_id,
			  g->int_to_nodes[z]->true_id, g->int_to_nodes[r]->true_id,
			  pi.d, pj.d, distx[z], disty[r], distx[r], disty[z], (_ITR2_(i) / 2) + j + 1);
		}
		else {
		  fprintf(stdout, "LB: [%.1f]\t[%u,%u,%u,%u]\t[%u,%u,%u,%u,%u,%u]\t4-tuples pos: [%" PRIu64 "]\n",
			  delta / 2.0,
			  g->int_to_nodes[pi.x]->true_id, g->int_to_nodes[pi.y]->true_id,
			  g->int_to_nodes[z]->true_id, g->int_to_nodes[r]->true_id,
			  pi.d, pj.d, distx[z], disty[r], distx[r], disty[z], (_ITR2_(i) / 2) + j + 1);
		}
#endif
	      }		
	    }
	  }
	}
      }
    }

// Sum the number of iteration done by each thread.
#ifdef COUNT
#pragma omp barrier
#pragma omp critical
    {
      cpt_total += cpt;
    }
#endif
  }

// Print the total number of iterations done by all the thread. Print -1 if the
// option was not compiled.
#ifdef COUNT
  fprintf(stdout, "Visited 4-tuples: [%" PRIu64 "]\n", cpt_total);
#endif

  return (delta / 2.0);
}




/*******************************************************************************/
/* Heuristics                                                                 */
/*******************************************************************************/

// Return 2*( max(S_1, S_2, S3) - max2( S_1, S_2, S3) )
uintN_t _delta_from_Si(uintN_t S1, uintN_t S2, uintN_t S3){
  uintN_t deltas;
  if(S1>S2) {
    if(S2>S3) deltas = S1-S2;
    else if(S1>S3) deltas = S1-S3;
    else deltas = S3-S1;
  }
  else {
    if(S1>S3) deltas = S2-S1;
    else if(S2>S3) deltas = S2-S3;
    else deltas = S3-S2;
  }
  return deltas;
}



// Selects randomly 3 vertices and compute delta(a,b,c,x) for all x in the graph.
// Repeat trials time or until max_time exceeded
float heuristic_random(graph_t *g, uint64_t trials, double max_time, char verbose){
  uintN_t a, b, c, delta, i, k, N, tmp;
  uintN_t *da, *db, *dc, *waiting_list;
  char *seen;
  double t0 = omp_get_wtime();

  N = g->n;
  test_ptr( (da = (uintN_t*)malloc(N*sizeof(uintN_t))), "heuristic_random: malloc error");
  test_ptr( (db = (uintN_t*)malloc(N*sizeof(uintN_t))), "heuristic_random: malloc error");
  test_ptr( (dc = (uintN_t*)malloc(N*sizeof(uintN_t))), "heuristic_random: malloc error");
  test_ptr( (waiting_list = (uintN_t*)malloc(N*sizeof(uintN_t))), "heuristic_random: malloc error");
  test_ptr( (seen = (char*)malloc(N*sizeof(char))), "heuristic_random: malloc error");

  srand(time(NULL));
  delta = 0;
  for(k=0;(k<trials)&&((omp_get_wtime()-t0)<max_time);k++){
    a = rand() % N;
    b = rand() % N;
    c = rand() % N;
    single_source_distances_BFS(g, a, da, seen, waiting_list);
    if((2*da[b]>delta)&&(2*da[c]>delta)){
      single_source_distances_BFS(g, b, db, seen, waiting_list);
      if((2*db[c]>delta)&&(_MAX_(da[b],_MAX_(da[c],db[c]))>delta)){
	single_source_distances_BFS(g, c, dc, seen, waiting_list);
    
	for(i=0;i<N;i++){
	  tmp = _delta_from_Si(da[b]+dc[i], da[c]+db[i], da[i]+db[c]);

	  if(tmp>delta){
	    delta = tmp;
	    if(verbose)
	      fprintf(stdout,"[%.1f] with (%d, %d, %d, %d) at time %.6f\n",
		      delta/2.0,g->int_to_nodes[a]->true_id,
		      g->int_to_nodes[b]->true_id,
		      g->int_to_nodes[c]->true_id,
		      g->int_to_nodes[i]->true_id,
		      omp_get_wtime()-t0);
	  }
	}
      }
    }
  }

  free(da);
  free(db);
  free(dc);
  free(waiting_list);
  free(seen);
  return (delta/2.0);
}

/* Starting from a random vertex, finds a pair (a,b) of distant vertices using
   k-sweep like approach (at least one BFS). Then selects a subset of vertices
   at same distant from a and b (within a ball), and compute delta(a,b,c,d) for
   all c,d in this subset. Repeat trials time and return best result. The search
   space is pruned whenever possible.
*/
float heuristic_CCL(graph_t *g, int ball, uint8_t ksweep, uintN_t max_mid_size,
		    uint64_t trials, double max_time, char verbose){
  uintN_t a=0, b, c, delta, i, j, k, x, N, tmp;
  uintN_t *da, *db, *dc, *mid, *waiting_list;
  uintN_t mid_size;
  char *seen;
  double t0 = omp_get_wtime();

  N = g->n;
  test_ptr( (da = (uintN_t*)malloc(N*sizeof(uintN_t))), "heuristic_CCL: malloc error");
  test_ptr( (db = (uintN_t*)malloc(N*sizeof(uintN_t))), "heuristic_CCL: malloc error");
  test_ptr( (dc = (uintN_t*)malloc(N*sizeof(uintN_t))), "heuristic_CCL: malloc error");
  test_ptr( (mid = (uintN_t*)malloc(N*sizeof(uintN_t))), "heuristic_CCL: malloc error");
  test_ptr( (waiting_list = (uintN_t*)malloc(N*sizeof(uintN_t))), "heuristic_CCL: malloc error");
  test_ptr( (seen = (char*)malloc(N*sizeof(char))), "heuristic_CCL: malloc error");

  ksweep = _MAX_(1, ksweep);
  srand(time(NULL));
  delta = 0;
  for(k=0;(k<trials)&&((omp_get_wtime()-t0)<max_time);k++){
    // choose random vertex a and make BFS from a
    b = rand() % N;
    for(i=0;i<ksweep;i++){
      a = b;
      // select vertex b at largest distance from a
      single_source_distances_BFS(g, a, da, seen, waiting_list);
      // Note: the last visited vertex of the BFS is at maximum distance (node
      // waiting_list[N-1]). However we get better results this way:
      b = a;
      for(i=0;i<N;i++)
	if(da[i]>da[b])
	b = i;
    }	

    if(2*da[b]>delta){
      single_source_distances_BFS(g, b, db, seen, waiting_list);

      // compute set of vertices at equal distance from a and b
      for(i=0,mid_size=0;i<N;i++)
	if(abs(da[i]-db[i])<=ball){
	  mid[mid_size] = i;
	  mid_size++;
	}
      //we shuffle mid
      _shuffle_array_uintN_t(mid, mid_size);

      // compute delta(a,b,c,d) for all c,d in mid
      for(i=0;i<_MIN_(mid_size,max_mid_size);i++){
	c = mid[i];

	if((2*da[c]>delta)&&(2*db[c]>delta)&&(_MAX_(da[b],_MAX_(da[c],db[c]))>delta)){
	  single_source_distances_BFS(g, c, dc, seen, waiting_list);

	  for(j=0;j<mid_size;j++){
	    x = mid[j];
	    tmp = _delta_from_Si(da[b]+dc[x], da[c]+db[x], da[x]+db[c]);

	    if(tmp>delta){
	      delta = tmp;
	      if(verbose)
		fprintf(stdout,"[%.1f] with (%d, %d, %d, %d) at time %.6f\n",
			delta/2.0,g->int_to_nodes[a]->true_id,
			g->int_to_nodes[b]->true_id,
			g->int_to_nodes[c]->true_id,
			g->int_to_nodes[x]->true_id,
			omp_get_wtime()-t0);
	    }
	  }
	}
      }
    }
  }
  if(verbose) fprintf(stdout,"k = %d\n",k);
  free(da);
  free(db);
  free(dc);
  free(mid);
  free(waiting_list);
  free(seen);
  return (delta/2.0);
}


// This function runs iter times all heuristics and display statistics
void multiple_heuristics(graph_t *g, char verbose, 
			 int max_time, uint64_t trials, int iter,
			 uint8_t ksweep, int ball, uintN_t SaSb){
  int i;
  float *delta_CCL, *delta_rand, *time_CCL;
  float begin, end;

  test_ptr( (delta_CCL = (float*)malloc(3*iter*sizeof(float))), "multiple_heuristics: malloc error");
  test_ptr( (time_CCL = (float*)malloc(iter*sizeof(float))), "multiple_heuristics: malloc error");
  test_ptr( (delta_rand = (float*)malloc(iter*sizeof(float))), "multiple_heuristics: malloc error");

  for(i=0;i<iter;i++){
    // Launch Heuristic CCL
    if(verbose) fprintf(stdout,"Heuristic CCL\n");
    begin = omp_get_wtime();
    delta_CCL[i] = heuristic_CCL(g, ball, ksweep, SaSb, trials, max_time, verbose);
    end = omp_get_wtime();
    if(verbose) print_result(begin, end, delta_CCL[i]);
    time_CCL[i] = end-begin;

    // Launch Heuristic random with same computation time than Heuristic CCL
    if(verbose) fprintf(stdout,"Heuristic random\n");
    begin = omp_get_wtime();
    delta_rand[i] = heuristic_random(g, trials*100, time_CCL[i], verbose);
    end = omp_get_wtime();
    if(verbose) print_result(begin, end, delta_rand[i]);
  }

  // Print some statistics
  fprintf(stdout,"\t CCL\t Rand\t Time\n");
  for(i=0;i<iter;i++)
    fprintf(stdout,"%d\t %.1f\t %.1f\t %.4f\n",i,delta_CCL[i], delta_rand[i],time_CCL[i]);
  fprintf(stdout,"Avg.\t[%.1f]\t[%.1f]\t[%.4f]\n",_avg_(delta_CCL,iter), _avg_(delta_rand,iter), _avg_(time_CCL,iter));
      
  free(delta_CCL);
  free(delta_rand);
  free(time_CCL);
}



/*******************************************************************************/
/* Main functions                                                              */
/*******************************************************************************/

void usage(char *name){
  fprintf(stderr, "Usage:\n\t%s algo=10 ball=1 ksweep=2 s=100 trials=50 iter=50 file=./roadNet-CA-lbcc.edgelist -verbose\n", name);
  fprintf(stderr,"\t%s algo=1 file=./as20000102-lbcc.edgelist\n",name);
  fprintf(stderr, "\nAlgorithms: (algo)\n");
  fprintf(stderr,"\t0: exact, single-thread\n");
  fprintf(stderr,"\t1: exact, multi-threads << default\n");
  fprintf(stderr,"\t2: exact, O(n^4) algorithm, single-thread\n");
  fprintf(stderr,"\t3: exact, O(n^4) algorithm, multi-threads\n");
  fprintf(stderr,"\t4: heuristic random. Uses <trials>\n");
  fprintf(stderr,"\t5: heuristic CCL. Uses <ball>, <ksweep>, <s>, and <trials>\n");
  fprintf(stderr,"\t10: heuristic CCL + random. Use also <iter>\n");
  fprintf(stderr,"\nParameters:\n");
  fprintf(stderr,"\talgo: algorithm to use\n");
  fprintf(stderr,"\tfile: name of the file containing the graph\n");
  fprintf(stderr,"\ttrials: number of trials of the heuristics\n");
  fprintf(stderr,"\ttime: maximum allowed running time for the heuristics\n");
  fprintf(stderr,"\tksweep: (in [0..255]) number of BFS until selection of a distant pair (a,b)\n");
  fprintf(stderr,"\tball: (in [0..255]) extra gap for the selection of vertices at mid distance of a and b\n");
  fprintf(stderr,"\ts: max number of selected vertices at mid distance of a and b\n");
  fprintf(stderr,"\titer: number of times to repeat each heuristic\n");
  fprintf(stderr,"\t-verbose: to display some informations during execution of the algorithm\n");
  fprintf(stderr,"\nCompilation options:\n");
  fprintf(stderr,"  -fopenmp:  to enable openmp. Faster for large graphs.\n");
  fprintf(stderr,"\n  Example: gcc hyp.c -O2 -fopenmp -o hyp\n\n");
  exit(-1);
}


// Parse input parameters
void parse_input_parameters(int argc, char **argv,
			    int *algo, char *filename, char *verbose, 
			    int *max_time, int *trials, int *iter,
			    int *ksweep, int *ball, int *SaSb){
  int i;

  fprintf(stdout,"# %s", argv[0]);
  for(i=1;i<argc;i++)
    fprintf(stdout," %s",argv[i]);
  fprintf(stdout,"\n");

  if (argc <= 1) usage(argv[0]);
  else{
      for(i=1;i<argc;i++){
	switch(argv[i][0]){
	case 'a':
	  if(!(sscanf(argv[i], "algo=%d", algo) == 1)) usage(argv[0]);
	  break;
	case 'f':
	  if(!(sscanf(argv[i], "file=%s", filename) == 1)) usage(argv[0]);
	  break;
	case '-': 
	  if(strcmp(argv[i],"-verbose")==0) *verbose = 1;
	  break;
	case 't':
	  if(!(sscanf(argv[i], "time=%d", max_time) == 1))
	    if(!(sscanf(argv[i], "trials=%d", trials) == 1))
	      usage(argv[0]);
	  break;
	case 'i':
	  if(!(sscanf(argv[i], "iter=%d", iter) == 1)) usage(argv[0]);
	  break;
	case 'k':
	  if(!(sscanf(argv[i], "ksweep=%d", ksweep) == 1)) usage(argv[0]);
	  if( (*ksweep<0) || (*ksweep>(uint8_t)-1) ) usage(argv[0]);
	  break;
	case 'b':
	  if(!(sscanf(argv[i], "ball=%d", ball) == 1)) usage(argv[0]);
	  if( (*ball<0) || (*ball>(uint8_t)-1) ) usage(argv[0]);
	  break;
	case 's':
	  if(!(sscanf(argv[i], "s=%d", SaSb) == 1)) usage(argv[0]);
	  break;
	default:
	  usage(argv[0]);
	}
      }
    }
}




int main(int argc, char **argv)
{
  graph_t *g;
  data_t *data = NULL;
  float delta = 0;
  double begin, end;
  FILE *f;             // input file
  int algo = 1;        // algorithm
  char verbose = 0;    // to display some information during processing
  char filename[1000]; // Input file name
  int max_time = 600;  // Heur: max allowed computation time
  int trials = 50;     // Heur: max number of trials
  int iter = 1;        // Heur: max number of execution of Heuristics (for statistics)
  int ksweep = 2;      // Heur CCL: number of sweeps to select first pair
  int ball = 0;        // Heur CCL: ball for the selection of Sa and Sb
  int SaSb = 100;      // Heur CCL: number of selected nodes in Sa \cap Sb

  filename[0]=-1;

  parse_input_parameters(argc, argv, &algo, filename, &verbose,
			 &max_time, &trials, &iter,
			 &ksweep, &ball, &SaSb);

  // Read the graph
  if(filename[0]==-1)
    g = graph_read_edgelist(stdin, verbose);
  else{
    test_ptr((f = fopen(filename, "r")), "Error when reading input file");
    g = graph_read_edgelist(f, verbose);
    fclose(f);
  }

  // Raise error if the input graph is not connected
  if(!graph_is_connected(g))
    raise_error("The input graph is not connected");

  fprintf(stdout,"N = %d\nM = %d\n",g->n, g->m);

  if(algo<5)
    data = data_make(g, verbose);
    

  fprintf(stdout, "Start computation with algorithm %d...\n", algo);
  begin = omp_get_wtime();
  switch (algo)
    {
    case 0: // Optimized implementation on a single thread 
      delta = hyperbolicity_single_thread(data, g);
      break;
    case 1: // Optimized implementation with multiple threads (using openmp)
      delta = hyperbolicity_with_openmp(data, g);
      break;
    case 2: // Naive O(n^4) algorithm on a single thread
      delta = hyperbolicity_naive_single_thread(data, g);
      break;
    case 3: // Naive O(n^4) algorithm with multiple threads (using openmp)
      delta = hyperbolicity_naive_with_openmp(data, g);
      break;
    case 4: // Heuristic random
      delta = heuristic_random(g, trials, max_time, verbose);
      break;
    case 5: // Heuristic CCL
      delta = heuristic_CCL(g, ball, ksweep, SaSb, trials, max_time, verbose);
      break;
    case 10: // Multiple iterations of heuristics with statistics
      multiple_heuristics(g, verbose, max_time, trials, iter, ksweep, ball, SaSb);
      break;
    }
  end = omp_get_wtime();
  if(algo<10)
    print_result(begin, end, delta);

  graph_free(g);
  if(data) data_free(data);
  return (0);
}

