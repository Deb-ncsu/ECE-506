#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#ifdef OPENMP_HARNESS
#include <omp.h>
#endif

#ifdef MPI_HARNESS
#include <mpi.h>
#endif

#ifdef HYBRID_HARNESS
#include <omp.h>
#include <mpi.h>
#endif

#include "sort.h"
#include "graph.h"

struct Graph *countSortEdgesBySource (struct Graph *graph , int bit_to_shift, int radix)
{
 
    struct Edge *sorted_edges_array = newEdgeArray(graph->num_edges);
    int total_edges = graph->num_edges;
    int num_threads = omp_get_max_threads();
    int* vertex_count[num_threads];
    //numThreads = omp_get_max_threads();
    //printf ("num_threads %d",omp_get_max_threads());
    // auxiliary arrays, allocated at the start up of the program
    #pragma omp parallel 
    {
	int i;
	int num_elements = pow(2,radix);
	int bit_to_and = num_elements -1;
	int tid = omp_get_thread_num();
	int P = omp_get_num_threads();
	//printf ("total num threads %d \n", P);
	if (tid==0) {
		//int P = omp_get_num_threads();
		//int* vertex_count[P];
        	for (i = 0; i < P; i++) {
    			vertex_count[i] = (int *)malloc(num_elements * sizeof(int)); // needed for Counting Sort
		}
	}
	//printf("vertex array allocated by thread %d \n", tid);
	//printf("Printing %d  thread \n", vertex_count[tid][i], tid);
        #pragma omp barrier
	//printf("Printing num_elements %d  %d\n", num_elements, tid);
    	for(i = 0; i < num_elements; ++i)
    	{
        	vertex_count[tid][i] = 0;
		//printf ("vertex count for thread tid :: %d array location %d", tid , i);
    	}
	//printf("vertex array initialized by thread %d \n", tid);
	//printf ("total_edges %d :: thread %d \n ", total_edges, tid);
    	// count occurrence of key: id of a source vertex
	int start_node = (total_edges / P) * tid;
	int end_node;
	if (tid == P-1) {
		end_node = total_edges;
	} else {
		end_node = (total_edges / P) * (tid + 1);
	}
        int key;	
    	for(i = start_node; i < end_node; ++i)
    	{
        	key = (graph->sorted_edges_array[i].src >> bit_to_shift) & bit_to_and;
        	vertex_count[tid][key]++;
    	}
	//printf("vertex array updated by thread %d \n", tid);
	#pragma omp barrier
    	// transform to cumulative sum
	if (tid==0) {
    		for(i = 0; i < num_elements; ++i)
    		{
			int j;
			if (i!=0) {
				vertex_count[0][i] += vertex_count[P-1][i-1];
			}
			for (j=1; j< P; j++) {
        			vertex_count[j][i] += vertex_count[j-1][i];
			}
    		}
        	//printf("final count array created \n");
    	}
        #pragma omp barrier
	
    	// fill-in the sorted array of edges
    	//printf ("final placement by process %d \n", tid);
	int pos;
	for(i = end_node-1; i >= start_node; --i)
    	{
        	key = (graph->sorted_edges_array[i].src >> bit_to_shift) & bit_to_and;
        	pos = vertex_count[tid][key] - 1;
        	sorted_edges_array[pos] = graph->sorted_edges_array[i];
        	vertex_count[tid][key]--;
    	}
    
	//printf ("final placement done %d \n", tid);
	#pragma omp barrier
	free(vertex_count[tid]);
    }
    printf ("openmp finished");
    free(graph->sorted_edges_array);
    graph->sorted_edges_array = sorted_edges_array;
    //free(graph->sorted_edges_array);
    return graph;

}

#ifdef OPENMP_HARNESS
struct Graph *radixSortEdgesBySourceOpenMP (struct Graph *graph)
{
        int i;
        int bit_to_shift;
	int radix=8;
        int iter = 32/radix;
        struct Graph* new_graph;
        new_graph = graph;
        for (i=0;i<iter;i++) {
           bit_to_shift = i*radix;
           //printf("Count sort iteration %d \n", i);
           new_graph = countSortEdgesBySource(new_graph , bit_to_shift , radix);
           //printf(" Finishing iteration %d \n", i);
        }

        //printf("radix sort done");
        //printEdgeArray(graph->sorted_edges_array, graph->num_edges);
        //printf("print done");
        // printf("*** START Radix Sort Edges By Source OpenMP *** \n");

        return new_graph;

}
#endif

#ifdef MPI_HARNESS
struct Graph *radixSortEdgesBySourceMPI (struct Graph *graph)
{

    printf("*** START Radix Sort Edges By Source MPI*** \n");
    return graph;
}
#endif

#ifdef HYBRID_HARNESS
struct Graph *radixSortEdgesBySourceHybrid (struct Graph *graph)
{

    printf("*** START Radix Sort Edges By Source Hybrid*** \n");
    return graph;
}
#endif
