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

/*struct Graph *countSortEdgesBySource (struct Graph *graph)
{

    int i;
    int key;
    int pos;
    struct Edge *sorted_edges_array = newEdgeArray(graph->num_edges);

    // auxiliary arrays, allocated at the start up of the program
    int *vertex_count = (int *)malloc(graph->num_vertices * sizeof(int)); // needed for Counting Sort

    for(i = 0; i < graph->num_vertices; ++i)
    {
        vertex_count[i] = 0;
    }

    // count occurrence of key: id of a source vertex
    for(i = 0; i < graph->num_edges; ++i)
    {
        key = graph->sorted_edges_array[i].src;
        vertex_count[key]++;
    }

    // transform to cumulative sum
    for(i = 1; i < graph->num_vertices; ++i)
    {
        vertex_count[i] += vertex_count[i - 1];
    }

    // fill-in the sorted array of edges
    for(i = graph->num_edges - 1; i >= 0; --i)
    {
        key = graph->sorted_edges_array[i].src;
        pos = vertex_count[key] - 1;
        sorted_edges_array[pos] = graph->sorted_edges_array[i];
        vertex_count[key]--;
    }



    free(vertex_count);
    free(graph->sorted_edges_array);

    graph->sorted_edges_array = sorted_edges_array;

    return graph;

} */


struct Graph* countSortEdgesBySource (struct Graph* graph, int bit_width , int radix, int world_rank, int world_size, int num_elements , int* new_array){
    
    /*MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);*/
    int i;
    int key;
    int pos;
    int bit_to_shift = bit_width;
    /* if (world_rank == 0) {
    	struct Edge *sorted_edges_array = newEdgeArray(graph->num_edges);
    } */
    //int vertices = graph->num_vertices / world_size;
    int vertices = pow(2,radix);
    unsigned int bit_to_and = vertices-1;
    int *vertex_count = (int*)malloc(vertices*sizeof(int)); // needed for Counting Sort
    //printf ("current_rank %d , world_size %d \n", world_rank , world_size);
    //printf ("Intializing the vertex_count array %d total vertices %d\n", world_rank , vertices);
    for(i = 0; i < vertices; ++i) {
        vertex_count[i] = 0;
    }
    //printf ("In process %d , vertex_count array created\n", world_rank); 
    // count occurrence of key: id of a source vertex
    /*int vertices_begin;
    vertices_begin = (graph->num_edges / world_size)*world_rank;
    int vertices_end;
    if (world_rank == world_size-1) {
    	vertices_end = graph->num_edges-1;
    } else {
	vertices_end = (graph->num_edges / world_size)*(world_rank+1) -1;
    } 
    //printf ("updating vertices by each process %d %d %d \n", world_rank, vertices_begin , vertices_end); 
    //printf ("debug %d" , graph->sorted_edges_array[0].src);*/
    for(i = 0; i < num_elements; ++i) {
        key =(new_array[i] >> bit_to_shift) & bit_to_and;
        //printf ("original key %d , actual_key %d \n", new_array[i] , key);
        vertex_count[key]++;
	//printf ("original key %d , actual_key %d", graph->sorted_edges_array[i].src , key);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //printf ("reached_barrier %d\n", world_rank);
    int *global_vertex_count = (int*)malloc(vertices*sizeof(int));
    MPI_Reduce(vertex_count, global_vertex_count, vertices, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // transform to cumulative sum
    if (world_rank==0) {
        //printf(" in final step of arranging %d \n", world_rank);
        struct Edge *sorted_edges_array = newEdgeArray(graph->num_edges);
    	for(i = 1; i < vertices; ++i) {
        	global_vertex_count[i] += global_vertex_count[i - 1];
    	}
    	for(i = graph->num_edges - 1; i >= 0; --i) {
        	key = (graph->sorted_edges_array[i].src >> bit_to_shift) & bit_to_and;
        	pos = global_vertex_count[key] - 1;
        	sorted_edges_array[pos] = graph->sorted_edges_array[i];
        	global_vertex_count[key]--;
    	}
       free(graph->sorted_edges_array);	
       graph->sorted_edges_array = sorted_edges_array;
       return graph;
    }

    free(vertex_count);
    if (world_rank==0) {
	return graph;
    } else {
	return NULL;
    }
    
}

#ifdef OPENMP_HARNESS
struct Graph *radixSortEdgesBySourceOpenMP (struct Graph *graph)
{

    printf("*** START Radix Sort Edges By Source OpenMP *** \n");
    return graph;
}
#endif

#ifdef MPI_HARNESS
struct Graph *radixSortEdgesBySourceMPI (struct Graph *graph , int world_rank , int world_size)
{
	//int num_vertices=graph->num_vertices;
	//printf("*** START Radix Sort Edges By Source MPI*** \n"); 
        int i;
        int bit_to_shift;
        struct Graph* new_graph;
        new_graph = graph;
        int radix=8;
        int k;
        int num_elements;
	int vertices_begin;
	int vertices_end;
	// send size of the graph to each node 
	int total_elements;
  	if (world_rank == 0) {
    		 total_elements = new_graph->num_edges;
    		 //printf("Process 0 broadcasting data %d\n", total_elements);
    		 MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
  	} else {
    		 MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
   		 //printf("Process %d received data %d from root process\n", world_rank, total_elements);
  	}

	// Each rank would need to have num_array to store a part of the vertices for its own count array
	//printf ("allocating new array for process %d \n", world_rank);
	if (world_rank!=world_size-1) {
             	num_elements = total_elements / world_size;
                //int *new_array = (int*)malloc(num_elements*sizeof(int));
                vertices_begin = (total_elements / world_size)*world_rank;
                vertices_end = (total_elements / world_size)*(world_rank+1) -1;
        } else {
                num_elements = (total_elements / world_size) + (total_elements % world_size);
                //int *new_array = (int*)malloc(num_elements*sizeof(int));
                vertices_begin = (total_elements / world_size)*world_rank;
                vertices_end = total_elements -1;
        }
	int *new_array = (int*)malloc(num_elements*sizeof(int));
        //printf ("new array allocated for the process :: %d", world_rank);
	///////////////////////////////////////////
        // Looping through the each radix 
	for (i=0;i<4;i++) {
        // root node need to send a block of vertices to other nodes
        if (world_rank==0) {
	//printf("in loop %d , trying to send a block to other nodes \n", i);
		int j;
		int num_elements;
		for (j=1; j<world_size; j++) {			
			if (j!=world_size-1) {
				num_elements = new_graph->num_edges / world_size;
				//int *new_array_to_send = (int*)malloc(num_elements*sizeof(int)); 				
				vertices_begin = (graph->num_edges / world_size)*j;
				vertices_end = (graph->num_edges / world_size)*(j+1) -1;
			} else {
				num_elements = (new_graph->num_edges / world_size) + (new_graph->num_edges % world_size);
				//int *new_array_to_send = (int*)malloc(num_elements*sizeof(int));
                                vertices_begin = (graph->num_edges / world_size)*j;
                                vertices_end = graph->num_edges -1;
				
			}
			int *new_array_to_send = (int*)malloc(num_elements*sizeof(int));
			for (k=vertices_begin; k<=vertices_end; k++ ) {
				new_array_to_send[k-vertices_begin] = graph->sorted_edges_array[k].src;
			}
			MPI_Send(new_array_to_send, num_elements, MPI_INT, j, 0, MPI_COMM_WORLD);
			//printf ("Data sent to node %d from node %d \n", j , world_rank);
			free(new_array_to_send);
		}
		vertices_begin = (graph->num_edges / world_size)*world_rank;
                vertices_end = (graph->num_edges / world_size)*(world_rank+1) -1;
                for (k=vertices_begin; k<=vertices_end; k++ ) {
                        new_array[k-vertices_begin] = graph->sorted_edges_array[k].src;
                }

		
	} else {
		//printf ("Waiting for the data %d \n", world_rank);
		MPI_Recv(new_array, num_elements , MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		//printf ("received data in the process %d \n", world_rank);		
	}
	int x;
	/*for (x=0; x<num_elements;x++) {
		printf("elements in process %d :: %d ", world_rank, new_array[x]);
	}*/
	///////////////
	// Except root node, setting new_graph to NULL since pointer to the new_graph does need to be passed
        if (world_rank!=0) {
                new_graph=NULL;
        }
	//////////////
        bit_to_shift = i*radix;
	//printf ("Calling countSortEdgesBySource %d\n", world_rank);
	new_graph = countSortEdgesBySource(new_graph , bit_to_shift, radix , world_rank , world_size , num_elements, new_array);
        MPI_Barrier(MPI_COMM_WORLD);

	}
			
        /*for (i=0;i<2;i++) {
           bit_to_shift = i*radix;
	   MPI_Barrier(MPI_COMM_WORLD);
	   if (world_rank!=0) {
		new_graph=NULL;
	   }
           new_graph = countSortEdgesBySource(new_graph , bit_to_shift, radix , world_rank , world_size);
           MPI_Barrier(MPI_COMM_WORLD);
	}*/
         if (world_rank==0) {
         	//printf("radix sort done \n");
         	//printEdgeArray(graph->sorted_edges_array, graph->num_edges);
         	//printf("print done \n");
         	//printf("*** START Radix Sort Edges By Source MPI *** \n");
         	return new_graph;
	}
}
#endif

#ifdef HYBRID_HARNESS
struct Graph *radixSortEdgesBySourceHybrid (struct Graph *graph)
{

    printf("*** START Radix Sort Edges By Source Hybrid*** \n");
    return graph;
}
#endif

/*struct Graph* radixSortEdgesBySourceMPI (struct Graph* graph){
        int i;
        int bit_to_shift;
        struct Graph* new_graph;
        new_graph = graph;
        for (i=0;i<2;i++) {
           bit_to_shift = i*16;
           //printf("Count sort iteration %d \n", i);
           new_graph = countSortEdgesBySource(new_graph , bit_to_shift);
           //printf(" Finishing iteration %d \n", i); 
         }
           //
         printf("radix sort done");
         printEdgeArray(graph->sorted_edges_array, graph->num_edges);
         printf("print done");
         printf("*** START Radix Sort Edges By Source OpenMP *** \n");
         return new_graph;                                                 
}*/

