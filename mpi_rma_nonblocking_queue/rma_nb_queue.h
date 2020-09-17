#pragma once

#include <mpi.h>

#define CODE_SUCCESS 0
#define CODE_ERROR 1
#define CODE_PARTLY_SUCCESS 2
#define CODE_DATA_BUFFER_FULL 3
#define CODE_DATA_BUFFER_EMPTY 4
#define CODE_QUEUE_EMPTY 5
#define CODE_NO_TAIL 6

#define UNDEFINED_RANK -1
#define MAIN_RANK 0

#define UNDEFINED_NODE_INFO -1LL // rank == -1, position == -1
#define UNDEFINED_TS -1.0

#define NODE_FREE 0
#define NODE_ACQUIRED 1
#define NODE_DELETED 2

typedef int val_t;

typedef union {
	struct {
		int rank;
		int position;
	} parsed;
	long long raw;
} u_node_info_t;

typedef struct {
	u_node_info_t head_info;
	u_node_info_t tail_info;
} queue_state_t;

typedef struct {
	u_node_info_t next_node_info;
	u_node_info_t info;
	val_t value;
	int state;
	double ts;
} elem_t;

typedef struct {
	MPI_Aint basedisp_local;	/* Base address of this struct (local) */
	MPI_Aint* basedisp;			/* Base address of this struct (all processes) */

	elem_t* data;				/* Physical buffer for queue elements */
	int data_ptr;
	int data_size;
	MPI_Aint datadisp_local;    /* Address of data buffer (local) */
	MPI_Aint* datadisp;         /* Address of data buffer (all processes) */

	queue_state_t state;		/* Contains info about head and tail */
	MPI_Aint statedisp_local;	/* Address of queue state struct (local) */
	MPI_Aint* statedisp;		/* Address of queue state struct (all processes) */

	elem_t sentinel;			/* Sentinel element (MAIN_RANK only) */
	MPI_Aint sentineldisp;		/* Address of sentinel in MAIN_RANK process */

	elem_t head;				/* Currently using head */
	MPI_Aint headdisp_local;	/* Address of head struct (local) */
	MPI_Aint* headdisp;			/* Address of head struct (all processes) */

	elem_t tail;				/* Currently using tail */
	MPI_Aint taildisp_local;	/* Address of tail struct (local) */
	MPI_Aint* taildisp;			/* Address of tail struct (all processes) */

	MPI_Win win;                /* RMA access window */
	MPI_Comm comm;              /* Communicator for the queue distribution */
	int n_proc;                 /* Number of processes in communicator */
	double ts_offset;           /* Timestamp offset from 0 process */
} rma_nb_queue_t;

typedef struct {
	int elem_value;
	int elem_state;
	int elem_next_node_info;
	int elem_info;
	int elem_ts;
	int qs_head;	// queue_state head
	int qs_tail;	// queue_state tail
} offsets_t;

typedef struct {
	int* nodes;
	int n_proc;
} rand_provider_t;

typedef struct bcast_meta_t {
	int (*bcast_method) (rma_nb_queue_t* queue, int target, elem_t* elem, bcast_meta_t* meta);

	bool should_update_head;
	bool should_update_tail;

	queue_state_t state;
	u_node_info_t head_info;
	u_node_info_t tail_info;
	elem_t head;
	elem_t tail;
} bcast_meta_t;


int rma_nb_queue_init(rma_nb_queue_t** queue, int size_per_node, MPI_Comm comm);
void rma_nb_queue_free(rma_nb_queue_t* queue);

int enqueue(rma_nb_queue_t* queue, val_t value);
int dequeue(rma_nb_queue_t* queue, val_t* value);
