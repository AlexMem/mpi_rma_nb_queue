#pragma once

#include <mpi.h>

#define CODE_SUCCESS 0
#define CODE_ERROR 1
#define CODE_DATA_BUFFER_FULL 2
#define CODE_DATA_BUFFER_EMPTY 3

#define UNKNOWN_RANK -1
#define MAIN_RANK 0

#define UNKNOWN_NEXT_NODE_INFO -1LL // rank == -1, position == -1

#define NODE_FREE 0
#define NODE_ACQUIRED 1
#define NODE_DELETED 2

typedef int val_t;

typedef union {
	struct {
		int rank;
		int position;
	} info;
	long long val;
} u_node_info_t;

typedef struct {
	u_node_info_t head;
	u_node_info_t tail;
} rma_nb_queue_state_t;

typedef struct {
	u_node_info_t next_info;
	val_t value;
	int state;
	double ts;
} elem_t;

typedef struct {
	MPI_Aint basedisp_local;    /* Base address of circbuf (local) */
	MPI_Aint* basedisp;         /* Base address of circbuf (all processes) */

	elem_t* data;                /* Physical buffer for queue elements */
	int data_ptr;
	int data_size;
	MPI_Aint datadisp_local;    /* Address of data buffer (local) */
	MPI_Aint* datadisp;         /* Address of data buffer (all processes) */

	MPI_Aint statedisp;			/* Address of queue state struct in process 0 */

	MPI_Win win;                /* RMA access window */
	MPI_Comm comm;              /* Communicator for the queue distribution */
	int nproc;                  /* Number of processes in communicator */
	double ts_offset;           /* Timestamp offset from 0 process */
} rma_nb_queue_t;

typedef struct {
	int elem_value_offset;
	int elem_state_offset;
	int elem_next_info_offset;
	int elem_ts_offset;
	int queue_state_tail_offset;
} offsets_t;



int rma_nb_queue_init(rma_nb_queue_t **queue, int size, MPI_Comm comm);

int enqueue(rma_nb_queue_t *queue, val_t value);

int dequeue(rma_nb_queue_t* queue, val_t *value);

void rma_nb_queue_free(rma_nb_queue_t* queue);