#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#include "rma_nb_queue.h"
//#include "utils.h"

#define USE_DEBUG 0

void print(elem_t elem);

int myrank;
rma_nb_queue_state_t queue_state;
offsets_t offsets;

void g_pause() {
	if (myrank == MAIN_RANK) {
		system("pause");
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

double get_timestamp(void) {
	return MPI_Wtime();
}

void error_msg(const char* msg, int _errno) {
	fprintf(stderr, "%d \t %s", myrank, msg);
	if (_errno != 0)
		fprintf(stderr, ": %s", strerror(_errno));
	fprintf(stderr, "\n");
}

void offsets_init(void) {
	offsets.elem_value_offset = offsetof(elem_t, value);
	offsets.elem_state_offset = offsetof(elem_t, state);
	offsets.elem_next_info_offset = offsetof(elem_t, next_info);
	offsets.elem_ts_offset = offsetof(elem_t, ts);

	offsets.queue_state_tail_offset = offsetof(rma_nb_queue_state_t, tail);
}

int disps_init(rma_nb_queue_t *queue) {
	MPI_Alloc_mem(sizeof(MPI_Aint) * queue->nproc, MPI_INFO_NULL, &queue->basedisp);
	MPI_Alloc_mem(sizeof(MPI_Aint) * queue->nproc, MPI_INFO_NULL, &queue->datadisp);
	if ((queue->basedisp == NULL) || (queue->datadisp == NULL)) {
		error_msg("basedisp or datadisp memory allocation failed", errno);
		return CODE_ERROR;
	}

	MPI_Allgather(&queue->basedisp_local, 1, MPI_AINT, queue->basedisp, 1, MPI_AINT, queue->comm);
	MPI_Allgather(&queue->datadisp_local, 1, MPI_AINT, queue->datadisp, 1, MPI_AINT, queue->comm);
	MPI_Bcast(&queue->statedisp, 1, MPI_AINT, MAIN_RANK, queue->comm);

	return CODE_SUCCESS;
}

int rma_nb_queue_init(rma_nb_queue_t** queue, int size, MPI_Comm comm) {
	MPI_Alloc_mem(sizeof(rma_nb_queue_t), MPI_INFO_NULL, queue);
	if (*queue == NULL) {
		error_msg("queue memory allocation failed", errno);
		return CODE_ERROR;
	}

	(*queue)->comm = comm;
	MPI_Comm_size(comm, &(*queue)->nproc);
	MPI_Get_address(*queue, &(*queue)->basedisp_local);

	int data_buffer_size = sizeof(elem_t) * size;
	MPI_Alloc_mem(data_buffer_size, MPI_INFO_NULL, &(*queue)->data);
	if ((*queue)->data == NULL) {
		error_msg("data buffer memory allocation failed", errno);
		return CODE_ERROR;
	}
	memset((*queue)->data, 0, data_buffer_size);
	MPI_Get_address((*queue)->data, &(*queue)->datadisp_local);

	(*queue)->data_ptr = 0;
	(*queue)->data_size = size;

	MPI_Win_create_dynamic(MPI_INFO_NULL, (*queue)->comm, &(*queue)->win);
	MPI_Win_attach((*queue)->win, *queue, sizeof(rma_nb_queue_t));
	MPI_Win_attach((*queue)->win, (*queue)->data, data_buffer_size);
	queue_state.head.val = UNKNOWN_NEXT_NODE_INFO;
	queue_state.tail.val = UNKNOWN_NEXT_NODE_INFO;
	if (myrank == MAIN_RANK) {

		MPI_Get_address(&queue_state, &(*queue)->statedisp);
		MPI_Win_attach((*queue)->win, &queue_state, sizeof(rma_nb_queue_state_t));
	}

	if (disps_init(*queue) != CODE_SUCCESS) {
		error_msg("displacements initialization failed", 0);
		return CODE_ERROR;
	}

	offsets_init();

	//(*queue)->ts_offset = mpi_sync_time(comm);

	return CODE_SUCCESS;
}

int get_position(rma_nb_queue_t *queue, int *position) {
	int start = queue->data_ptr;
	while (1) {
		if (queue->data[queue->data_ptr].state == NODE_FREE) {
			*position = queue->data_ptr;
			return CODE_SUCCESS;
		}

		queue->data_ptr = (queue->data_ptr + 1) % queue->data_size;
		if (queue->data_ptr == start) {
			return CODE_DATA_BUFFER_FULL;
		}
	}
}

void begin_epoch_one(int rank, MPI_Win win) {
	MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win);
}

void end_epoch_one(int rank, MPI_Win win) {
	MPI_Win_unlock(rank, win);
}

void begin_epoch_all(MPI_Win win) {
	MPI_Win_lock_all(0, win);
}

void end_epoch_all(MPI_Win win) {
	MPI_Win_unlock_all(win);
}

int get_queue_state(rma_nb_queue_t *queue, rma_nb_queue_state_t *queue_state) {
	int op_res;
	op_res = MPI_Get(queue_state, sizeof(rma_nb_queue_state_t), MPI_BYTE,
					 MAIN_RANK, queue->statedisp, sizeof(rma_nb_queue_state_t), MPI_BYTE, queue->win);
	MPI_Win_flush(MAIN_RANK, queue->win);
	if (USE_DEBUG) std::cout << "rank " << myrank << ": got queue state head(" << queue_state->head.info.rank << ", " << queue_state->head.info.position << ") tail(" << queue_state->tail.info.rank << ", " << queue_state->tail.info.position << ")\t" << op_res << std::endl;
	return op_res;
}

int enqueue(rma_nb_queue_t *queue, val_t value) {
	int pos;
	int op_res;
	int i_result;
	bool success;
	u_node_info_t this_node_info;
	u_node_info_t unknown_node_info;
	u_node_info_t result;
	elem_t elem;

	begin_epoch_all(queue->win);

	int rc = get_position(queue, &pos);
	if (rc == CODE_DATA_BUFFER_FULL) {
		end_epoch_all(queue->win);
		//error_msg("data buffer full", CODE_DATA_BUFFER_FULL);
		return CODE_DATA_BUFFER_FULL;
	}

	queue->data[pos].value = value;
	queue->data[pos].state = NODE_ACQUIRED;
	queue->data[pos].next_info.val = UNKNOWN_NEXT_NODE_INFO;

	this_node_info.info.rank = myrank;
	this_node_info.info.position = pos;

	unknown_node_info.val = UNKNOWN_NEXT_NODE_INFO;

	result.val = -2LL;

	MPI_Win_sync(queue->win);

	do {
		//g_pause();
		get_queue_state(queue, &queue_state);
		
		queue->data[pos].ts = get_timestamp() + queue->ts_offset;
		MPI_Win_sync(queue->win);

		if (queue_state.head.info.rank == UNKNOWN_RANK) {
			// queue is empty
			// update head
			op_res = MPI_Compare_and_swap(&this_node_info, &unknown_node_info, &result, MPI_LONG_LONG,
								 MAIN_RANK, queue->statedisp, queue->win);
			MPI_Win_flush(MAIN_RANK, queue->win);
			
			if (USE_DEBUG) std::cout << "rank " << myrank << ": insert cas 1 result (" << result.info.rank << ", " << result.info.position << "; " << this_node_info.info.rank << ", " << this_node_info.info.position << ")\t" << op_res << std::endl;

			if (result.val != unknown_node_info.val) {
				continue;
			}

			// update tail
			MPI_Put(&this_node_info, sizeof(u_node_info_t), MPI_BYTE, MAIN_RANK,
					MPI_Aint_add(queue->statedisp, offsets.queue_state_tail_offset), sizeof(u_node_info_t), MPI_BYTE, queue->win);
			MPI_Win_flush(MAIN_RANK, queue->win);
		} else {
			// queue is not empty
			// try to set link from head elem to this elem
			op_res = MPI_Compare_and_swap(&this_node_info, &unknown_node_info, &result, MPI_LONG_LONG, queue_state.head.info.rank,
								 MPI_Aint_add(queue->datadisp[queue_state.head.info.rank], sizeof(elem_t) * queue_state.head.info.position),
								 queue->win);
			MPI_Win_flush(queue_state.head.info.rank, queue->win);
			
			if (USE_DEBUG) std::cout << "rank " << myrank << ": insert cas 2 result (" << result.info.rank << ", " << result.info.position << "; " << this_node_info.info.rank << ", " << this_node_info.info.position << ")\t" << op_res << std::endl;

			success = (result.val == unknown_node_info.val);
			if (!success) {
				continue;
			}
			// get previous elem
			op_res = MPI_Get(&elem, sizeof(elem_t), MPI_BYTE, queue_state.head.info.rank,
							 MPI_Aint_add(queue->datadisp[queue_state.head.info.rank], sizeof(elem_t) * queue_state.head.info.position),
							 sizeof(elem_t), MPI_BYTE, queue->win);
			MPI_Win_flush(queue_state.head.info.rank, queue->win);

			// move head
			MPI_Compare_and_swap(&elem.next_info, &queue_state.head.info, &result, MPI_LONG_LONG,
								 MAIN_RANK, queue->statedisp, queue->win);
			MPI_Win_flush(MAIN_RANK, queue->win);

			/*if (!success) {
				continue;
			}*/
			
			if (elem.state == NODE_DELETED) {
				int node_state_free = NODE_FREE;
				rma_nb_queue_state_t new_queue_state;
				get_queue_state(queue, &new_queue_state);
				if (new_queue_state.tail.val == queue_state.tail.val || new_queue_state.tail.val == UNKNOWN_NEXT_NODE_INFO) {
					// move tail if it was not moved or reseted by anyone
					MPI_Compare_and_swap(&this_node_info, &new_queue_state.tail.info, &result, MPI_LONG_LONG, MAIN_RANK,
										 MPI_Aint_add(queue->statedisp, offsets.queue_state_tail_offset), queue->win);
					MPI_Win_flush(MAIN_RANK, queue->win);
				}

				// mark element as FREE
				MPI_Compare_and_swap(&node_state_free, &elem.state, &i_result, MPI_INT, queue_state.tail.info.rank,
									 MPI_Aint_add(queue->datadisp[queue_state.tail.info.rank], sizeof(elem_t) * queue_state.tail.info.position + offsets.elem_state_offset),
									 queue->win);
				MPI_Win_flush(queue_state.tail.info.rank, queue->win);
			}
		}

		break;
	} while (1);

	end_epoch_all(queue->win);

	if (USE_DEBUG) std::cout << "rank " << myrank << ": added " << value << std::endl;

	return CODE_SUCCESS;
}

int dequeue(rma_nb_queue_t *queue, val_t *value) {
	int op_res;
	bool success;
	int node_state_free = NODE_FREE;
	int node_state_acquired = NODE_ACQUIRED;
	int node_state_deleted = NODE_DELETED;
	int i_result;
	u_node_info_t l_result;

	begin_epoch_all(queue->win);
	//int i = 0;
	do {
		//++i;
		//if(i > 1000) g_pause();

		// get head and tail
		get_queue_state(queue, &queue_state);
		if (queue_state.tail.info.rank == UNKNOWN_RANK) {
			end_epoch_all(queue->win);
			//error_msg("data buffer is empty", CODE_DATA_BUFFER_EMPTY);
			return CODE_DATA_BUFFER_EMPTY;
		}
		
		elem_t elem;
		// get tail element
		MPI_Get(&elem, sizeof(elem_t), MPI_BYTE, queue_state.tail.info.rank,
				MPI_Aint_add(queue->datadisp[queue_state.tail.info.rank], sizeof(elem_t) * queue_state.tail.info.position),
				sizeof(elem_t), MPI_BYTE, queue->win);
		MPI_Win_flush(queue_state.tail.info.rank, queue->win);
		if(USE_DEBUG) print(elem);

		if (elem.state == NODE_ACQUIRED) {
			// try to mark tail element as DELETED
			MPI_Compare_and_swap(&node_state_deleted, &node_state_acquired, &i_result, MPI_INT, queue_state.tail.info.rank,
								 MPI_Aint_add(queue->datadisp[queue_state.tail.info.rank], sizeof(elem_t) * queue_state.tail.info.position + offsets.elem_state_offset),
								 queue->win);
			MPI_Win_flush(queue_state.tail.info.rank, queue->win);

			if (USE_DEBUG) {
				print(elem);
				std::cout << "rank " << myrank << ": delete cas 1 result " << i_result << ", elem.state " << elem.state << std::endl;
			}

			success = (i_result == node_state_acquired);
			if (success) {
				// mark succeed
				*value = elem.value;
			} else {
				//continue;
			}

			// move tail
			MPI_Compare_and_swap(&elem.next_info, &queue_state.tail, &l_result, MPI_LONG_LONG, MAIN_RANK,
								 MPI_Aint_add(queue->statedisp, offsets.queue_state_tail_offset), queue->win);
			MPI_Win_flush(MAIN_RANK, queue->win);

			if (USE_DEBUG) std::cout << "rank " << myrank << ": delete cas 1.1 result " << l_result.info.rank << ", " << l_result.info.position << std::endl;

			if (!success) {
				continue;
			}
			
			if (queue_state.head.val != queue_state.tail.val) {
				// if element is not last in the queue - mark element as FREE
				MPI_Compare_and_swap(&node_state_free, &node_state_deleted, &i_result, MPI_INT, queue_state.tail.info.rank,
									 MPI_Aint_add(queue->datadisp[queue_state.tail.info.rank], sizeof(elem_t) * queue_state.tail.info.position + offsets.elem_state_offset),
									 queue->win);
				MPI_Win_flush(queue_state.tail.info.rank, queue->win);

				if (USE_DEBUG) std::cout << "rank " << myrank << ": delete cas 1.3 result " << i_result << std::endl;
			}
		} else {
			//continue;
			// move tail
			MPI_Compare_and_swap(&elem.next_info, &queue_state.tail, &l_result, MPI_LONG_LONG, MAIN_RANK,
								 MPI_Aint_add(queue->statedisp, offsets.queue_state_tail_offset), queue->win);
			MPI_Win_flush(MAIN_RANK, queue->win);

			if (USE_DEBUG) std::cout << "rank " << myrank << ": delete cas 2 result " << l_result.info.rank << ", " << l_result.info.position << std::endl;

			continue;
		}

		break;
	} while (1);

	end_epoch_all(queue->win);

	return CODE_SUCCESS;
}

void rma_nb_queue_free(rma_nb_queue_t *queue) {
	MPI_Barrier(queue->comm);

	MPI_Free_mem(queue->basedisp);
	MPI_Free_mem(queue->datadisp);
	MPI_Free_mem(queue->data);
	MPI_Free_mem(queue);
}

void print(elem_t elem) {
	std::cout << "rank " << myrank << ":\t" << elem.ts << "\t" << elem.value << "\t" << elem.state << "\t(" << elem.next_info.info.rank << ", " << elem.next_info.info.position << ")" << std::endl;
}

void print(rma_nb_queue_t *queue) {
	elem_t elem;
	u_node_info_t next_node_info;

	begin_epoch_all(queue->win);

	MPI_Get(&next_node_info, sizeof(u_node_info_t), MPI_BYTE, MAIN_RANK,
			MPI_Aint_add(queue->statedisp, offsets.queue_state_tail_offset), sizeof(u_node_info_t), MPI_BYTE, queue->win);
	MPI_Win_flush(MAIN_RANK, queue->win);

	if (next_node_info.val != UNKNOWN_RANK) {
		do {
			MPI_Get(&elem, sizeof(elem_t), MPI_BYTE, next_node_info.info.rank,
					MPI_Aint_add(queue->datadisp[next_node_info.info.rank], sizeof(elem_t) * next_node_info.info.position),
					sizeof(elem_t), MPI_BYTE, queue->win);
			MPI_Win_flush(next_node_info.info.rank, queue->win);

			printf("%f\t%d\n", elem.ts, elem.value);
			next_node_info = elem.next_info;
		} while (next_node_info.val != UNKNOWN_NEXT_NODE_INFO);
	}

	end_epoch_all(queue->win);
}

void file_print(rma_nb_queue_t* queue, const char *path) {
	std::ofstream file;
	file.open(path);
	if (!file.is_open()) {
		error_msg("can't write to file", errno);
		return;
	}
	
	elem_t elem;
	u_node_info_t next_node_info;

	begin_epoch_all(queue->win);

	MPI_Get(&next_node_info, sizeof(u_node_info_t), MPI_BYTE, MAIN_RANK,
			MPI_Aint_add(queue->statedisp, offsets.queue_state_tail_offset), sizeof(u_node_info_t), MPI_BYTE, queue->win);
	MPI_Win_flush(MAIN_RANK, queue->win);

	if (next_node_info.val != UNKNOWN_RANK) {
		if (USE_DEBUG) std::cout << "rank " << myrank << ":\t" << next_node_info.info.rank << "\t" << next_node_info.info.position << std::endl;
		do {
			MPI_Get(&elem, sizeof(elem_t), MPI_BYTE, next_node_info.info.rank,
					MPI_Aint_add(queue->datadisp[next_node_info.info.rank], sizeof(elem_t) * next_node_info.info.position),
					sizeof(elem_t), MPI_BYTE, queue->win);
			MPI_Win_flush(next_node_info.info.rank, queue->win);

			file << elem.ts << "\t\t(" << next_node_info.info.rank << ", " << next_node_info.info.position << ")\t\t" << elem.value << "\t\t(" << elem.next_info.info.rank << ", " << elem.next_info.info.position << ")\n";
			next_node_info = elem.next_info;
		} while (next_node_info.val != UNKNOWN_NEXT_NODE_INFO);
	}

	end_epoch_all(queue->win);

	file.close();
}

int main(int argc, char **argv) {
	val_t value;
	int added = 0;
	int deleted = 0;
	int num_of_ops_per_node = 10000;
	int comm_size;
	int start, end;
	double result_time;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	srand(time(0) * myrank);

	rma_nb_queue_t *queue;
	rma_nb_queue_init(&queue, 15000, MPI_COMM_WORLD);
	//init_random_generator();

	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < num_of_ops_per_node; ++i) {
		if (enqueue(queue, rand() % 10000) == CODE_SUCCESS) ++added;
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (myrank == 0) std::cout << "step 1\n";
	if (myrank == 0) file_print(queue, "result1.txt");

	if (myrank == 0) start = clock();

	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < num_of_ops_per_node; ++i) {
		if (rand() % 2) {
			if(enqueue(queue, rand() % 10000) == CODE_SUCCESS) ++added;
		} else {
			if(dequeue(queue, &value) == CODE_SUCCESS) ++deleted;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (myrank == 0) {
		end = clock();
		result_time = ((double)(end - start)) / CLOCKS_PER_SEC;
	}

	if (myrank == 0) std::cout << "step 2\n";
	std::cout << "rank " << myrank << ": added\t" << added << ",\tdeleted\t" << deleted << std::endl;
	if (myrank == 0) std::cout << "time taken: " << result_time << " s\tthroughput: " << ((double)(num_of_ops_per_node*comm_size))/result_time << "\n";

	if(myrank == 0) file_print(queue, "result2.txt");
	MPI_Barrier(MPI_COMM_WORLD);

	rma_nb_queue_free(queue);

	MPI_Finalize();

	return 0;
}