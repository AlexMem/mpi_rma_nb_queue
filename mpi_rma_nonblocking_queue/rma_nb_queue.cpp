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

double get_ts(void) {
	return MPI_Wtime();
}
int set_ts(elem_t* elem) {
	elem->ts = get_ts();
}

bool CAS(const u_node_info_t* origin_addr, const u_node_info_t* compare_addr, u_node_info_t* result_addr, int target_rank, MPI_Aint target_disp, MPI_Win win) {
	MPI_Compare_and_swap(origin_addr, compare_addr, result_addr, MPI_LONG_LONG, target_rank, target_disp, win);
	MPI_Win_flush(target_rank, win);
	return result_addr->raw == compare_addr->raw;
}
bool CAS(const int* origin_addr, const int* compare_addr, int* result_addr, int target_rank, MPI_Aint target_disp, MPI_Win win) {
	MPI_Compare_and_swap(origin_addr, compare_addr, result_addr, MPI_INT, target_rank, target_disp, win);
	MPI_Win_flush(target_rank, win);
	return *result_addr == *compare_addr;
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
	offsets.elem_next_info_offset = offsetof(elem_t, next_node_info);
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
	queue_state.head.raw = UNDEFINED_NODE_INFO;
	queue_state.tail.raw = UNDEFINED_NODE_INFO;
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
void rma_nb_queue_free(rma_nb_queue_t* queue) {
	MPI_Barrier(queue->comm);

	MPI_Free_mem(queue->basedisp);
	MPI_Free_mem(queue->datadisp);
	MPI_Free_mem(queue->data);
	MPI_Free_mem(queue);
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
	if (USE_DEBUG) std::cout << "rank " << myrank << ": got queue state head(" << queue_state->head.parsed.rank << ", " << queue_state->head.parsed.position << ") tail(" << queue_state->tail.parsed.rank << ", " << queue_state->tail.parsed.position << ")\t" << op_res << std::endl;
	return op_res;
}
int get_head_info(rma_nb_queue_t *queue, u_node_info_t *head_info) {}
int get_tail_info(rma_nb_queue_t *queue, u_node_info_t *tail_info) {}

int get_next_node_rand() {}

int get_elem(rma_nb_queue_t *queue, u_node_info_t elem_info, elem_t *elem) {}
int get_sentinel(rma_nb_queue_t* queue, elem_t *sent) {}

int set_head_info(rma_nb_queue_t *queue, int target, u_node_info_t head_info) {}
int set_tail_info(rma_nb_queue_t *queue, int target, u_node_info_t tail_info) {}

int bcast_head_info(rma_nb_queue_t* queue, elem_t elem) {
	rma_nb_queue_state_t qs;
	elem_t head;
	int target_node;

	while (1) {
		target_node = get_next_node_rand();
		if (target_node) return CODE_SUCCESS;

		get_queue_state(queue, &qs);
		get_elem(queue, qs.head, &head);
		if (head.ts > elem.ts) return CODE_SUCCESS;
		set_head_info(queue, target_node, elem.info);
	}
}
int bcast_tail_info(rma_nb_queue_t* queue, elem_t elem) {
	rma_nb_queue_state_t qs;
	elem_t tail;
	int target_node;

	while (1) {
		target_node = get_next_node_rand();
		if (target_node) return CODE_SUCCESS;

		get_queue_state(queue, &qs);
		get_elem(queue, qs.tail, &tail);
		if (tail.ts > elem.ts) return CODE_SUCCESS;
		set_tail_info(queue, target_node, elem.info);
	}
}
int bcast_head_tail_info(rma_nb_queue_t* queue, elem_t elem) {
	rma_nb_queue_state_t qs;
	elem_t head;
	elem_t tail;
	int target_node;
	bool should_update_head = true;
	bool should_update_tail = true;

	while (1) {
		target_node = get_next_node_rand();
		if (target_node) return CODE_SUCCESS;

		get_queue_state(queue, &qs);

		if (should_update_head) {
			get_elem(queue, qs.head, &head);
			if (head.ts > elem.ts) {
				should_update_head = false;
			} else {
				set_head_info(queue, target_node, elem.info);
			}
		}

		if (should_update_tail) {
			get_elem(queue, qs.tail, &tail);
			if (tail.ts > elem.ts) {
				should_update_tail = false;
			} else {
				set_tail_info(queue, target_node, elem.info);
			}
		}

		if (!should_update_head && !should_update_tail) return CODE_SUCCESS;
	}
}
int bcast_tail_head_info(rma_nb_queue_t* queue, elem_t elem) {
	rma_nb_queue_state_t qs;
	elem_t head;
	elem_t tail;
	int target_node;
	bool should_update_head = true;
	bool should_update_tail = true;

	while (1) {
		target_node = get_next_node_rand();
		if (target_node) return CODE_SUCCESS;

		get_queue_state(queue, &qs);

		if (should_update_tail) {
			get_elem(queue, qs.tail, &tail);
			if (tail.ts > elem.ts) {
				should_update_tail = false;
			} else {
				set_tail_info(queue, target_node, elem.info);
			}
		}

		if (should_update_head) {
			get_elem(queue, qs.head, &head);
			if (head.ts > elem.ts) {
				should_update_head = false;
			} else {
				set_head_info(queue, target_node, elem.info);
			}
		}

		if (!should_update_head && !should_update_tail) return CODE_SUCCESS;
	}
}

int elem_init(rma_nb_queue_t* queue, elem_t *elem, val_t value) {
	int pos;
	int op_res = get_position(queue, &pos);
	if (op_res == CODE_DATA_BUFFER_FULL) {
		return CODE_DATA_BUFFER_FULL;
	}

	queue->data[pos].value = value;
	queue->data[pos].state = NODE_ACQUIRED;
	queue->data[pos].next_node_info.raw = UNDEFINED_NODE_INFO;
	queue->data[pos].info.parsed.rank = myrank;
	queue->data[pos].info.parsed.position = pos;

	elem = &queue->data[pos];
	return CODE_SUCCESS;
}
int enqueue(rma_nb_queue_t *queue, val_t value) {
	int op_res;

	int node_state_free = NODE_FREE;
	int node_state_acquired = NODE_ACQUIRED;
	int node_state_deleted = NODE_DELETED;

	int state_cas_result;
	u_node_info_t info_cas_result;

	u_node_info_t tail_info;
	u_node_info_t undefined_node_info;
	
	elem_t *new_elem;
	elem_t sentinel;
	elem_t tail;

	begin_epoch_all(queue->win);

	op_res = elem_init(queue, new_elem, value);
	if (op_res == CODE_DATA_BUFFER_FULL) {
		return CODE_DATA_BUFFER_FULL;
	}

	undefined_node_info.raw = UNDEFINED_NODE_INFO;
	info_cas_result.raw = -2LL;

start:
	get_tail_info(queue, &tail_info);
	if (tail_info.raw == UNDEFINED_NODE_INFO) {
		get_sentinel(queue, &sentinel);
		if (sentinel.next_node_info.raw == UNDEFINED_NODE_INFO) {
			set_ts(new_elem);
			if (CAS(&new_elem->info, &undefined_node_info, &info_cas_result, sentinel.info.parsed.rank, NULL, queue->win)) {
				bcast_tail_head_info(queue, *new_elem);
				end_epoch_all(queue->win);
				return CODE_SUCCESS;
			}
		}
		tail_info = sentinel.next_node_info;
	}

	get_elem(queue, tail_info, &tail);

	while (1) {
		if (tail.state == NODE_ACQUIRED) {
			if (tail.next_node_info.raw == UNDEFINED_NODE_INFO) {
				set_ts(new_elem);
				if (CAS(&new_elem->info, &undefined_node_info, &info_cas_result, tail.info.parsed.rank, NULL, queue->win)) {
					get_elem(queue, tail.info, &tail);
					if (tail.state == NODE_DELETED) {
						bcast_tail_head_info(queue, *new_elem);
						CAS(&node_state_free, &node_state_deleted, &state_cas_result, tail.info.parsed.rank, NULL, queue->win);
					} else {
						bcast_tail_info(queue, *new_elem);
					}

					end_epoch_all(queue->win);
					return CODE_SUCCESS;
				}
			}

			get_elem(queue, tail.next_node_info, &tail);
			continue;
		}

		if (tail.state == NODE_DELETED) {
			if (tail.next_node_info.raw == UNDEFINED_NODE_INFO) {
				set_ts(new_elem);
				if (CAS(&new_elem->info, &undefined_node_info, &info_cas_result, tail.info.parsed.rank, NULL, queue->win)) {
					bcast_tail_head_info(queue, *new_elem);
					CAS(&node_state_free, &node_state_deleted, &state_cas_result, tail.info.parsed.rank, NULL, queue->win);

					end_epoch_all(queue->win);
					return CODE_SUCCESS;
				}
			} else {
				get_elem(queue, tail.next_node_info, &tail);
				continue;
			}
		}

		goto start;
	}
}

int dequeue(rma_nb_queue_t *queue, val_t *value) {
	int op_res;

	int node_state_free = NODE_FREE;
	int node_state_acquired = NODE_ACQUIRED;
	int node_state_deleted = NODE_DELETED;

	int state_cas_result;
	u_node_info_t info_cas_result;

	u_node_info_t head_info;

	elem_t sentinel;
	elem_t head;

	begin_epoch_all(queue->win);
	
start:
	get_head_info(queue, &head_info);
	if (head_info.raw == UNDEFINED_NODE_INFO) {
		get_sentinel(queue, &sentinel);
		if (sentinel.next_node_info.raw == UNDEFINED_NODE_INFO) {
			end_epoch_all(queue->win);
			return CODE_NO_HEAD;
		}
		head_info = sentinel.next_node_info;
	}

	get_elem(queue, head_info, &head);

	while (1) {
		if (head.state == NODE_ACQUIRED) {
			if (CAS(&node_state_deleted, &node_state_acquired, &state_cas_result, head.info.parsed.rank, NULL, queue->win)) {
				*value = head.value;

				if (head.next_node_info.raw != UNDEFINED_NODE_INFO) {
					elem_t next_head;
					get_elem(queue, head.next_node_info, &next_head);
					bcast_head_info(queue, next_head);
					CAS(&node_state_free, &node_state_deleted, &state_cas_result, head.info.parsed.rank, NULL, queue->win);
				}

				end_epoch_all(queue->win);
				return CODE_SUCCESS;
			}

			get_elem(queue, head_info, &head); // refresh same head
		}

		if (head.state == NODE_DELETED) {
			if (head.next_node_info.raw != UNDEFINED_NODE_INFO) {
				get_elem(queue, head.next_node_info, &head); // move next
				continue;
			} else {
				end_epoch_all(queue->win);
				return CODE_NO_HEAD;
			}
		}

		goto start; // head become freed, redo algorithm
	}
}

void print(elem_t elem) {
	std::cout << "rank " << myrank << ":\t" << elem.ts << "\t" << elem.value << "\t" << elem.state << "\t(" << elem.next_node_info.parsed.rank << ", " << elem.next_node_info.parsed.position << ")" << std::endl;
}
void print(rma_nb_queue_t *queue) {
	elem_t elem;
	u_node_info_t next_node_info;

	begin_epoch_all(queue->win);

	MPI_Get(&next_node_info, sizeof(u_node_info_t), MPI_BYTE, MAIN_RANK,
			MPI_Aint_add(queue->statedisp, offsets.queue_state_tail_offset), sizeof(u_node_info_t), MPI_BYTE, queue->win);
	MPI_Win_flush(MAIN_RANK, queue->win);

	if (next_node_info.raw != UNKNOWN_RANK) {
		do {
			MPI_Get(&elem, sizeof(elem_t), MPI_BYTE, next_node_info.parsed.rank,
					MPI_Aint_add(queue->datadisp[next_node_info.parsed.rank], sizeof(elem_t) * next_node_info.parsed.position),
					sizeof(elem_t), MPI_BYTE, queue->win);
			MPI_Win_flush(next_node_info.parsed.rank, queue->win);

			printf("%f\t%d\n", elem.ts, elem.value);
			next_node_info = elem.next_node_info;
		} while (next_node_info.raw != UNDEFINED_NODE_INFO);
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

	if (next_node_info.raw != UNKNOWN_RANK) {
		if (USE_DEBUG) std::cout << "rank " << myrank << ":\t" << next_node_info.parsed.rank << "\t" << next_node_info.parsed.position << std::endl;
		do {
			MPI_Get(&elem, sizeof(elem_t), MPI_BYTE, next_node_info.parsed.rank,
					MPI_Aint_add(queue->datadisp[next_node_info.parsed.rank], sizeof(elem_t) * next_node_info.parsed.position),
					sizeof(elem_t), MPI_BYTE, queue->win);
			MPI_Win_flush(next_node_info.parsed.rank, queue->win);

			file << elem.ts << "\t\t(" << next_node_info.parsed.rank << ", " << next_node_info.parsed.position << ")\t\t" << elem.value << "\t\t(" << elem.next_node_info.parsed.rank << ", " << elem.next_node_info.parsed.position << ")\n";
			next_node_info = elem.next_node_info;
		} while (next_node_info.raw != UNDEFINED_NODE_INFO);
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