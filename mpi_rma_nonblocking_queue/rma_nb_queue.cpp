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
offsets_t offsets;

void g_pause() {
	if (myrank == MAIN_RANK) {
		system("pause");
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

double get_ts(void) {
	return MPI_Wtime()*MPI_Wtick();
}
void set_ts(elem_t* elem) {
	elem->ts = get_ts();
}

bool CAS(const u_node_info_t* origin_addr, const u_node_info_t* compare_addr, int target_rank, MPI_Aint target_disp, MPI_Win win) {
	u_node_info_t result;
	MPI_Compare_and_swap(origin_addr, compare_addr, &result, MPI_LONG_LONG, target_rank, target_disp, win);
	MPI_Win_flush(target_rank, win);
	return result.raw == compare_addr->raw;
}
bool CAS(const int* origin_addr, const int* compare_addr, int target_rank, MPI_Aint target_disp, MPI_Win win) {
	int result = -2LL;
	MPI_Compare_and_swap(origin_addr, compare_addr, &result, MPI_INT, target_rank, target_disp, win);
	MPI_Win_flush(target_rank, win);
	return result == *compare_addr;
}

void error_msg(const char* msg, int _errno) {
	fprintf(stderr, "%d \t %s", myrank, msg);
	if (_errno != 0)
		fprintf(stderr, ": %s", strerror(_errno));
	fprintf(stderr, "\n");
}

void offsets_init(void) {
	offsets.elem_value = offsetof(elem_t, value);
	offsets.elem_state = offsetof(elem_t, state);
	offsets.elem_next_node_info = offsetof(elem_t, next_node_info);
	offsets.elem_info = offsetof(elem_t, info);
	offsets.elem_ts = offsetof(elem_t, ts);

	offsets.qs_head = offsetof(queue_state_t, head);
	offsets.qs_tail = offsetof(queue_state_t, tail);
}
int disps_init(rma_nb_queue_t* queue) {
	MPI_Alloc_mem(sizeof(MPI_Aint) * queue->n_proc, MPI_INFO_NULL, &queue->basedisp);
	MPI_Alloc_mem(sizeof(MPI_Aint) * queue->n_proc, MPI_INFO_NULL, &queue->datadisp);
	MPI_Alloc_mem(sizeof(MPI_Aint) * queue->n_proc, MPI_INFO_NULL, &queue->statedisp);
	if ((queue->basedisp == NULL) || (queue->datadisp == NULL)) {
		error_msg("basedisp or datadisp memory allocation failed", errno);
		return CODE_ERROR;
	}

	MPI_Allgather(&queue->basedisp_local, 1, MPI_AINT, queue->basedisp, 1, MPI_AINT, queue->comm);
	MPI_Allgather(&queue->datadisp_local, 1, MPI_AINT, queue->datadisp, 1, MPI_AINT, queue->comm);
	MPI_Allgather(&queue->statedisp_local, 1, MPI_AINT, queue->statedisp, 1, MPI_AINT, queue->comm);
	MPI_Bcast(&queue->sentineldisp, 1, MPI_AINT, MAIN_RANK, queue->comm);

	return CODE_SUCCESS;
}
int rma_nb_queue_init(rma_nb_queue_t** queue, int size, MPI_Comm comm) {
	MPI_Alloc_mem(sizeof(rma_nb_queue_t), MPI_INFO_NULL, queue);
	if (*queue == NULL) {
		error_msg("queue memory allocation failed", errno);
		return CODE_ERROR;
	}

	(*queue)->comm = comm;
	MPI_Comm_size(comm, &(*queue)->n_proc);
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

	(*queue)->queue_state.head.raw = UNDEFINED_NODE_INFO;
	(*queue)->queue_state.tail.raw = UNDEFINED_NODE_INFO;
	MPI_Get_address(&(*queue)->queue_state, &(*queue)->statedisp_local);
	MPI_Win_attach((*queue)->win, &(*queue)->queue_state, sizeof(queue_state_t));

	if (myrank == MAIN_RANK) {
		MPI_Get_address(&(*queue)->sentinel, &(*queue)->sentineldisp);
		MPI_Win_attach((*queue)->win, &(*queue)->sentinel, sizeof(elem_t));
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
	MPI_Free_mem(queue->statedisp);
	MPI_Free_mem(queue->data);
	MPI_Free_mem(queue);
}

int get_position(rma_nb_queue_t* queue, int* position) {
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

int get_queue_state(rma_nb_queue_t* queue, int target, queue_state_t* queue_state) {
	if (target == myrank) {
		*queue_state = queue->queue_state;
		return CODE_SUCCESS;
	} else {
		int op_res = MPI_Get_accumulate(queue_state, sizeof(queue_state_t), MPI_BYTE,
										queue_state, sizeof(queue_state_t), MPI_BYTE,
										target, queue->statedisp[target], sizeof(queue_state_t), MPI_BYTE,
										MPI_NO_OP, queue->win);
		return (op_res == MPI_SUCCESS) ? CODE_SUCCESS : CODE_ERROR;
	}
}
int get_head_info(rma_nb_queue_t* queue, int target, u_node_info_t* head_info) {
	if (target == myrank) {
		*head_info = queue->queue_state.head;
		return CODE_SUCCESS;
	} else {
		int op_res = MPI_Get_accumulate(head_info, sizeof(queue_state_t), MPI_BYTE,
										head_info, sizeof(queue_state_t), MPI_BYTE,
										target, MPI_Aint_add(queue->statedisp[target], offsets.qs_head), sizeof(queue_state_t), MPI_BYTE,
										MPI_NO_OP, queue->win);
		return (op_res == MPI_SUCCESS) ? CODE_SUCCESS : CODE_ERROR;
	}
}
int get_tail_info(rma_nb_queue_t* queue, int target, u_node_info_t* tail_info) {
	if (target == myrank) {
		*tail_info = queue->queue_state.tail;
		return CODE_SUCCESS;
	} else {
		int op_res = MPI_Get_accumulate(tail_info, sizeof(queue_state_t), MPI_BYTE,
										tail_info, sizeof(queue_state_t), MPI_BYTE,
										target, MPI_Aint_add(queue->statedisp[target], offsets.qs_tail), sizeof(queue_state_t), MPI_BYTE,
										MPI_NO_OP, queue->win);
		return (op_res == MPI_SUCCESS) ? CODE_SUCCESS : CODE_ERROR;
	}
}
int set_head_info(rma_nb_queue_t* queue, int target, u_node_info_t new_head_info, u_node_info_t prev_head_info) {
	return (CAS(&new_head_info, &prev_head_info, target,
				MPI_Aint_add(queue->statedisp[target], offsets.qs_head), queue->win)) ? CODE_SUCCESS : CODE_ERROR;
}
int set_tail_info(rma_nb_queue_t* queue, int target, u_node_info_t new_tail_info, u_node_info_t prev_tail_info) {
	return (CAS(&new_tail_info, &prev_tail_info, target,
				MPI_Aint_add(queue->statedisp[target], offsets.qs_tail), queue->win)) ? CODE_SUCCESS : CODE_ERROR;
}

MPI_Aint get_elem_disp(rma_nb_queue_t* queue, u_node_info_t elem_info) {
	return MPI_Aint_add(queue->datadisp[elem_info.parsed.rank], sizeof(elem_t) * elem_info.parsed.position);
}
MPI_Aint get_elem_next_node_info_disp(rma_nb_queue_t* queue, u_node_info_t elem_info) {
	return MPI_Aint_add(get_elem_disp(queue, elem_info), offsets.elem_next_node_info);
}
MPI_Aint get_elem_state_disp(rma_nb_queue_t* queue, u_node_info_t elem_info) {
	return MPI_Aint_add(get_elem_disp(queue, elem_info), offsets.elem_state);
}
int get_elem(rma_nb_queue_t* queue, u_node_info_t elem_info, elem_t* elem) {
	int op_res;
	op_res = MPI_Get_accumulate(elem, sizeof(elem_t), MPI_BYTE,
								elem, sizeof(elem_t), MPI_BYTE,
								elem_info.parsed.rank, get_elem_disp(queue, elem_info), sizeof(elem_t), MPI_BYTE,
								MPI_NO_OP, queue->win);
	return op_res;
}
int get_sentinel(rma_nb_queue_t* queue, elem_t* sent) {
	int op_res;
	op_res = MPI_Get_accumulate(sent, sizeof(elem_t), MPI_BYTE,
								sent, sizeof(elem_t), MPI_BYTE,
								MAIN_RANK, queue->sentineldisp, sizeof(elem_t), MPI_BYTE,
								MPI_NO_OP, queue->win);
	return op_res;
}

void exchange(int* a, int* b) {
	*a = *a + *b;
	*b = *a - *b;
	*a = *a - *b;
}
void exclude_rank(rand_provider_t* provider, int rank) {
	for (int i = 0; i < provider->n_proc; ++i) {
		if (provider->nodes[i] == rank) {
			exchange(&provider->nodes[i], &provider->nodes[provider->n_proc - 1]);
			--provider->n_proc;
			break;
		}
	}
}
int get_next_node_rand(rand_provider_t* provider) {
	if (provider->n_proc == 0) return UNDEFINED_RANK;

	int n_proc = provider->n_proc;
	int i = rand() % n_proc;
	int node = provider->nodes[i];
	if (i < provider->n_proc - 1) {
		exchange(&provider->nodes[i], &provider->nodes[n_proc - 1]);
	}
	--provider->n_proc;
	return node;
}
void rand_provider_init(rand_provider_t* provider, int n_proc) {
	provider->n_proc = n_proc;
	provider->nodes = new int[n_proc]; //TODO mb use MPI_Alloc_mem?
	for (int i = 0; i < n_proc; ++i) {
		provider->nodes[i] = i;
	}
}
void rand_provider_free(rand_provider_t* provider) {
	delete provider->nodes;
}
void bcast_meta_init(bcast_meta_t* meta) {
	meta->should_update_head = true;
	meta->should_update_tail = true;
}

int bcastpart_head_info(rma_nb_queue_t* queue, int target, elem_t* elem, bcast_meta_t* meta) {
	do {
		get_head_info(queue, target, &meta->head_info);
		get_elem(queue, meta->head_info, &meta->head);
		if (meta->head.ts > elem->ts) return CODE_ERROR;
	} while (set_head_info(queue, target, elem->info, meta->head_info) != CODE_SUCCESS);

	return CODE_SUCCESS;
}
int bcastpart_tail_info(rma_nb_queue_t* queue, int target, elem_t* elem, bcast_meta_t* meta) {
	do {
		get_head_info(queue, target, &meta->tail_info);
		get_elem(queue, meta->tail_info, &meta->tail);
		if (meta->tail.ts > elem->ts) return CODE_ERROR;
	} while (set_tail_info(queue, target, elem->info, meta->tail_info) != CODE_SUCCESS);

	return CODE_SUCCESS;
}
int bcastpart_head_tail_info(rma_nb_queue_t* queue, int target, elem_t* elem, bcast_meta_t* meta) {
	if (meta->should_update_head) {
		do {
			get_head_info(queue, target, &meta->head_info);
			get_elem(queue, meta->head_info, &meta->head);
			if (meta->head.ts > elem->ts) {
				meta->should_update_head = false;
			} else {
				if (set_head_info(queue, target, elem->info, meta->head_info) != CODE_SUCCESS) continue;
			}
		} while (0);
	}

	if (meta->should_update_tail) {
		do {
			get_tail_info(queue, target, &meta->tail_info);
			get_elem(queue, meta->tail_info, &meta->tail);
			if (meta->tail.ts > elem->ts) {
				meta->should_update_tail = false;
			} else {
				if(set_tail_info(queue, target, elem->info, meta->tail_info) != CODE_SUCCESS) continue;
			}
		} while (0);
	}

	if (!meta->should_update_head && !meta->should_update_tail) return CODE_ERROR;

	return CODE_SUCCESS;
}
int bcastpart_tail_head_info(rma_nb_queue_t* queue, int target, elem_t* elem, bcast_meta_t* meta) {
	if (meta->should_update_tail) {
		do {
			get_tail_info(queue, target, &meta->tail_info);
			get_elem(queue, meta->tail_info, &meta->tail);
			if (meta->tail.ts > elem->ts) {
				meta->should_update_tail = false;
			} else {
				if (set_tail_info(queue, target, elem->info, meta->tail_info) != CODE_SUCCESS) continue;
			}
		} while (0);
	}

	if (meta->should_update_head) {
		do {
			get_head_info(queue, target, &meta->head_info);
			get_elem(queue, meta->head_info, &meta->head);
			if (meta->head.ts > elem->ts) {
				meta->should_update_head = false;
			} else {
				if (set_head_info(queue, target, elem->info, meta->head_info) != CODE_SUCCESS) continue;
			}
		} while (0);
	}

	if (!meta->should_update_head && !meta->should_update_tail) return CODE_ERROR;

	return CODE_SUCCESS;
}
int bcast_node_info_template(rma_nb_queue_t* queue, elem_t* elem, int priority_rank, bcast_meta_t* meta) {
	int op_res;
	int target_node;
	rand_provider_t provider;

	rand_provider_init(&provider, queue->n_proc);

	if (priority_rank != myrank) {	// send to priority rank, if priority is myrank - skip this step
		op_res = meta->bcast_method(queue, priority_rank, elem, meta);
		if (op_res == CODE_ERROR) {
			rand_provider_free(&provider);
			return CODE_PARTLY_SUCCESS;
		}
		exclude_rank(&provider, priority_rank);
	}

	op_res = meta->bcast_method(queue, myrank, elem, meta);	// send to me
	if (op_res == CODE_ERROR) {
		rand_provider_free(&provider);
		return CODE_PARTLY_SUCCESS;
	}
	exclude_rank(&provider, myrank);

	while (1) {
		target_node = get_next_node_rand(&provider);
		if (target_node == UNDEFINED_RANK) {
			rand_provider_free(&provider);
			return CODE_SUCCESS;
		}

		op_res = meta->bcast_method(queue, target_node, elem, meta);
		if (op_res == CODE_ERROR) {
			rand_provider_free(&provider);
			return CODE_PARTLY_SUCCESS;
		}
	}
}

int bcast_head_info(rma_nb_queue_t* queue, elem_t elem, int priority_rank) {
	bcast_meta_t meta;
	bcast_meta_init(&meta);
	meta.bcast_method = bcastpart_head_info;

	return bcast_node_info_template(queue, &elem, priority_rank, &meta);
}
int bcast_tail_info(rma_nb_queue_t* queue, elem_t elem, int priority_rank) {
	bcast_meta_t meta;
	bcast_meta_init(&meta);
	meta.bcast_method = bcastpart_tail_info;

	return bcast_node_info_template(queue, &elem, priority_rank, &meta);
}
int bcast_head_tail_info(rma_nb_queue_t* queue, elem_t elem, int priority_rank) {
	bcast_meta_t meta;
	bcast_meta_init(&meta);
	meta.bcast_method = bcastpart_head_tail_info;

	return bcast_node_info_template(queue, &elem, priority_rank, &meta);
}
int bcast_tail_head_info(rma_nb_queue_t* queue, elem_t elem, int priority_rank) {
	bcast_meta_t meta;
	bcast_meta_init(&meta);
	meta.bcast_method = bcastpart_tail_head_info;

	return bcast_node_info_template(queue, &elem, priority_rank, &meta);
}

int elem_init(rma_nb_queue_t* queue, elem_t** elem, val_t value) {
	int position;
	int op_res = get_position(queue, &position);
	if (op_res == CODE_DATA_BUFFER_FULL) {
		return CODE_DATA_BUFFER_FULL;
	}

	queue->data[position].value = value;
	queue->data[position].state = NODE_ACQUIRED;
	queue->data[position].next_node_info.raw = UNDEFINED_NODE_INFO;
	queue->data[position].info.parsed.rank = myrank;
	queue->data[position].info.parsed.position = position;

	*elem = &queue->data[position];
	return CODE_SUCCESS;
}
int enqueue(rma_nb_queue_t* queue, val_t value) {
	int op_res;

	int node_state_free = NODE_FREE;
	int node_state_acquired = NODE_ACQUIRED;
	int node_state_deleted = NODE_DELETED;

	u_node_info_t tail_info;
	u_node_info_t undefined_node_info;
	
	elem_t *new_elem;
	elem_t sentinel;
	elem_t tail;

	begin_epoch_all(queue->win);

	op_res = elem_init(queue, &new_elem, value);
	if (op_res == CODE_DATA_BUFFER_FULL) {
		return CODE_DATA_BUFFER_FULL;
	}

	undefined_node_info.raw = UNDEFINED_NODE_INFO;

start:
	get_tail_info(queue, myrank, &tail_info);
	if (tail_info.raw == UNDEFINED_NODE_INFO) {
		get_sentinel(queue, &sentinel);
		if (sentinel.next_node_info.raw == UNDEFINED_NODE_INFO) {
			set_ts(new_elem);
			if (CAS(&new_elem->info, &undefined_node_info, sentinel.info.parsed.rank,
					MPI_Aint_add(queue->sentineldisp, offsets.elem_next_node_info), queue->win)) {
				bcast_tail_head_info(queue, *new_elem, myrank);
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
				if (CAS(&new_elem->info, &undefined_node_info, tail.info.parsed.rank,
						get_elem_next_node_info_disp(queue, tail.info), queue->win)) {
					get_elem(queue, tail.info, &tail);
					if (tail.state == NODE_DELETED) {
						bcast_tail_head_info(queue, *new_elem, tail.info.parsed.rank);
						CAS(&node_state_free, &node_state_deleted, tail.info.parsed.rank,
							get_elem_state_disp(queue, tail.info), queue->win);
					} else {
						bcast_tail_info(queue, *new_elem, myrank);
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
				if (CAS(&new_elem->info, &undefined_node_info, tail.info.parsed.rank,
						get_elem_next_node_info_disp(queue, tail.info), queue->win)) {
					bcast_tail_head_info(queue, *new_elem, tail.info.parsed.rank);
					CAS(&node_state_free, &node_state_deleted, tail.info.parsed.rank,
						get_elem_state_disp(queue, tail.info), queue->win);

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

int dequeue(rma_nb_queue_t* queue, val_t* value) {
	int op_res;

	int node_state_free = NODE_FREE;
	int node_state_acquired = NODE_ACQUIRED;
	int node_state_deleted = NODE_DELETED;

	u_node_info_t head_info;

	elem_t sentinel;
	elem_t head;

	begin_epoch_all(queue->win);
	
start:
	get_head_info(queue, myrank, &head_info);
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
			if (CAS(&node_state_deleted, &node_state_acquired, head.info.parsed.rank,
					get_elem_state_disp(queue, head.info), queue->win)) {
				*value = head.value;

				if (head.next_node_info.raw != UNDEFINED_NODE_INFO) {
					elem_t next_head;
					get_elem(queue, head.next_node_info, &next_head);
					bcast_head_info(queue, next_head, myrank);
					CAS(&node_state_free, &node_state_deleted, head.info.parsed.rank,
						get_elem_state_disp(queue, head.info), queue->win);
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

		goto start; // head became free, redo algorithm
	}
}

void print(elem_t elem) {
	std::cout << "rank " << myrank << ":\t" << elem.ts << "\t" << elem.value << "\t" << elem.state << "\t(" << elem.next_node_info.parsed.rank << ", " << elem.next_node_info.parsed.position << ")" << std::endl;
}
void print(rma_nb_queue_t* queue) {
	elem_t elem;
	u_node_info_t next_node_info;

	begin_epoch_all(queue->win);

	MPI_Get(&next_node_info, sizeof(u_node_info_t), MPI_BYTE, MAIN_RANK,
			MPI_Aint_add(queue->statedisp_local, offsets.qs_tail), sizeof(u_node_info_t), MPI_BYTE, queue->win);
	MPI_Win_flush(MAIN_RANK, queue->win);

	if (next_node_info.raw != UNDEFINED_RANK) {
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
void print(rand_provider_t* provider) {
	std::cout << "{ ";
	for (int i = 0; i < provider->n_proc; ++i) {
		std::cout << provider->nodes[i] << ' ';
	}
	std::cout << "} " << provider->n_proc << std::endl;
}
void file_print(rma_nb_queue_t* queue, const char* path) {
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
			MPI_Aint_add(queue->statedisp_local, offsets.qs_tail), sizeof(u_node_info_t), MPI_BYTE, queue->win);
	MPI_Win_flush(MAIN_RANK, queue->win);

	if (next_node_info.raw != UNDEFINED_RANK) {
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

void test_get_next_node_rand() {
	srand(time(0));
	int n_proc = 15;
	int next_node;
	rand_provider_t provider;
	rand_provider_init(&provider, n_proc);

	print(&provider);
	exclude_rank(&provider, n_proc / 2);
	std::cout << "Excluded node: " << n_proc / 2 << std::endl;

	do {
		print(&provider);
		next_node = get_next_node_rand(&provider);
		std::cout << "Next node: " << next_node << std::endl;
	} while (next_node != UNDEFINED_RANK);

	rand_provider_free(&provider);
}
void test_displacements(int argc, char** argv) {


	MPI_Init(&argc, &argv);

	MPI_Finalize();
}
void test1(int argc, char** argv) {
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

	rma_nb_queue_t* queue;
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
			if (enqueue(queue, rand() % 10000) == CODE_SUCCESS) ++added;
		} else {
			if (dequeue(queue, &value) == CODE_SUCCESS) ++deleted;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (myrank == 0) {
		end = clock();
		result_time = ((double)(end - start)) / CLOCKS_PER_SEC;
	}

	if (myrank == 0) std::cout << "step 2\n";
	std::cout << "rank " << myrank << ": added\t" << added << ",\tdeleted\t" << deleted << std::endl;
	if (myrank == 0) std::cout << "time taken: " << result_time << " s\tthroughput: " << ((double)(num_of_ops_per_node * comm_size)) / result_time << "\n";

	if (myrank == 0) file_print(queue, "result2.txt");
	MPI_Barrier(MPI_COMM_WORLD);

	rma_nb_queue_free(queue);

	MPI_Finalize();
}

int main(int argc, char** argv) {
	

	return 0;
}