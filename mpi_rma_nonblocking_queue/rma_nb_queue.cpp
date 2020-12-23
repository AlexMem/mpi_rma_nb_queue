#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>
#include <limits>
#include <iomanip>

#include "rma_nb_queue.h"
#include "include/utils.h"

#define USE_DEBUG 0
#define USE_MPI_CALLS_COUNTING 1

#define LOG_PRINT_CONSOLE 0b10
#define LOG_PRINT_FILE 0b01
#define LOG_PRINT_MODE 0b01 // 0b10 - console only, 0b01 - file (one per process) only, 0b11 - console and file

MPI_Aint get_elem_lock_disp(rma_nb_queue_t* queue, u_node_info_t elem_info);
MPI_Aint get_qs_head_info_lock_disp(rma_nb_queue_t* queue, int target);
MPI_Aint get_qs_tail_info_lock_disp(rma_nb_queue_t* queue, int target);
void exclude_rank(rand_provider_t* provider, int rank);
int get_next_node_rand(rand_provider_t* provider);
void rand_provider_init(rand_provider_t* provider, int n_proc);
void rand_provider_free(rand_provider_t* provider);
int get_elem(rma_nb_queue_t* queue, u_node_info_t elem_info, elem_t* elem);
std::string print(u_node_info_t info);
std::string print(elem_t elem);

typedef struct {
	unsigned int added;
	unsigned int deleted;
} test_result;
typedef struct {
	int* to;
	int n;
} mpi_call_counter_t;

std::string log_file_name;
std::ofstream log_file;
std::ostringstream l_str;

mpi_call_counter_t mpi_call_counter;

int myrank;
offsets_t offsets;

int unlocked = UNDEFINED_RANK;



// ----- LOGGING -----
void log_init(int rank) {
	if (!(LOG_PRINT_MODE & LOG_PRINT_FILE)) return;
	if (log_file.is_open()) return;

	std::ostringstream s;
	s << "log_proc_" << rank << ".txt";
	log_file_name = s.str();
	log_file.open(log_file_name, std::ios::trunc);
}
void log_(std::string content, int mode) {
	if (mode & LOG_PRINT_CONSOLE) {
		std::cout << "rank " << myrank << ": " << content;
	}

	if (mode & LOG_PRINT_FILE) {
		if (log_file.is_open()) {
			log_file << "rank " << myrank << ": " << content;
		}
	}
}
void log_(std::ostringstream& content, int mode) {
	log_(content.str(), mode);
	content.str(std::string());
	content.clear();
}
void log_(std::string content) {
	log_(content, LOG_PRINT_MODE);
}
void log_(std::ostringstream& content) {
	log_(content, LOG_PRINT_MODE);
}
void log_close() {
	if (log_file.is_open()) {
		log_file.close();
	}
}

// ----- MPI CALLS COUNTING -----
void mpi_call_counter_init(mpi_call_counter_t* counter, int n_proc) {
	counter->to = new int[n_proc];
	counter->n = n_proc;
	for(int i = 0; i < n_proc; ++i) {
		counter->to[i] = 0;
	}
}
void count(mpi_call_counter_t* counter, int proc) {
	++counter->to[proc];
}
void sum(mpi_call_counter_t* result, mpi_call_counter_t* addition) {
	for(int i = 0; i < result->n && i < addition->n; ++i) {
		result->to[i] += addition->to[i];
	}
}
void mpi_call_counter_free(mpi_call_counter_t* counter) {
	delete counter->to;
}

// ----- PAUSE -----
void g_pause() {
	if (myrank == MAIN_RANK) {
		std::cin.get();
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

// ----- TIMESTAMPING -----
double get_ts(void) {
	return MPI_Wtime();
}
void set_ts(elem_t* elem, double offset) {
	elem->ts = get_ts() + offset;
}

// ----- CAS WRAPPING -----
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
bool CAS(const bool* origin_addr, const bool* compare_addr, int target_rank, MPI_Aint target_disp, MPI_Win win) {
	bool result;
	MPI_Compare_and_swap(origin_addr, compare_addr, &result, MPI_BYTE, target_rank, target_disp, win);
	MPI_Win_flush(target_rank, win);
	return result == *compare_addr;
}

// ----- LOCKING/UNLOCKING -----
// lock/unlock elem_t
void lock(rma_nb_queue_t* queue, u_node_info_t target) {
	if (USE_DEBUG) {
		l_str << "trying to lock elem " << print(target) << std::endl;
		// log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_MODE);
		log_(l_str);
	}

	while(!CAS(&myrank, &unlocked, target, get_elem_lock_disp(queue, target), queue->win));

	if (USE_DEBUG) {
		l_str << "\tlocked elem " << print(target) << std::endl;
		log_(l_str);
	}
}
void unlock(rma_nb_queue_t* queue, u_node_info_t target) {
	bool op_res = CAS(&unlocked, &myrank, get_elem_lock_disp(queue, target), queue->win);
	// prev_locked_proc = UNDEFINED_RANK;
	if (USE_DEBUG && op_res) {
		l_str << "unlocked elem " << print(target) << std::endl;
		// log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_MODE);
		log_(l_str);
	}
}
// lock/unlock head_info
void lock_head_info(rma_nb_queue_t* queue, int target) {
	if (USE_DEBUG) {
		l_str << "trying to lock head_info in target " << target << std::endl;
		// log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_MODE);
		log_(l_str);
	}

	while(!CAS(&myrank, &unlocked, target, get_qs_head_info_lock_disp(queue, target), queue->win));

	if (USE_DEBUG) {
		l_str << "\tlocked head_info in target " << target << std::endl;
		log_(l_str);
	}
}
void unlock_head_info(rma_nb_queue_t* queue, int target) {
	bool op_res = CAS(&unlocked, &myrank, get_qs_head_info_lock_disp(queue, target), queue->win);
	// prev_locked_proc = UNDEFINED_RANK;
	if (USE_DEBUG && op_res) {
		l_str << "unlocked head_info in target " << print(target) << std::endl;
		// log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_MODE);
		log_(l_str);
	}
}
// lock/unlock tail_info
void lock_head_info(rma_nb_queue_t* queue, int target) {
	if (USE_DEBUG) {
		l_str << "trying to lock tail_info in target " << target << std::endl;
		// log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_MODE);
		log_(l_str);
	}

	while(!CAS(&myrank, &unlocked, target, get_qs_tail_info_lock_disp(queue, target), queue->win));

	if (USE_DEBUG) {
		l_str << "\tlocked tail_info in target " << target << std::endl;
		log_(l_str);
	}
}
void unlock_head_info(rma_nb_queue_t* queue, int target) {
	bool op_res = CAS(&unlocked, &myrank, get_qs_tail_info_lock_disp(queue, target), queue->win);
	// prev_locked_proc = UNDEFINED_RANK;
	if (USE_DEBUG && op_res) {
		l_str << "unlocked tail_info in target " << print(target) << std::endl;
		// log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_MODE);
		log_(l_str);
	}
}

void sentinel_init(elem_t* sentinel) {
	sentinel->lock = UNDEFINED_RANK;
	sentinel->ts = UNDEFINED_TS;
	sentinel->value = 0;
	sentinel->state = NODE_ACQUIRED;
	sentinel->info.parsed.rank = MAIN_RANK;
	sentinel->info.parsed.position = -1;
	sentinel->next_node_info.raw = UNDEFINED_NODE_INFO;
}
void elem_reset(elem_t* elem) {
	elem->lock = UNDEFINED_RANK;
	elem->ts = UNDEFINED_TS;
	elem->value = 0;
	elem->state = NODE_FREE;
	elem->info.raw = UNDEFINED_NODE_INFO;
	elem->next_node_info.raw = UNDEFINED_NODE_INFO;
}
void offsets_init(void) {
	offsets.elem_value = offsetof(elem_t, value);
	offsets.elem_state = offsetof(elem_t, state);
	offsets.elem_next_node_info = offsetof(elem_t, next_node_info);
	offsets.elem_info = offsetof(elem_t, info);
	offsets.elem_ts = offsetof(elem_t, ts);
	offsets.elem_lock = offsetof(elem_t, lock);
	offsets.qs_head = offsetof(queue_state_t, head_info);
	offsets.qs_tail = offsetof(queue_state_t, tail_info);
	offsets.qs_head_info_lock = offsetof(queue_state_t, head_info_lock);
	offsets.qs_tail_info_lock = offsetof(queue_state_t, tail_info_lock);
}
int disps_init(rma_nb_queue_t* queue) {
	MPI_Alloc_mem(sizeof(MPI_Aint) * queue->n_proc, MPI_INFO_NULL, &queue->basedisp);
	MPI_Alloc_mem(sizeof(MPI_Aint) * queue->n_proc, MPI_INFO_NULL, &queue->datadisp);
	MPI_Alloc_mem(sizeof(MPI_Aint) * queue->n_proc, MPI_INFO_NULL, &queue->statedisp);
	MPI_Alloc_mem(sizeof(MPI_Aint) * queue->n_proc, MPI_INFO_NULL, &queue->headdisp);
	MPI_Alloc_mem(sizeof(MPI_Aint) * queue->n_proc, MPI_INFO_NULL, &queue->taildisp);
	if ((queue->basedisp == NULL) || (queue->datadisp == NULL) ||
		(queue->statedisp == NULL) || (queue->headdisp == NULL) ||
		(queue->taildisp == NULL)) {
		return CODE_ERROR;
	}

	MPI_Allgather(&queue->basedisp_local, 1, MPI_AINT, queue->basedisp, 1, MPI_AINT, queue->comm);
	MPI_Allgather(&queue->datadisp_local, 1, MPI_AINT, queue->datadisp, 1, MPI_AINT, queue->comm);
	MPI_Allgather(&queue->statedisp_local, 1, MPI_AINT, queue->statedisp, 1, MPI_AINT, queue->comm);
	MPI_Allgather(&queue->headdisp_local, 1, MPI_AINT, queue->headdisp, 1, MPI_AINT, queue->comm);
	MPI_Allgather(&queue->taildisp_local, 1, MPI_AINT, queue->taildisp, 1, MPI_AINT, queue->comm);
	MPI_Bcast(&queue->sentineldisp, 1, MPI_AINT, MAIN_RANK, queue->comm);

	return CODE_SUCCESS;
}
int rma_nb_queue_init(rma_nb_queue_t** queue, int size_per_node, MPI_Comm comm) {
	MPI_Alloc_mem(sizeof(rma_nb_queue_t), MPI_INFO_NULL, queue);
	if (*queue == NULL) {
		return CODE_ERROR;
	}

	(*queue)->comm = comm;
	MPI_Comm_rank((*queue)->comm, &myrank);
	MPI_Comm_size((*queue)->comm, &(*queue)->n_proc);
	MPI_Get_address(*queue, &(*queue)->basedisp_local);

	int data_buffer_size = sizeof(elem_t) * size_per_node;
	MPI_Alloc_mem(data_buffer_size, MPI_INFO_NULL, &(*queue)->data);
	if ((*queue)->data == NULL) {
		return CODE_ERROR;
	}
	memset((*queue)->data, 0, data_buffer_size);
	MPI_Get_address((*queue)->data, &(*queue)->datadisp_local);

	(*queue)->data_ptr = 0;
	(*queue)->data_size = size_per_node;

	MPI_Win_create_dynamic(MPI_INFO_NULL, (*queue)->comm, &(*queue)->win);
	MPI_Win_attach((*queue)->win, *queue, sizeof(rma_nb_queue_t));
	MPI_Win_attach((*queue)->win, (*queue)->data, data_buffer_size);

	(*queue)->state.head_info.raw = UNDEFINED_NODE_INFO;
	(*queue)->state.tail_info.raw = UNDEFINED_NODE_INFO;
	(*queue)->state.head_info_lock = UNDEFINED_RANK;
	(*queue)->state.tail_info_lock = UNDEFINED_RANK;
	MPI_Get_address(&(*queue)->state, &(*queue)->statedisp_local);

	elem_reset(&(*queue)->head);
	MPI_Get_address(&(*queue)->head, &(*queue)->headdisp_local);
	
	elem_reset(&(*queue)->tail);
	MPI_Get_address(&(*queue)->tail, &(*queue)->taildisp_local);

	if (myrank == MAIN_RANK) {
		sentinel_init(&(*queue)->sentinel);
		MPI_Get_address(&(*queue)->sentinel, &(*queue)->sentineldisp);
	}

	if (disps_init(*queue) != CODE_SUCCESS) {
		return CODE_ERROR;
	}

	offsets_init();

	(*queue)->ts_offset = mpi_sync_time((*queue)->comm);
	init_random_generator();

	MPI_Barrier((*queue)->comm);
	return CODE_SUCCESS;
}
void rma_nb_queue_free(rma_nb_queue_t* queue) {
	MPI_Barrier(queue->comm);

	MPI_Free_mem(queue->basedisp);
	MPI_Free_mem(queue->datadisp);
	MPI_Free_mem(queue->statedisp);
	MPI_Free_mem(queue->headdisp);
	MPI_Free_mem(queue->taildisp);
	MPI_Free_mem(queue->data);
	MPI_Free_mem(queue);
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
	// if (target == myrank) {
	// 	*queue_state = queue->state; // mb MPI_get_acc here?
	// 	return CODE_SUCCESS;
	// } else {
		if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, target);
		int op_res = MPI_Get_accumulate(queue_state, sizeof(queue_state_t), MPI_BYTE,
										queue_state, sizeof(queue_state_t), MPI_BYTE,
										target, queue->statedisp[target], sizeof(queue_state_t), MPI_BYTE,
										MPI_NO_OP, queue->win);
		return (op_res == MPI_SUCCESS) ? CODE_SUCCESS : CODE_ERROR;
	// }
}
int get_head_info(rma_nb_queue_t* queue, int target, u_node_info_t* head_info) {
	// if (target == myrank) {
	// 	*head_info = queue->state.head_info;

	// 	if (USE_DEBUG) {
	// 		l_str << "got head info [local] (" << head_info->parsed.rank << ", " << head_info->parsed.position << ": " << head_info->raw << ")" << std::endl;
	// 		log_(l_str);
	// 	}
	// 	return CODE_SUCCESS;
	// } else {
		if (USE_DEBUG) {
			l_str << "trying get head info on rank " << target << ", disp " << MPI_Aint_add(queue->statedisp[target], offsets.qs_head) << std::endl;
			log_(l_str);
		}
		if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, target);

		int op_res = MPI_Get_accumulate(head_info, sizeof(queue_state_t), MPI_BYTE,
										head_info, sizeof(queue_state_t), MPI_BYTE,
										target, MPI_Aint_add(queue->statedisp[target], offsets.qs_head), sizeof(queue_state_t), MPI_BYTE,
										MPI_NO_OP, queue->win);
		MPI_Win_flush(target, queue->win);

		if (USE_DEBUG) {
			l_str << "\tgot head info (" << head_info->parsed.rank << ", " << head_info->parsed.position << ": " << head_info->raw << ")" << std::endl;
			log_(l_str);
		}
		return (op_res == MPI_SUCCESS) ? CODE_SUCCESS : CODE_ERROR;
	// }
}
int get_tail_info(rma_nb_queue_t* queue, int target, u_node_info_t* tail_info) {
	// if (target == myrank) {
	// 	*tail_info = queue->state.tail_info;

	// 	if (USE_DEBUG) {
	// 		l_str << "got tail info [local] (" << tail_info->parsed.rank << ", " << tail_info->parsed.position << ": " << tail_info->raw << ")" << std::endl;
	// 		log_(l_str);
	// 	}
	// 	return CODE_SUCCESS;
	// } else {
		if (USE_DEBUG) {
			l_str << "trying get tail info on rank " << target << ", disp " << MPI_Aint_add(queue->statedisp[target], offsets.qs_tail) << std::endl;
			log_(l_str);
		}
		if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, target);

		int op_res = MPI_Get_accumulate(tail_info, sizeof(queue_state_t), MPI_BYTE,
										tail_info, sizeof(queue_state_t), MPI_BYTE,
										target, MPI_Aint_add(queue->statedisp[target], offsets.qs_tail), sizeof(queue_state_t), MPI_BYTE,
										MPI_NO_OP, queue->win);
		MPI_Win_flush(target, queue->win);

		if (USE_DEBUG) {
			l_str << "\tgot tail info (" << tail_info->parsed.rank << ", " << tail_info->parsed.position << ": " << tail_info->raw << ")" << std::endl;
			log_(l_str);
		}
		return (op_res == MPI_SUCCESS) ? CODE_SUCCESS : CODE_ERROR;
	// }
}
int set_head_info(rma_nb_queue_t* queue, int target, u_node_info_t new_head_info, u_node_info_t prev_head_info) {
	int op_res;
	if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, MAIN_RANK);

	op_res = MPI_Put(&new_head_info, sizeof(u_node_info_t), MPI_BYTE, MAIN_RANK, 
					 MPI_Aint_add(queue->statedisp, offsets.qs_head), sizeof(u_node_info_t), MPI_BYTE, queue->win);
	MPI_Win_flush(MAIN_RANK, queue->win);

	if (USE_DEBUG) {
		l_str << "set head info in target " << MAIN_RANK << " to " << print(new_head_info) << std::endl;
		log_(l_str);
	}
	return op_res;
}
int set_tail_info(rma_nb_queue_t* queue, int target, u_node_info_t new_tail_info, u_node_info_t prev_tail_info) {
	int op_res;
	if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, MAIN_RANK);

	op_res = MPI_Put(&new_tail_info, sizeof(u_node_info_t), MPI_BYTE, MAIN_RANK, 
					 MPI_Aint_add(queue->statedisp, offsets.qs_tail), sizeof(u_node_info_t), MPI_BYTE, queue->win);
	MPI_Win_flush(MAIN_RANK, queue->win);

	if (USE_DEBUG) {
		l_str << "set tail info in target " << MAIN_RANK << " to " << print(new_tail_info) << std::endl;
		log_(l_str);
	}
	return op_res;
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
MPI_Aint get_elem_lock_disp(rma_nb_queue_t* queue, u_node_info_t elem_info) {
	return MPI_Aint_add(get_elem_disp(queue, elem_info), offsets.elem_lock);
}
MPI_Aint get_qs_head_info_lock_disp(rma_nb_queue_t* queue, int target) {
	return MPI_Aint_add(queue->statedisp[target], offsets.qs_head_info_lock);
}
MPI_Aint get_qs_tail_info_lock_disp(rma_nb_queue_t* queue, int target) {
	return MPI_Aint_add(queue->statedisp[target], offsets.qs_tail_info_lock);
}

int get_currently_using_head(rma_nb_queue_t* queue, int target, elem_t* head) {
	int op_res;
	if (USE_DEBUG) {
		l_str << "trying get currently using head on rank " << target << ", disp " << queue->headdisp[target] << std::endl;
		log_(l_str);
	}
	if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, target);

	op_res = MPI_Get_accumulate(head, sizeof(elem_t), MPI_BYTE,
								head, sizeof(elem_t), MPI_BYTE,
								target, queue->headdisp[target], sizeof(elem_t), MPI_BYTE,
								MPI_NO_OP, queue->win);
	MPI_Win_flush(target, queue->win);

	if (USE_DEBUG) {
		l_str << "\t" << op_res << " got currently using head " << print(*head) << std::endl;
		log_(l_str);
	}
	//g_pause();
	return op_res;
}
int get_currently_using_tail(rma_nb_queue_t* queue, int target, elem_t* tail) {
	int op_res;
	if (USE_DEBUG) {
		l_str << "trying get currently using tail on rank " << target << ", disp " << queue->taildisp[target] << std::endl;
		log_(l_str);
	}
	if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, target);

	op_res = MPI_Get_accumulate(tail, sizeof(elem_t), MPI_BYTE,
								tail, sizeof(elem_t), MPI_BYTE,
								target, queue->taildisp[target], sizeof(elem_t), MPI_BYTE,
								MPI_NO_OP, queue->win);
	MPI_Win_flush(target, queue->win);

	if (USE_DEBUG) {
		l_str << "\t" << op_res << " got currently using tail " << print(*tail) << std::endl;
		log_(l_str);
	}
	//g_pause();
	return op_res;
}
double get_min_using_ts(rma_nb_queue_t* queue) {
	double min_using_ts;
	int target;
	rand_provider_t rand_provider;
	rand_provider_init(&rand_provider, queue->n_proc);

	get_elem(queue, queue->state.head_info, &queue->head);
	get_elem(queue, queue->state.tail_info, &queue->tail);
	min_using_ts = std::min(queue->head.ts, queue->tail.ts);
	
	exclude_rank(&rand_provider, myrank);
	while(1) {
		if(USE_DEBUG) {
			l_str << std::setprecision(15) << "current min_using_ts " << min_using_ts << std::endl;
			log_(l_str);
		}

		target = get_next_node_rand(&rand_provider);
		if(target == UNDEFINED_RANK) {
			elem_reset(&queue->head);
			elem_reset(&queue->tail);
			rand_provider_free(&rand_provider);
			return min_using_ts;
		}

		get_head_info(queue, target, &queue->head.info);
		get_tail_info(queue, target, &queue->tail.info);
		if(queue->head.info.raw != UNDEFINED_NODE_INFO) {
			get_elem(queue, queue->head.info, &queue->head);
		} else {
			queue->head.ts = min_using_ts;
		}
		if(queue->tail.info.raw != UNDEFINED_NODE_INFO) {
			get_elem(queue, queue->tail.info, &queue->tail);
		} else {
			queue->tail.ts = min_using_ts;
		}
		min_using_ts = std::min(min_using_ts, std::min(queue->head.ts, queue->tail.ts));

		get_currently_using_head(queue, target, &queue->head);
		get_currently_using_tail(queue, target, &queue->tail);
		if(queue->head.ts == UNDEFINED_TS) queue->head.ts = min_using_ts;
		if(queue->tail.ts == UNDEFINED_TS) queue->tail.ts = min_using_ts;
		min_using_ts = std::min(min_using_ts, std::min(queue->head.ts, queue->tail.ts));
	}
}
void cleaning(rma_nb_queue_t* queue, int from) {
	if(USE_DEBUG) {
		log_("START CLEANING\n");
	}

	double min_ts = get_min_using_ts(queue);
	for(int i = 0; queue->data[from+i].state == NODE_DELETED; ++i) {
		if(queue->data[from+i].ts < min_ts) {
			queue->data[from+i].state = NODE_FREE;
			
			if(USE_DEBUG) {
				l_str << "position " << from + i << " FREED" << std::endl;
				log_(l_str);
			}
		}
	}
	
	if(USE_DEBUG) {
		log_("EXIT CLEANING\n");
	}
}
int get_position(rma_nb_queue_t* queue, int* position) {
	bool made_cleaning = false;
	int start = queue->data_ptr;
	while (1) {
		if (queue->data[queue->data_ptr].state == NODE_FREE) {
			*position = queue->data_ptr;
			queue->data_ptr = (queue->data_ptr + 1) % queue->data_size;
			return CODE_SUCCESS;
		}

		if (!made_cleaning && queue->data[queue->data_ptr].state == NODE_DELETED) {
			cleaning(queue, queue->data_ptr);
			made_cleaning = true;
			continue;
		}

		queue->data_ptr = (queue->data_ptr + 1) % queue->data_size;
		if (queue->data_ptr == start) {
			return CODE_DATA_BUFFER_FULL;
		}
	}

}
int get_elem(rma_nb_queue_t* queue, u_node_info_t elem_info, elem_t* elem) {
	int op_res;
	// if(myrank == elem_info.parsed.rank) {
	// 	*elem = queue->data[elem_info.parsed.position];

	// 	if (USE_DEBUG) {
	// 		l_str << "got elem [local] " << print(*elem) << std::endl;
	// 		log_(l_str);
	// 	}
	// 	op_res = CODE_SUCCESS;
	// } else {
		if (USE_DEBUG) {
			l_str << "trying get elem on rank " << elem_info.parsed.rank << ", disp " << get_elem_disp(queue, elem_info) << std::endl;
			log_(l_str);
		}
		if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, elem_info.parsed.rank);

		op_res = MPI_Get_accumulate(elem, sizeof(elem_t), MPI_BYTE,
									elem, sizeof(elem_t), MPI_BYTE,
									elem_info.parsed.rank, get_elem_disp(queue, elem_info), sizeof(elem_t), MPI_BYTE,
									MPI_NO_OP, queue->win);
		MPI_Win_flush(elem_info.parsed.rank, queue->win);

		if (USE_DEBUG) {
			l_str << "\t" << op_res << " got elem " << print(*elem) << std::endl;
			log_(l_str);
		}
	// }
	//g_pause();
	return op_res;
}
int get_sentinel(rma_nb_queue_t* queue, elem_t* sent) {
	int op_res;
	// if(myrank == MAIN_RANK) {
	// 	*sent = queue->sentinel;

	// 	if (USE_DEBUG) {
	// 		l_str << "got sentinel [local] next(" << sent->next_node_info.parsed.rank << ", " << sent->next_node_info.parsed.position << ": " << sent->next_node_info.raw << ")" << std::endl;
	// 		log_(l_str);
	// 	}
	// 	op_res = CODE_SUCCESS;
	// } else {
		if (USE_DEBUG) {
			l_str << "trying get sentinel on disp " << queue->sentineldisp << std::endl;
			log_(l_str);
		}
		if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, MAIN_RANK);

		op_res = MPI_Get_accumulate(sent, sizeof(elem_t), MPI_BYTE,
									sent, sizeof(elem_t), MPI_BYTE,
									MAIN_RANK, queue->sentineldisp, sizeof(elem_t), MPI_BYTE,
									MPI_NO_OP, queue->win);
		MPI_Win_flush(MAIN_RANK, queue->win);

		if (USE_DEBUG) {
			l_str << "\tgot sentinel next(" << sent->next_node_info.parsed.rank << ", " << sent->next_node_info.parsed.position << ": " << sent->next_node_info.raw << ")" << std::endl;
			log_(l_str);
		}
	// }
	return op_res;
}
// TODO
int set_sentinel_next_node_info(rma_nb_queue_t* queue, u_node_info_t sentinel_info, u_node_info_t next_node_info, u_node_info_t expected_node_info) {
	bool op_res;
	if (USE_DEBUG) {
		l_str << "trying to set sentinel's next node info to " << print(next_node_info) << ", expected " << print(expected_node_info) << std::endl;
		log_(l_str);
	}
	if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, sentinel_info.parsed.rank);

	op_res = CAS(&next_node_info, &expected_node_info, sentinel_info.parsed.rank,
				 MPI_Aint_add(queue->sentineldisp, offsets.elem_next_node_info), queue->win);

	if (USE_DEBUG) {
		l_str << "\t" << (op_res ? "succeed" : "failed") << " to set sentinel's next node info to " << print(next_node_info) << std::endl;
		log_(l_str);
	}
	return op_res;
}
int set_next_node_info(rma_nb_queue_t* queue, u_node_info_t target_elem_info, u_node_info_t next_node_info, u_node_info_t expected_node_info) {
	bool op_res;
	if (USE_DEBUG) {
		l_str << "trying to set elem " << print(target_elem_info) << " next node info to " << print(next_node_info) << ", expected " << print(expected_node_info) << std::endl;
		log_(l_str);
	}
	if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, target_elem_info.parsed.rank);

	op_res = CAS(&next_node_info, &expected_node_info, target_elem_info.parsed.rank,
				 get_elem_next_node_info_disp(queue, target_elem_info), queue->win);

	if (USE_DEBUG) {
		l_str << "\t" << (op_res ? "succeed" : "failed") << " to set elem " << print(target_elem_info) << " next node info to " << print(next_node_info) << std::endl;
		log_(l_str);
	}
	return op_res;
}
int set_state(rma_nb_queue_t* queue, u_node_info_t target_elem_info, int state, int expected_state) {
	bool op_res;
	if (USE_DEBUG) {
		l_str << "trying to set elem " << print(target_elem_info) << " state to " << state << ", expected " << expected_state << std::endl;
		log_(l_str);
	}
	if (USE_MPI_CALLS_COUNTING) count(&mpi_call_counter, target_elem_info.parsed.rank);

	op_res = CAS(&state, &expected_state, target_elem_info.parsed.rank,
				 get_elem_state_disp(queue, target_elem_info), queue->win);

	if (USE_DEBUG) {
		l_str << "\t" << (op_res ? "succeed" : "failed") << " to set elem " << print(target_elem_info) << " state to " << state << std::endl;
		log_(l_str);
	}
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
	// int i = rand() % n_proc;
	int i = get_rand(n_proc);
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
		if (meta->head_info.raw != UNDEFINED_NODE_INFO) {
			get_elem(queue, meta->head_info, &meta->head);
			if (elem->ts < meta->head.ts) return CODE_ERROR;
		}
	} while (set_head_info(queue, target, elem->info, meta->head_info) != CODE_SUCCESS);

	return CODE_SUCCESS;
}
int bcastpart_tail_info(rma_nb_queue_t* queue, int target, elem_t* elem, bcast_meta_t* meta) {
	do {
		get_tail_info(queue, target, &meta->tail_info);
		if (meta->tail_info.raw != UNDEFINED_NODE_INFO) {
			get_elem(queue, meta->tail_info, &meta->tail);
			if (elem->ts < meta->tail.ts) return CODE_ERROR;
		}
	} while (set_tail_info(queue, target, elem->info, meta->tail_info) != CODE_SUCCESS);

	return CODE_SUCCESS;
}
int bcastpart_head_tail_info(rma_nb_queue_t* queue, int target, elem_t* elem, bcast_meta_t* meta) {
	if(meta->should_update_head) {
		do {
			get_head_info(queue, target, &meta->head_info);
			if(meta->head_info.raw != UNDEFINED_NODE_INFO) {
				get_elem(queue, meta->head_info, &meta->head);
				if (elem->ts < meta->head.ts) {
					meta->should_update_head = false;
					break;
				}
			}
			if (set_head_info(queue, target, elem->info, meta->head_info) == CODE_SUCCESS) break;
		} while (1);
	}

	if(meta->should_update_tail) {
		do {
			get_tail_info(queue, target, &meta->tail_info);
			if (meta->tail_info.raw != UNDEFINED_NODE_INFO) {
				get_elem(queue, meta->tail_info, &meta->tail);
				if (elem->ts < meta->tail.ts) {
					meta->should_update_tail = false;
					break;
				}
			}
			if (set_tail_info(queue, target, elem->info, meta->tail_info) == CODE_SUCCESS) break;
		} while (1);
	}

	if(!meta->should_update_head && !meta->should_update_tail) return CODE_ERROR;

	return CODE_SUCCESS;
}
int bcastpart_tail_head_info(rma_nb_queue_t* queue, int target, elem_t* elem, bcast_meta_t* meta) {
	if(meta->should_update_tail) {
		do {
			get_tail_info(queue, target, &meta->tail_info);
			if (meta->tail_info.raw != UNDEFINED_NODE_INFO) {
				get_elem(queue, meta->tail_info, &meta->tail);
				if (elem->ts < meta->tail.ts) {
					meta->should_update_tail = false;
					break;
				}
			}
			if (set_tail_info(queue, target, elem->info, meta->tail_info) == CODE_SUCCESS) break;
		} while (1);
	}

	if(meta->should_update_head) {
		do {
			get_head_info(queue, target, &meta->head_info);
			if(meta->head_info.raw != UNDEFINED_NODE_INFO) {
				get_elem(queue, meta->head_info, &meta->head);
				if (elem->ts < meta->head.ts) {
					meta->should_update_head = false;
					break;
				}
			}
			if (set_head_info(queue, target, elem->info, meta->head_info) == CODE_SUCCESS) break;
		} while (1);
	}

	if(!meta->should_update_head && !meta->should_update_tail) return CODE_ERROR;

	return CODE_SUCCESS;
}
int bcast_node_info_template(rma_nb_queue_t* queue, elem_t* elem, int priority_rank, bcast_meta_t* meta) {
	int op_res;
	int target_node;
	rand_provider_t provider;

	rand_provider_init(&provider, queue->n_proc);

	if (USE_DEBUG) {
		l_str << "bcasting elem " << print(*elem) << std::endl;
		log_(l_str);
	}

	if (priority_rank != myrank) {	// send to priority rank, if priority is myrank - skip this step
		if (USE_DEBUG) {
			l_str << "bcasting to priority " << priority_rank << std::endl;
			log_(l_str);
		}
		op_res = meta->bcast_method(queue, priority_rank, elem, meta);
		if(op_res == CODE_ERROR) {
			rand_provider_free(&provider);
			return CODE_PARTLY_SUCCESS;
		}
		exclude_rank(&provider, priority_rank);
	}

	if (USE_DEBUG) {
		l_str << "bcasting to me\n";
		log_(l_str);
	}
	op_res = meta->bcast_method(queue, myrank, elem, meta);	// send to me
	if(op_res == CODE_ERROR) {
		rand_provider_free(&provider);
		return CODE_PARTLY_SUCCESS;
	}
	exclude_rank(&provider, myrank);

	while (1) {
		target_node = get_next_node_rand(&provider);
		if (USE_DEBUG) {
			l_str << "bcasting to " << target_node << std::endl;
			log_(l_str);
		}
		if (target_node == UNDEFINED_RANK) {
			rand_provider_free(&provider);
			return CODE_SUCCESS;
		}

		op_res = meta->bcast_method(queue, target_node, elem, meta); // send to target_node
		if(op_res == CODE_ERROR) {
			rand_provider_free(&provider);
			return CODE_PARTLY_SUCCESS;
		}
	}
}

int bcast_head_info(rma_nb_queue_t* queue, elem_t elem, int priority_rank) {
	bcast_meta_t meta;
	bcast_meta_init(&meta);
	meta.bcast_method = bcastpart_head_info;

	if (USE_DEBUG) {
		l_str << "START BCAST (H)" << std::endl;
		log_(l_str);
	}
	int op_res = bcast_node_info_template(queue, &elem, priority_rank, &meta);
	if (USE_DEBUG) {
		l_str << "EXIT BCAST with code " << op_res << std::endl;
		log_(l_str);
	}
	return op_res;
}
int bcast_tail_info(rma_nb_queue_t* queue, elem_t elem, int priority_rank) {
	bcast_meta_t meta;
	bcast_meta_init(&meta);
	meta.bcast_method = bcastpart_tail_info;

	if (USE_DEBUG) {
		l_str << "START BCAST (T)" << std::endl;
		log_(l_str);
	}
	int op_res = bcast_node_info_template(queue, &elem, priority_rank, &meta);
	if (USE_DEBUG) {
		l_str << "EXIT BCAST with code " << op_res << std::endl;
		log_(l_str);
	}
	return op_res;
}
int bcast_head_tail_info(rma_nb_queue_t* queue, elem_t elem, int priority_rank) {
	bcast_meta_t meta;
	bcast_meta_init(&meta);
	meta.bcast_method = bcastpart_head_tail_info;

	if (USE_DEBUG) {
		l_str << "START BCAST (H|T)" << std::endl;
		log_(l_str);
	}
	int op_res = bcast_node_info_template(queue, &elem, priority_rank, &meta);
	if (USE_DEBUG) {
		l_str << "EXIT BCAST with code " << op_res << std::endl;
		log_(l_str);
	}
	return op_res;
}
int bcast_tail_head_info(rma_nb_queue_t* queue, elem_t elem, int priority_rank) {
	bcast_meta_t meta;
	bcast_meta_init(&meta);
	meta.bcast_method = bcastpart_tail_head_info;

	if (USE_DEBUG) {
		l_str << "START BCAST (T|H)" << std::endl;
		log_(l_str);
	}
	int op_res = bcast_node_info_template(queue, &elem, priority_rank, &meta);
	if (USE_DEBUG) {
		l_str << "EXIT BCAST with code " << op_res << std::endl;
		log_(l_str);
	}
	return op_res;
}

int elem_init(rma_nb_queue_t* queue, elem_t** elem, val_t value) {
	int position;
	int op_res = get_position(queue, &position);
	if (op_res == CODE_DATA_BUFFER_FULL) {
		return CODE_DATA_BUFFER_FULL;
	}

	if (USE_DEBUG) {
		l_str << "chosen position " << position << std::endl;
		log_(l_str);
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

	u_node_info_t undefined_node_info;
	undefined_node_info.raw = UNDEFINED_NODE_INFO;

	elem_t *new_elem;
	elem_t sentinel;

	if (USE_DEBUG) {
		l_str << "START ENQUEUE value " << value << std::endl;
		log_(l_str);
	}

	begin_epoch_all(queue->win);

	op_res = elem_init(queue, &new_elem, value);
	if (op_res == CODE_DATA_BUFFER_FULL) {
		end_epoch_all(queue->win);
		if (USE_DEBUG) {
			l_str << "EXIT ENQUEUE with code " << CODE_DATA_BUFFER_FULL << std::endl;
			log_(l_str);
		}
		return CODE_DATA_BUFFER_FULL;
	}

	if(USE_DEBUG) {
		l_str << "new_elem " << print(*new_elem) << std::endl;
		log_(l_str);
	}
start:
	if (queue->state.tail_info.raw == UNDEFINED_NODE_INFO) {
		
	}

	get_elem(queue, queue->state.tail_info, &queue->tail);
	// g_pause();

	while (1) {
		if (queue->tail.state == NODE_ACQUIRED) {
			
		}

		if (queue->tail.state == NODE_DELETED) {
			
		}

		goto start;
	}
}
int dequeue(rma_nb_queue_t* queue, val_t* value) {
	int op_res;

	int node_state_free = NODE_FREE;
	int node_state_acquired = NODE_ACQUIRED;
	int node_state_deleted = NODE_DELETED;

	elem_t sentinel;

	if (USE_DEBUG) log_("START DEQUEUE\n");

	begin_epoch_all(queue->win);

start:
	if (queue->state.head_info.raw == UNDEFINED_NODE_INFO) {
		get_sentinel(queue, &sentinel);
		if (sentinel.next_node_info.raw == UNDEFINED_NODE_INFO) {
			end_epoch_all(queue->win);
			if (USE_DEBUG) {
				l_str << "EXIT ENQUEUE with code " << CODE_QUEUE_EMPTY << std::endl;
				log_(l_str);
			}
			return CODE_QUEUE_EMPTY;
		}
		queue->state.head_info = sentinel.next_node_info; // is not safe?
	}

	get_elem(queue, queue->state.head_info, &queue->head);

	while (1) {
		if (queue->head.state == NODE_ACQUIRED) {
			if (set_state(queue, queue->head.info, node_state_deleted, node_state_acquired)) {
				*value = queue->head.value;
				get_elem(queue, queue->head.info, &queue->head); // refresh same head
				if (queue->head.next_node_info.raw != UNDEFINED_NODE_INFO) {
					elem_t next_head;
					get_elem(queue, queue->head.next_node_info, &next_head);
					bcast_head_info(queue, next_head, myrank);
				}
				elem_reset(&queue->head);
				end_epoch_all(queue->win);
				if (USE_DEBUG) {
					l_str << "EXIT DEQUEUE with code " << CODE_SUCCESS << ", value is " << *value << std::endl;
					log_(l_str);
				}
				return CODE_SUCCESS;
			}
			get_elem(queue, queue->head.info, &queue->head); // refresh same head
		}

		if (queue->head.state == NODE_DELETED) {
			if (queue->head.next_node_info.raw != UNDEFINED_NODE_INFO) {
				get_elem(queue, queue->head.next_node_info, &queue->head); // move next
				continue;
			} else {
				elem_reset(&queue->head);
				end_epoch_all(queue->win);
				if (USE_DEBUG) {
					l_str << "EXIT DEQUEUE with code " << CODE_QUEUE_EMPTY << std::endl;
					log_(l_str);
				}
				return CODE_QUEUE_EMPTY;
			}
		}

		goto start; // head became free, redo algorithm
	}
}

unsigned long long get_actual_size(rma_nb_queue_t* queue) {
	unsigned long long size = 0;
	elem_t elem;

	begin_epoch_all(queue->win);

	get_head_info(queue, myrank, &elem.next_node_info);
	if (elem.next_node_info.raw != UNDEFINED_NODE_INFO) {
		do {
			++size;
			get_elem(queue, elem.next_node_info, &elem);
		} while (elem.next_node_info.raw != UNDEFINED_NODE_INFO);

		if(elem.state == NODE_DELETED) --size;
	}

	end_epoch_all(queue->win);
	return size;	
}

std::string print(u_node_info_t info) {
	std::stringstream s;
	s << "(" << info.parsed.rank << ", " << info.parsed.position << /*": " << info.raw <<*/ ")";
	return s.str();
}
std::string print(queue_state_t queue_state) {
	std::ostringstream s;
	s	<< "head " << print(queue_state.head_info)
		<< "\ttail " << print(queue_state.tail_info);
	return s.str();
}
std::string print(elem_t elem) {
	std::ostringstream s;
	s	<< std::setprecision(20)
		<< elem.ts << "\t"
		<< elem.value << "\t"
		<< elem.state << "\tcurrent "
		<< print(elem.info) << "\tnext "
		<< print(elem.next_node_info);
	return s.str();
}
std::string print(rma_nb_queue_t* queue) {
	std::ostringstream s;

	int total_elem = 0;
	elem_t elem;
	u_node_info_t next_node_info;

	s << "queue:" << std::endl;

	begin_epoch_all(queue->win);

	get_head_info(queue, myrank, &next_node_info);
	if (next_node_info.raw != UNDEFINED_NODE_INFO) {
		do {
			++total_elem;
			get_elem(queue, next_node_info, &elem);
			s << "\t" << print(elem) << std::endl;
			next_node_info = elem.next_node_info;
		} while (next_node_info.raw != UNDEFINED_NODE_INFO);
	}
	s << "total: " << total_elem << std::endl;

	end_epoch_all(queue->win);
	return s.str();
}
std::string print_from_sentinel(rma_nb_queue_t* queue) {
	std::ostringstream s;

	int total_elem = 0;
	elem_t elem;
	u_node_info_t next_node_info;

	begin_epoch_all(queue->win);

	get_sentinel(queue, &elem);
	next_node_info = elem.next_node_info;
	if (next_node_info.raw != UNDEFINED_NODE_INFO) {
		do {
			++total_elem;
			get_elem(queue, next_node_info, &elem);
			s << print(elem) << std::endl;
			next_node_info = elem.next_node_info;
		} while (next_node_info.raw != UNDEFINED_NODE_INFO);
	}
	s << "total: " << total_elem << std::endl;

	end_epoch_all(queue->win);
	return s.str();
}
std::string print_attributes(rma_nb_queue_t* queue) {
	std::ostringstream s;

	s << "queue attributes:" << std::endl;

	s << "\twindow:\t\t" << queue->win << std::endl;

	s << "\tbase_dl:\t" << queue->basedisp_local << std::endl;
	s << "\tbase_ds:\t";
	for (int i = 0; i < queue->n_proc; ++i) {
		s << queue->basedisp[i] << " ";
	}

	s << "\n\tdata_dl:\t" << queue->datadisp_local << std::endl;
	s << "\tdata_ds:\t";
	for (int i = 0; i < queue->n_proc; ++i) {
		s << queue->datadisp[i] << " ";
	}
	s << "\n\tdata:\t\t" << queue->data << std::endl;
	s << "\tdata_ptr:\t" << queue->data_ptr << std::endl;
	s << "\tdata_size:\t" << queue->data_size << std::endl;

	s << "\tstate_dl:\t" << queue->statedisp_local << std::endl;
	s << "\tstate_ds:\t";
	for (int i = 0; i < queue->n_proc; ++i) {
		s << queue->statedisp[i] << " ";
	}
	s << "\n\tqueue_state:\t" << print(queue->state) << std::endl;

	s << "\tsent_d:\t\t" << queue->sentineldisp << std::endl;
	if (myrank == MAIN_RANK) {
		s << "\tsentinel:\n\t\t" << print(queue->sentinel) << std::endl;
	}

	s << "\thead_dl:\t" << queue->headdisp_local << std::endl;
	s << "\thead_ds:\t";
	for (int i = 0; i < queue->n_proc; ++i) {
		s << queue->headdisp[i] << " ";
	}
	s << "\n\thead:\t" << print(queue->head) << std::endl;

	s << "\ttail_dl:\t" << queue->taildisp_local << std::endl;
	s << "\ttail_ds:\t";
	for (int i = 0; i < queue->n_proc; ++i) {
		s << queue->taildisp[i] << " ";
	}
	s << "\n\ttail:\t" << print(queue->tail) << std::endl;

	s << "\tcomm:\t" << queue->comm << std::endl;
	s << "\tn_proc:\t" << queue->n_proc << std::endl;
	s << "\tts_offset:\t" << queue->ts_offset << std::endl;

	return s.str();
}
std::string print(rand_provider_t* provider) {
	std::ostringstream s;

	s << "{ ";
	for (int i = 0; i < provider->n_proc; ++i) {
		s << provider->nodes[i] << ' ';
	}
	s << "} " << provider->n_proc << std::endl;

	return s.str();
}
std::string print(mpi_call_counter_t* counter) {
	std::ostringstream s;

	s << "mpi calls to processes:" << std::endl;
	for (int i = 0; i < counter->n; ++i) {
		s << '\t' << i << ": " << counter->to[i] << std::endl;
	}

	return s.str();
}
void file_print(rma_nb_queue_t* queue, const char* path) {
	std::ofstream file;
	file.open(path);
	if (!file.is_open()) {
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

void submit_hostname(MPI_Comm comm) {
	const int host_name_size = 20;
	char host_name[host_name_size];
	int comm_size;

	MPI_Comm_size(comm, &comm_size);
	gethostname(host_name, host_name_size);
	for(int i = 0; i < comm_size; ++i) {
		if(i == myrank) {
			l_str << "running on host " << host_name << std::endl;
			log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_FILE);
		}
		MPI_Barrier(comm);
	}
}
double calc_throughput(int num_of_ops_per_node, int n_proc, double time_taken) {
	return ((double)(num_of_ops_per_node * n_proc)) / time_taken;
}
void calc_and_print_total_mpi_calls(rma_nb_queue_t* queue, mpi_call_counter_t* counter) {
	if(USE_MPI_CALLS_COUNTING == 0) return;
	
	l_str << "made " << print(counter);
	log_(l_str);
	if (myrank == MAIN_RANK) {
		mpi_call_counter_t current;
		mpi_call_counter_t total;
		mpi_call_counter_init(&current, queue->n_proc);
		mpi_call_counter_init(&total, queue->n_proc);
		
		for (int i = 0; i < queue->n_proc; ++i) {
			if (i == myrank) continue;
			MPI_Recv(current.to, queue->n_proc, MPI_INT, i, 
						MPI_ANY_TAG, queue->comm, MPI_STATUS_IGNORE);
			sum(&total, &current);
		}
		sum(&total, counter);

		l_str << "total " << print(&total);
		log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_FILE);
		// mpi_call_counter_free(&total);
	} else {
		MPI_Send(counter->to, queue->n_proc, MPI_INT, MAIN_RANK, 0, queue->comm);
	}
}
void check_results(rma_nb_queue_t* queue, test_result result) {
	if(myrank == MAIN_RANK) {
		bool success;
		int actual_size;
		int expected_size;
		test_result current;
		test_result total = result;

		for(int target = 1; target < queue->n_proc; ++target) {
			MPI_Recv(&current, sizeof(test_result), MPI_BYTE,
					 target, MPI_ANY_TAG, queue->comm, MPI_STATUS_IGNORE);
			total.added += current.added;
			total.deleted += current.deleted;
		}

		expected_size = total.added - total.deleted;
		actual_size = get_actual_size(queue);
		success = expected_size == actual_size;
		l_str << "overall added " << total.added << ", deleted " << total.deleted << std::endl
			<< "\tqueue actual size " << actual_size << ", expected size " << expected_size << std::endl
			<< "\tTEST " << (success ? "OK" : "FAILED") << std::endl;
		log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_FILE);
	} else {
		MPI_Send(&result, sizeof(test_result), MPI_BYTE,
				 MAIN_RANK, 0, queue->comm);
	}
}
void check_using_memory_model(rma_nb_queue_t* queue) {
	int *memory_model;
	int flag;
	MPI_Win_get_attr(queue->win, MPI_WIN_MODEL, &memory_model, &flag);
	l_str << "using " << (*memory_model == MPI_WIN_UNIFIED ? "UNIFIED" : "SEPARATE") << " memory model (" << *memory_model << ", " << flag << ")" << std::endl;
	log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_FILE);
}
void test_wtime_wtick() {
	double wtime = MPI_Wtime();
	double wtick = MPI_Wtick();
	double result1 = wtime * wtick;
	double result2 = result1 + 0.00000000000000001;
	double sub = result2 - result1;
	bool bl = result1 < result2;
	bool beq = result1 == result2;
	//bool bas = AreSame(result1, result2);
	bool bm = result1 > result2;
	l_str << std::setprecision(200)
		<< "wtime - " << wtime
		<< "\n\twtick - " << wtick
		<< "\n\tresult1 - " << result1
		<< "\n\tresult2 - " << result2
		<< "\n\tsub - " << sub
		<< "\n\tr1 < r2 - " << bl
		<< "\n\tr1 == r2 - " << beq
		//<< "\n\tr1 as r2 - " << bas
		<< "\n\tr1 > r2 - " << bm << std::endl;
	log_(l_str);
}
void test_get_next_node_rand() {
	srand(time(0));
	int n_proc = 15;
	int next_node;
	rand_provider_t provider;
	rand_provider_init(&provider, n_proc);

	std::cout << print(&provider);
	exclude_rank(&provider, n_proc / 2);
	std::cout << "Excluded node: " << n_proc / 2 << std::endl;

	do {
		std::cout << print(&provider);
		next_node = get_next_node_rand(&provider);
		std::cout << "Next node: " << next_node << std::endl;
	} while (next_node != UNDEFINED_RANK);

	rand_provider_free(&provider);
}
void test_queue_init(int argc, char** argv) {
	int data_size_per_node = 7;

	rma_nb_queue_t* queue;
	rma_nb_queue_init(&queue, data_size_per_node, MPI_COMM_WORLD);

	MPI_Barrier(queue->comm);
	for (int i = 0; i < queue->n_proc; ++i) {
		if (i == myrank) {
			log_(print_attributes(queue));
		}
		MPI_Barrier(queue->comm);
	}
	MPI_Barrier(queue->comm);

	rma_nb_queue_free(queue);
}
void test_enq_single_proc(int argc, char** argv) {
	int op_res;
	int size_per_node = 50;
	int num_of_enqs = 10;
	int max_value = 1000;
	int value;

	rma_nb_queue_t* queue;
	rma_nb_queue_init(&queue, size_per_node, MPI_COMM_WORLD);

	log_(print_attributes(queue));

	srand(time(0));

	MPI_Barrier(queue->comm);
	for (int i = 0; i < num_of_enqs; ++i) {
		// value = rand() % max_value;
		value = get_rand(max_value);
		op_res = enqueue(queue, value);
		if (op_res == CODE_SUCCESS) {
			l_str << "added " << value << std::endl;
		} else {
			l_str << "failed to add " << value << ", code " << op_res << std::endl;
		}
		log_(l_str);
	}
	MPI_Barrier(queue->comm);

	log_(print(queue));

	rma_nb_queue_free(queue);
}
void test_enq_multiple_proc(int argc, char** argv) {
	int op_res;
	int size_per_node = 50;
	int num_of_enqs = 10;
	int max_value = 1000;
	int added = 0;
	int value;

	rma_nb_queue_t* queue;
	rma_nb_queue_init(&queue, size_per_node, MPI_COMM_WORLD);

	for (int i = 0; USE_DEBUG && i < queue->n_proc; ++i) {
		if (i == myrank) log_(print_attributes(queue));
		MPI_Barrier(queue->comm);
	}

	MPI_Barrier(queue->comm);
	for (int i = 0; i < num_of_enqs; ++i) {
		// value = rand() % max_value;
		value = get_rand(max_value);
		op_res = enqueue(queue, value);
		if (op_res == CODE_SUCCESS) {
			++added;
			l_str << "added " << value << std::endl;
		} else {
			l_str << "failed to add " << value << ", code " << op_res << std::endl;
		}
		log_(l_str);
	}
	MPI_Barrier(queue->comm);

	if(myrank == MAIN_RANK) log_(print(queue));

	MPI_Barrier(queue->comm);
	l_str << "added " << added << " elements\n";
	log_(l_str);

	MPI_Barrier(queue->comm);
	for (int i = 0; USE_DEBUG && i < queue->n_proc; ++i) {
		if (i == myrank) log_(print_attributes(queue));
		MPI_Barrier(queue->comm);
	}

	rma_nb_queue_free(queue);
}
void test_deq_single_proc(int argc, char** argv) {
	int op_res;
	int size_per_node = 50;
	int num_of_enqs = 10;
	int num_of_deqs = 10;
	int max_value = 1000;
	int value;

	rma_nb_queue_t* queue;
	rma_nb_queue_init(&queue, size_per_node, MPI_COMM_WORLD);

	log_(print_attributes(queue));

	MPI_Barrier(queue->comm);
	for (int i = 0; i < num_of_enqs; ++i) {
		// value = rand() % max_value;
		value = get_rand(max_value);
		op_res = enqueue(queue, value);
		if (op_res == CODE_SUCCESS) {
			l_str << "added " << value << std::endl;
		} else {
			l_str << "failed to add " << value << ", code " << op_res << std::endl;
		}
		log_(l_str);
	}
	MPI_Barrier(queue->comm);

	log_(print(queue));
	log_(print_attributes(queue));

	MPI_Barrier(queue->comm);
	for (int i = 0; i < num_of_deqs; ++i) {
		op_res = dequeue(queue, &value);
		if (op_res == CODE_SUCCESS) {
			l_str << "deleted " << value << std::endl;
		} else {
			l_str << "failed to delete, code " << op_res << std::endl;
		}
		log_(l_str);
	}
	MPI_Barrier(queue->comm);

	log_(print(queue));
	log_(print_attributes(queue));

	rma_nb_queue_free(queue);
}
void test_deq_multiple_proc(int argc, char** argv) {
	int op_res;
	int size_per_node = 9;
	int num_of_enqs = 10;
	int num_of_deqs = 10;
	int max_value = 1000;
	int added = 0;
	int deleted = 0;
	int value;

	rma_nb_queue_t* queue;
	rma_nb_queue_init(&queue, size_per_node, MPI_COMM_WORLD);

	for (int i = 0; /*USE_DEBUG &&*/ i < queue->n_proc; ++i) {
		if (i == myrank) log_(print_attributes(queue));
		MPI_Barrier(queue->comm);
	}

	MPI_Barrier(queue->comm);
	for (int i = 0; i < num_of_enqs; ++i) {
		// value = rand() % max_value;
		value = get_rand(max_value);
		op_res = enqueue(queue, value);
		if (op_res == CODE_SUCCESS) {
			++added;
			l_str << "added " << value << std::endl;
		} else {
			l_str << "failed to add " << value << ", code " << op_res << std::endl;
		}
		log_(l_str);
	}
	MPI_Barrier(queue->comm);

	if(myrank == MAIN_RANK) log_(print(queue));

	MPI_Barrier(queue->comm);
	l_str << "added " << added << " elements\n";
	log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_FILE);

	MPI_Barrier(queue->comm);
	for (int i = 0; /*USE_DEBUG &&*/ i < queue->n_proc; ++i) {
		if (i == myrank) log_(print_attributes(queue));
		MPI_Barrier(queue->comm);
	}

	MPI_Barrier(queue->comm);
	for (int i = 0; i < num_of_deqs; ++i) {
		op_res = dequeue(queue, &value);
		if (op_res == CODE_SUCCESS) {
			++deleted;
			l_str << "deleted " << value << std::endl;
		} else {
			l_str << "failed to delete, code " << op_res << std::endl;
		}
		log_(l_str);
	}
	MPI_Barrier(queue->comm);

	if (myrank == MAIN_RANK) log_(print(queue));

	MPI_Barrier(queue->comm);
	l_str << "deleted " << deleted << " elements\n";
	log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_FILE);

	MPI_Barrier(queue->comm);
	for (int i = 0; /*USE_DEBUG &&*/ i < queue->n_proc; ++i) {
		if (i == myrank) log_(print_attributes(queue));
		MPI_Barrier(queue->comm);
	}

	rma_nb_queue_free(queue);
}
void test_enq_deq_multiple_proc(int size_per_node, int num_of_ops_per_node, MPI_Comm comm) {
	log_("STARTING TEST_ENQ_DEQ_M_PROC\n");
	l_str << "size_per_node: " << size_per_node
			<< "\n\tnum_of_ops_per_node: " << num_of_ops_per_node << std::endl;
	log_(l_str);

	val_t value;
	int start, end;
	double result_time;
	int start_own, end_own;
	double result_time_own;

	test_result result;
	result.added = 0;
	result.deleted = 0;

	rma_nb_queue_t* queue;
	rma_nb_queue_init(&queue, size_per_node, comm);

	l_str << "n_proc: " << queue->n_proc
			<< "\n\tts_offset: " << queue->ts_offset << std::endl;
	log_(l_str);

	if(myrank == MAIN_RANK) {
		check_using_memory_model(queue);
		l_str << "number of procs " << queue->n_proc << std::endl;
		log_(l_str, LOG_PRINT_CONSOLE);
	}

	MPI_Barrier(queue->comm);
	for (int i = 0; i < num_of_ops_per_node/2; ++i) {
		if (enqueue(queue, get_rand(10000)) == CODE_SUCCESS) ++result.added;
		
		if(USE_DEBUG) {
			l_str << "added " << result.added << std::endl;
			log_(l_str);
			// g_pause();
		}
	}
	if(USE_DEBUG) {
		l_str << "added " << result.added << std::endl;
		log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_FILE);
	}
	MPI_Barrier(queue->comm);

	if (myrank == MAIN_RANK) {
		// log_(print(queue));
		start = clock();
	}

	MPI_Barrier(queue->comm);
	start_own = clock();
	for (int i = 0; i < num_of_ops_per_node; ++i) {
		if (get_rand(2)) {
			if (enqueue(queue, get_rand(10000)) == CODE_SUCCESS) ++result.added;
		} else {
			if (dequeue(queue, &value) == CODE_SUCCESS) ++result.deleted;
		}

		if(USE_DEBUG) {
			l_str << "added " << result.added << "\tdeleted " << result.deleted << std::endl;
			log_(l_str);
		}
	}
	end_own = clock();
	if (USE_DEBUG) log_("finished adding/deleting\n", LOG_PRINT_CONSOLE | LOG_PRINT_FILE);
	MPI_Barrier(queue->comm);

	if (myrank == MAIN_RANK) {
		end = clock();
		result_time = ((double)(end - start)) / CLOCKS_PER_SEC;
		// log_(print(queue));
	}

	result_time_own = ((double)(end_own - start_own)) / CLOCKS_PER_SEC;

	for(int i = 0; i < queue->n_proc; ++i) {
		if (i == myrank) {
			l_str << "total added\t" << result.added << ",\tdeleted\t" << result.deleted << std::endl;
			log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_FILE);
		}
		MPI_Barrier(queue->comm);
	}

	check_results(queue, result);

	if (myrank == MAIN_RANK) {
		l_str << "time taken: " << result_time << " s\tthroughput: " << calc_throughput(num_of_ops_per_node, queue->n_proc, result_time) << " ops/s" << std::endl;
		log_(l_str, LOG_PRINT_CONSOLE | LOG_PRINT_FILE);
	}

	// l_str << print_attributes(queue) << std::endl;
	l_str << "OWN time taken: " << result_time_own << " s\tthroughput: " << calc_throughput(num_of_ops_per_node, queue->n_proc, result_time_own) << " ops/s" << std::endl;
	log_(l_str);

	calc_and_print_total_mpi_calls(queue, &mpi_call_counter);

	rma_nb_queue_free(queue);
	log_("EXITING TEST_ENQ_DEQ_M_PROC\n");
}
void test_complex(int size_per_node, int num_of_ops_per_node) {
	log_("STARTING TEST_COMPEX\n");
	
	MPI_Comm temp_comm;
	int world_rank;
	int world_size;
	int color;

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	for(int i = 0; i < world_size; ++i) {
		color = 1;
		if (i >= world_rank) color = 0;

		MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &temp_comm);
		if (i >= world_rank) test_enq_deq_multiple_proc(size_per_node, num_of_ops_per_node, temp_comm);
		MPI_Comm_free(&temp_comm);

		if(world_rank == MAIN_RANK) std::cout << "--------------------------------------------------" << std::endl;
		log_("--------------------------------------------------\n");
		MPI_Barrier(MPI_COMM_WORLD);
	}
	myrank = world_rank;

	log_("EXITING TEST_COMPLEX\n");
}
void tests(int argc, char** argv) {
	int size_per_node = 150000;
	int num_of_ops_per_node = 10000;
	int n_proc;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

	// srand(time(0) * myrank);
	log_init(myrank);
	mpi_call_counter_init(&mpi_call_counter, n_proc);

	submit_hostname(MPI_COMM_WORLD);
	// if(myrank == MAIN_RANK) {
	// 	test_get_next_node_rand();
	// }
	// test_queue_init(argc, argv);
	// test_enq_single_proc(argc, argv);
	// test_enq_multiple_proc(argc, argv);
	// test_deq_single_proc(argc, argv);
	// test_deq_multiple_proc(argc, argv);
	// test_wtime_wtick();
	test_enq_deq_multiple_proc(size_per_node, num_of_ops_per_node, MPI_COMM_WORLD);
	// test_complex(size_per_node, num_of_ops_per_node);

	// mpi_call_counter_free(&mpi_call_counter);
	log_close();
	MPI_Finalize();
}

int main(int argc, char** argv) {
	tests(argc, argv);
	return 0;
}
