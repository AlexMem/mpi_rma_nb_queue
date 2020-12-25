#!/bin/bash
mpicxx -O3 -std=c++11 -o rma_nb_queue_exec.out include/utils.c include/mpigclock.c include/hpctimer.c rma_nb_queue.cpp
