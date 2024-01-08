#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
void AllocateCudaMemory2D(cudaError_t& err, int**& dev_c, int size, int**& dev_a, int**& dev_b, 
	int** temp, int** matt_a, int** temp2, int** matt_b, int** temp3, int** matt);
void TwoDAddWithCuda(int** matt, int** matt_a, int** matt_b, int size);
void TwoDAddWithCuda2(int** matt, int** matt_a, int** matt_b, int size);