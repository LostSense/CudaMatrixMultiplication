#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
void TwoDAddWithCuda(int** matt, int** matt_a, int** matt_b, int size);
void TwoDAddWithCuda2(int** matt, int** matt_a, int** matt_b, int size);