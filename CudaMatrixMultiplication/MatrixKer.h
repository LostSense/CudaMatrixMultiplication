#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t CopyMatrix(double* dest, const double* source, unsigned int size);
cudaError_t AddMatrix(double* dest, const double* source, unsigned int size);
cudaError_t MultiplyMatrix(const double* leftMat, const double* rightMat, double* outMat, size_t x, size_t y, size_t z);
cudaError_t GetValue(double* src, double& out);
cudaError_t SetValue(double* dest, double value);