#include "MatrixKer.h"

__global__ void CopyKernel(double* dest, const double* source, unsigned int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size)
		dest[x] = source[x];
}

__global__ void AddMatrixKernel(double* dest, const double* source, unsigned int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size)
		dest[x] += source[x];
}

cudaError_t CopyMatrix(double* dest, const double* source, unsigned int size)
{
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		return err;
	dim3 threadsPerBlock(256, 1);
	dim3 numOfBlocks(size / threadsPerBlock.x + 1, 1);
	CopyKernel <<<numOfBlocks, threadsPerBlock >>> (dest, source, size);

	return cudaGetLastError();
}

cudaError_t AddMatrix(double* dest, const double* source, unsigned int size)
{
	//0 - make math
	cudaDeviceSynchronize();
	dim3 threadsPerBlock(256, 1);
	dim3 numOfBlocks(size / threadsPerBlock.x + 1, 1, 1);
	AddMatrixKernel << <numOfBlocks, threadsPerBlock>> > (dest, source, size);

	return cudaGetLastError();
}

cudaError_t GetValue(double* src, double& out)
{
	cudaDeviceSynchronize();
	cudaMemcpy(&out, src, sizeof(double), cudaMemcpyDeviceToHost);
	return cudaGetLastError();
}

cudaError_t SetValue(double* dest, double value)
{
	cudaDeviceSynchronize();
	cudaMemcpy(dest, &value, sizeof(double), cudaMemcpyHostToDevice);
	return cudaGetLastError();
}
