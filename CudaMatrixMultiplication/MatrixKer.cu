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

__global__ void MultiplyMatrixKernel(const double* leftMat, const double* rightMat, double* out, size_t x, size_t y, size_t z)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < x && col < z)
	{
		for (int i = 0; i < y; ++i)
			out[row * x + col] += leftMat[row * x + i] * rightMat[i * z + col];
	}
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

cudaError_t MultiplyMatrix(const double* leftMat, const double* rightMat, double* outMat, size_t x, size_t y, size_t z)
{
	dim3 threadsPerBlock(16, 16);
	dim3 numOfBlocks(x / threadsPerBlock.x + 1, z / threadsPerBlock.y + 1, 1);
	MultiplyMatrixKernel << <numOfBlocks, threadsPerBlock >> > (leftMat, rightMat, outMat, x, y, z);
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
