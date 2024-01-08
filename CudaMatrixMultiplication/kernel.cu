
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>

#include "kernel.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] += a[i] + b[i];
}

__global__ void TwoDAddKernel(int** c, int** a, int** b)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    c[x][y] = a[x][y] + b[x][y];
}

void TwoDAddWithCuda(int **matt, int** matt_a, int** matt_b, int size)
{
    int** dev_c = nullptr;
    int** dev_a = nullptr;
    int** dev_b = nullptr;

    cudaError_t err = cudaSetDevice(0);

    err = cudaMalloc((void**)&dev_c, size * sizeof(int*));
    err = cudaMalloc((void**)&dev_a, size * sizeof(int*));
    err = cudaMalloc((void**)&dev_b, size * sizeof(int*));

    int** temp = new int*[size];
    int** temp2 = new int*[size];
    int** temp3 = new int*[size];

    for (int i = 0; i < size; ++i)
    {
        err = cudaMalloc((void**)&(temp[i]), size * sizeof(int));
        err = cudaMemcpy(temp[i], matt_a[i], size * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMalloc((void**)&(temp2[i]), size * sizeof(int));
        err = cudaMemcpy(temp2[i], matt_b[i], size * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMalloc((void**)&(temp3[i]), size * sizeof(int));
        err = cudaMemcpy(temp3[i], matt[i], size * sizeof(int), cudaMemcpyHostToDevice);
    }

    err = cudaMemcpy(dev_a, temp, size * sizeof(int*), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_b, temp2, size * sizeof(int*), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_c, temp3, size * sizeof(int*), cudaMemcpyHostToDevice);

    dim3 tpb(size, size);
    TwoDAddKernel<<<1, tpb >>> (dev_c, dev_a, dev_b);
    err = cudaGetLastError();
    err = cudaDeviceSynchronize();

    for (int i = 0; i < size; ++i)
    {
        err = cudaMemcpy(matt[i], temp3[i], size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(temp[i]);
        cudaFree(temp2[i]);
        cudaFree(temp3[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

void TwoDAddWithCuda2(int **matt, int** matt_a, int** matt_b, int size)
{
    int** dev_c = nullptr;
    int** dev_a = nullptr;
    int** dev_b = nullptr;

    cudaError_t err = cudaSetDevice(0);

    err = cudaMalloc((void**)&dev_c, size * sizeof(int*));
    err = cudaMalloc((void**)&dev_a, size * sizeof(int*));
    err = cudaMalloc((void**)&dev_b, size * sizeof(int*));

    int** temp = new int*[size];
    int** temp2 = new int*[size];
    int** temp3 = new int*[size];

    for (int i = 0; i < size; ++i)
    {
        err = cudaMalloc((void**)&(temp[i]), size * sizeof(int));
        err = cudaMemcpy(temp[i], matt_a[i], size * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMalloc((void**)&(temp2[i]), size * sizeof(int));
        err = cudaMemcpy(temp2[i], matt_b[i], size * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMalloc((void**)&(temp3[i]), size * sizeof(int));
        err = cudaMemcpy(temp3[i], matt[i], size * sizeof(int), cudaMemcpyHostToDevice);
    }

    err = cudaMemcpy(dev_a, temp, size * sizeof(int*), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_b, temp2, size * sizeof(int*), cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_c, temp3, size * sizeof(int*), cudaMemcpyHostToDevice);

    dim3 tpb(size, size);
    TwoDAddKernel<<<1, tpb >>> (dev_c, dev_a, dev_b);
    err = cudaGetLastError();
    err = cudaDeviceSynchronize();

    for (int i = 0; i < size; ++i)
    {
        err = cudaMemcpy(matt[i], temp3[i], size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(temp[i]);
        cudaFree(temp2[i]);
        cudaFree(temp3[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaMemset(dev_c, 0, size * sizeof(int));

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
