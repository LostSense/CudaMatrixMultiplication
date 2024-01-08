#include "CUAdditor.h"
#include "kernel.h"
#include <iostream>

CUAdditor::CUAdditor()
{
}

void CUAdditor::Run()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return;
    }
    
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return;
    }
}

void CUAdditor::Run2()
{
    int size = 10;

    int** matt = new int* [size];
    int** matt_a = new int* [size];
    int** matt_b = new int* [size];

    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            matt[i] = new int[size];
            matt_a[i] = new int[size];
            matt_b[i] = new int[size];
        }

    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            matt_a[i][j] = i + j;
            matt_b[i][j] = i * j;
            matt[i][j] = 0;
        }


    TwoDAddWithCuda(matt, matt_a, matt_b, size);
    std::cout << "The matrix are: \n";

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            std::cout << matt[i][j] << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < size; ++i)
    {
        delete[] matt[i];
        delete[] matt_a[i];
        delete[] matt_b[i];
    }

    delete[] matt;
    delete[] matt_a;
    delete[] matt_b;
}

void CUAdditor::Run3()
{
    int size = 10;

    int** matt = new int* [size];
    int** matt_a = new int* [size];
    int** matt_b = new int* [size];

    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            matt[i] = new int[size];
            matt_a[i] = new int[size];
            matt_b[i] = new int[size];
        }

    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            matt_a[i][j] = i + j;
            matt_b[i][j] = i * j;
            matt[i][j] = 0;
        }


    TwoDAddWithCuda2(matt, matt_a, matt_b, size);
    std::cout << "The matrix are: \n";

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            std::cout << matt[i][j] << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < size; ++i)
    {
        delete[] matt[i];
        delete[] matt_a[i];
        delete[] matt_b[i];
    }

    delete[] matt;
    delete[] matt_a;
    delete[] matt_b;
}
