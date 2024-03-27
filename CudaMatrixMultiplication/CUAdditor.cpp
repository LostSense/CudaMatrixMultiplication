#include "CUAdditor.h"
#include "kernel.h"
#include <iostream>
#include <memory>

#include "Matrix.h"
#include "Timer.h"

using namespace NS_Matrix;
void PrintMatrix(double* arr, size_t row, size_t col);
void PrintCudaMatrix(double* arr, size_t row, size_t col);

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
    int size = 25;

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

void CUAdditor::RunMatrixTests1()
{
    Matrix<DeviceType::CPU, 10, 10> cMat;
    Matrix<DeviceType::CPU, 10, 10> cMat2;
    cMat = cMat2;
    Matrix<DeviceType::GPU, 10, 10> gMat;
    Matrix<DeviceType::GPU, 10, 10> gMat2;
    gMat = gMat2;
}

void CUAdditor::RunMatrixTests2()
{
    std::shared_ptr<Matrix<DeviceType::CPU, 100, 100>> cMat = std::make_shared<Matrix<DeviceType::CPU, 100, 100 >>();
    std::shared_ptr<Matrix<DeviceType::CPU, 100, 100>> cMat2 = std::make_shared<Matrix<DeviceType::CPU, 100, 100 >>();
    std::shared_ptr<Matrix<DeviceType::CPU, 100, 100>> cMat3 = std::make_shared<Matrix<DeviceType::CPU, 100, 100 >>();
    for (int i = 0; i < 100; ++i)
        for (int j = 0; j < 100; ++j)
        {
            (*cMat)(i, j) = i * j;
            (*cMat2)(i, j) = i + j;
        }
   /* std::cout << "Print cMat1\n";
    PrintMatrix(cMat->GetCArr(), 100, 100);
    std::cout << "Print cMat2\n";
    PrintMatrix(cMat2->GetCArr(), 100, 100);*/

    *cMat3 = *cMat + *cMat2;
    /*std::cout << "Print cMat3\n";
    PrintMatrix(cMat3->GetCArr(), 100, 100);*/
    
    std::shared_ptr<Matrix<DeviceType::GPU, 100, 100>> gMat = std::make_shared<Matrix<DeviceType::GPU, 100, 100>>();
    std::shared_ptr<Matrix<DeviceType::GPU, 100, 100>> gMat2 = std::make_shared<Matrix<DeviceType::GPU, 100, 100>>();
    std::shared_ptr<Matrix<DeviceType::GPU, 100, 100>> gMat3 = std::make_shared<Matrix<DeviceType::GPU, 100, 100>>();

    for (int i = 0; i < 100; ++i)
        for (int j = 0; j < 100; ++j)
        {
            (*gMat)(i, j) = i * j;
            (*gMat2)(i, j) = i + j;
        }
    //PrintCudaMatrix(gMat->GetCArr(), 100, 100);
    //PrintCudaMatrix(gMat2->GetCArr(), 100, 100);

    *gMat3 = *gMat + *gMat2;
    //PrintCudaMatrix(gMat3->GetCArr(), 100, 100);

    *cMat3 = *cMat3 + *gMat3;
    //PrintMatrix(cMat3->GetCArr(), 100, 100);

    *gMat3 = *cMat3 + *gMat3;
    //PrintCudaMatrix(gMat3->GetCArr(), 100, 100);
}

void CUAdditor::RunMatrixTests3()
{
    std::shared_ptr<Matrix<DeviceType::CPU, 10000, 10000>> cMat = std::make_shared<Matrix<DeviceType::CPU, 10000, 10000 >>();
    std::shared_ptr<Matrix<DeviceType::CPU, 10000, 10000>> cMat2 = std::make_shared<Matrix<DeviceType::CPU, 10000, 10000 >>();
    std::shared_ptr<Matrix<DeviceType::CPU, 10000, 10000>> cMat3 = std::make_shared<Matrix<DeviceType::CPU, 10000, 10000 >>();
    cMat->SetExecutionPolicy(NS_Matrix::MatrixExecutionPolicy::SEQUENCED_POLICY);
    cMat2->SetExecutionPolicy(NS_Matrix::MatrixExecutionPolicy::SEQUENCED_POLICY);
    cMat3->SetExecutionPolicy(NS_Matrix::MatrixExecutionPolicy::SEQUENCED_POLICY);

    for (int i = 0; i < 10000; ++i)
        for (int j = 0; j < 10000; ++j)
        {
            (*cMat)(i, j) = i * j;
            (*cMat2)(i, j) = i + j;
        }

    Timer tm;
    tm.Start();
    cMat3->Add(cMat, cMat2);
    tm.End();
    std::cout << "Sequenced: \n";
    std::cout << tm;

    cMat->SetExecutionPolicy(NS_Matrix::MatrixExecutionPolicy::PARALLEL_POLICY);
    cMat2->SetExecutionPolicy(NS_Matrix::MatrixExecutionPolicy::PARALLEL_POLICY);
    cMat3->SetExecutionPolicy(NS_Matrix::MatrixExecutionPolicy::PARALLEL_POLICY);
    for (int i = 0; i < 10000; ++i)
        for (int j = 0; j < 10000; ++j)
        {
            (*cMat)(i, j) = i * j;
            (*cMat2)(i, j) = i + j;
        }
    tm.Start();
    cMat3->Add(cMat, cMat2);
    tm.End();
    std::cout << "Parralel: \n";
    std::cout << tm;

    cMat->SetExecutionPolicy(NS_Matrix::MatrixExecutionPolicy::PARALLEL_UNSEQUENCED_POLICY);
    cMat2->SetExecutionPolicy(NS_Matrix::MatrixExecutionPolicy::PARALLEL_UNSEQUENCED_POLICY);
    cMat3->SetExecutionPolicy(NS_Matrix::MatrixExecutionPolicy::PARALLEL_UNSEQUENCED_POLICY);
    for (int i = 0; i < 10000; ++i)
        for (int j = 0; j < 10000; ++j)
        {
            (*cMat)(i, j) = i * j;
            (*cMat2)(i, j) = i + j;
        }
    tm.Start();
    cMat3->Add(cMat, cMat2);
    tm.End();
    std::cout << "Parralel unsequenced: \n";
    std::cout << tm;


    std::shared_ptr<Matrix<DeviceType::GPU, 10000, 10000>> gMat = std::make_shared<Matrix<DeviceType::GPU, 10000, 10000 >>(*cMat);
    std::shared_ptr<Matrix<DeviceType::GPU, 10000, 10000>> gMat2 = std::make_shared<Matrix<DeviceType::GPU, 10000, 10000 >>(*cMat2);
    std::shared_ptr<Matrix<DeviceType::GPU, 10000, 10000>> gMat3 = std::make_shared<Matrix<DeviceType::GPU, 10000, 10000 >>();

    tm.Start();
    *gMat3 = *gMat + *gMat2;
    tm.End();
    std::cout << "GPU : \n";
    std::cout << tm;
}

void CUAdditor::RunMatrixTests4()
{
    std::shared_ptr<Matrix<DeviceType::GPU, 10000, 10000>> gMat = std::make_shared<Matrix<DeviceType::GPU, 10000, 10000 >>();
    std::shared_ptr<Matrix<DeviceType::GPU, 10000, 10000>> gMat2 = std::make_shared<Matrix<DeviceType::GPU, 10000, 10000 >>();
    std::shared_ptr<Matrix<DeviceType::GPU, 10000, 10000>> gMat3 = std::make_shared<Matrix<DeviceType::GPU, 10000, 10000 >>();

    for (int i = 0; i < 10000; ++i)
        for (int j = 0; j < 10000; ++j)
        {
            (*gMat)(i, j) = i * j;
            (*gMat2)(i, j) = i + j;
        }

    Timer tm;
    tm.Start();
    gMat3->Add(gMat, gMat2);
    tm.End();
    std::cout << "GPU : \n";
    std::cout << tm;
}

void PrintMatrix(double *arr, size_t row, size_t col)
{
    std::cout << "The matrix are: \n";

    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            std::cout << arr[row * i + j]<< " ";
        }
        std::cout << std::endl;
    }
}

void PrintCudaMatrix(double *arr, size_t row, size_t col)
{
    std::cout << "The matrix are: \n";
    double val;
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            GetValue(arr + row * i + j, val);
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}