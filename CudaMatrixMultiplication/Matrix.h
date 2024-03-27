#pragma once
#include <cstddef>
#include <algorithm>
#include <exception>
#include <execution>
#include <memory>
#include "MatrixKer.h"

namespace NS_Matrix
{
	using namespace std;

	enum class DeviceType
	{
		CPU,
		GPU
	};

	enum class MatrixExecutionPolicy
	{
		SEQUENCED_POLICY,
		PARALLEL_POLICY,
		PARALLEL_UNSEQUENCED_POLICY,
		UNSEQUENCED_POLICY
	};

	template<DeviceType T, std::size_t X, std::size_t Y>
	class Matrix;

	template<size_t X, size_t Y>
	class Matrix<DeviceType::CPU, X, Y>
	{
	public:
		Matrix();
		~Matrix() = default;
		Matrix(const Matrix<DeviceType::CPU, X, Y>& other) noexcept;
		Matrix(const Matrix<DeviceType::GPU, X, Y>& other);
		Matrix(const Matrix<DeviceType::CPU, X, Y>&& other) noexcept;
		Matrix(const Matrix<DeviceType::GPU, X, Y>&& other);
		const Matrix<DeviceType::CPU, X, Y>& operator=(const Matrix<DeviceType::CPU, X, Y>& other) noexcept;
		const Matrix<DeviceType::CPU, X, Y>& operator=(const Matrix<DeviceType::GPU, X, Y>& other);
		const Matrix<DeviceType::CPU, X, Y>& operator=(std::shared_ptr<Matrix<DeviceType::CPU, X, Y>> other);
		const Matrix<DeviceType::CPU, X, Y>& operator=(const Matrix<DeviceType::CPU, X, Y>&& other) noexcept;
		const Matrix<DeviceType::CPU, X, Y>& operator=(const Matrix<DeviceType::GPU, X, Y>&& other);
		const Matrix<DeviceType::CPU, X, Y> operator+=(const Matrix<DeviceType::CPU, X, Y>& other) noexcept;
		const Matrix<DeviceType::CPU, X, Y> operator+=(const Matrix<DeviceType::GPU, X, Y>& other);
		const Matrix<DeviceType::CPU, X, Y> operator+(const Matrix<DeviceType::CPU, X, Y>& other);
		const Matrix<DeviceType::CPU, X, Y> operator+(const Matrix<DeviceType::GPU, X, Y>& other);
		double* GetCArr() noexcept;
		const double* GetConstCArr() const noexcept;
		void SetExecutionPolicy(MatrixExecutionPolicy pol);
		double& operator() (size_t row, size_t col);

		//==================================================================================================================
		// This section for matricies of size 10k +
		//==================================================================================================================
		template<typename ...Args>
		void Add(const shared_ptr<Matrix<DeviceType::CPU, X, Y>> other, Args ...args);
		template<typename ...Args>
		void Add(const shared_ptr<Matrix<DeviceType::GPU, X, Y>> other, Args ...args);
		void Add(const shared_ptr<Matrix<DeviceType::CPU, X, Y>> other);
		void Add(const shared_ptr<Matrix<DeviceType::GPU, X, Y>> other);
	private:
		void InnerCopy(const Matrix<DeviceType::CPU, X, Y>* other);
		void InnerAdd(const Matrix<DeviceType::CPU, X, Y>* other);
		void InnerAddFromGPU(const Matrix<DeviceType::GPU, X, Y>* other);

		
		double arr[X * Y];
		MatrixExecutionPolicy m_policy;

	};

	template<size_t X, size_t Y>
	class Matrix<DeviceType::GPU, X, Y>
	{
	private:
		class ProxyCell;
	public:
		Matrix();
		~Matrix();
		Matrix(const Matrix<DeviceType::CPU, X, Y>& other);
		Matrix(const Matrix<DeviceType::GPU, X, Y>& other);
		Matrix(const Matrix<DeviceType::CPU, X, Y>&& other);
		Matrix(const Matrix<DeviceType::GPU, X, Y>&& other);
		const Matrix<DeviceType::GPU, X, Y>& operator=(const Matrix<DeviceType::CPU, X, Y>& other);
		const Matrix<DeviceType::GPU, X, Y>& operator=(const Matrix<DeviceType::GPU, X, Y>& other);
		const Matrix<DeviceType::GPU, X, Y>& operator=(const Matrix<DeviceType::CPU, X, Y>&& other);
		const Matrix<DeviceType::GPU, X, Y>& operator=(Matrix<DeviceType::GPU, X, Y>&& other);
		const Matrix<DeviceType::GPU, X, Y> operator+=(const Matrix<DeviceType::CPU, X, Y>& other);
		const Matrix<DeviceType::GPU, X, Y> operator+=(const Matrix<DeviceType::GPU, X, Y>& other);
		const Matrix<DeviceType::GPU, X, Y> operator+(const Matrix<DeviceType::CPU, X, Y>& other);
		const Matrix<DeviceType::GPU, X, Y> operator+(const Matrix<DeviceType::GPU, X, Y>& other);
		double* GetCArr() noexcept;
		const double* GetConstCArr() const noexcept;
		ProxyCell operator() (size_t row, size_t col); //rework it. we can't do assigment to double, use proxy.
		
		
		template<typename ...Args>
		void Add(const shared_ptr<Matrix<DeviceType::CPU, X, Y>> other, Args ...args);
		template<typename ...Args>
		void Add(const shared_ptr<Matrix<DeviceType::GPU, X, Y>> other, Args ...args);
		void Add(const shared_ptr<Matrix<DeviceType::CPU, X, Y>> other);
		void Add(const shared_ptr<Matrix<DeviceType::GPU, X, Y>> other);
	private:
		double* Init();

		
		double* arr;

		class ProxyCell
		{
		public:
			ProxyCell(size_t row, size_t column, Matrix<DeviceType::GPU, X, Y>& mat) : m_value(0), m_row(row), m_column(column), m_hostMat(mat)  {}
			const ProxyCell& operator=(double value);
			operator double();
		private:
			double m_value;
			size_t m_row, m_column;
			class Matrix<DeviceType::GPU, X, Y>& m_hostMat;
		};
	};

	//****************************************************************************************************************
	// ----------------------------------------------GPU MATRIX IMPLEMENTATION----------------------------------------
	//****************************************************************************************************************

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::GPU, X, Y>::Matrix() : arr(Init())
	{
		cudaError_t error = cudaMemset(arr, 0, X * Y * sizeof(double));

		if (error != cudaSuccess)
			throw std::exception("Can't reset matrix.");
	}

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::GPU, X, Y>::~Matrix()
	{
		cudaFree(arr);
	}

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::GPU, X, Y>::Matrix(const Matrix<DeviceType::CPU, X, Y>& other) : arr(Init())
	{
		cudaError_t error = cudaMemcpy(arr, other.GetConstCArr(), X * Y * sizeof(double), cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
			throw std::exception("Can't copy from device matrix to host matrix.");
	}
	
	template<size_t X, size_t Y>
	inline Matrix<DeviceType::GPU, X, Y>::Matrix(const Matrix<DeviceType::GPU, X, Y>& other) : arr(Init())
	{
		cudaError_t error = CopyMatrix(arr, other.arr, X * Y);
		if (error != cudaSuccess)
			throw std::exception("Can't copy from device matrix to host matrix.");
	}

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::GPU, X, Y>::Matrix(const Matrix<DeviceType::CPU, X, Y>&& other) : arr(Init())
	{
		cudaError_t error = cudaMemcpy(arr, other.GetCArr(), X * Y * sizeof(double), cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
			throw std::exception("Can't copy from device matrix to host matrix.");
	}

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::GPU, X, Y>::Matrix(const Matrix<DeviceType::GPU, X, Y>&& other)
	{
		arr = other.arr;
		other.arr = nullptr;

		return *this;
	}

	template<size_t X, size_t Y>
	inline double* Matrix<DeviceType::GPU, X, Y>::GetCArr() noexcept
	{
		return arr;
	}
	
	template<size_t X, size_t Y>
	inline const double* Matrix<DeviceType::GPU, X, Y>::GetConstCArr() const noexcept
	{
		return arr;
	}

	template<size_t X, size_t Y>
	inline double* Matrix<DeviceType::GPU, X, Y>::Init()
	{
		double* ptr = nullptr;
		cudaError_t error = cudaMalloc(&ptr, X * Y * sizeof(double));
		if (error != cudaSuccess)
			throw std::exception("Can't allocate memory in device.");

		return ptr;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::GPU, X, Y>& Matrix<DeviceType::GPU, X, Y>::operator=(const Matrix<DeviceType::CPU, X, Y>& other)
	{
		cudaError_t error = cudaMemcpy(arr, other.GetCArr(), X * Y * sizeof(double), cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
			throw std::exception("Can't copy from device matrix to host matrix.");
		return *this;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::GPU, X, Y>& Matrix<DeviceType::GPU, X, Y>::operator=(const Matrix<DeviceType::GPU, X, Y>& other)
	{
		cudaError_t error = CopyMatrix(arr, other.arr, X * Y);
		if (error != cudaSuccess)
			throw std::exception("Can't copy from device matrix to host matrix.");
		return *this;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::GPU, X, Y>& Matrix<DeviceType::GPU, X, Y>::operator=(const Matrix<DeviceType::CPU, X, Y>&& other)
	{
		cudaError_t error = cudaMemcpy(this->arr, other.GetConstCArr(), X * Y * sizeof(double), cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
			throw std::exception("Can't copy from Host matrix to Device matrix.");
		return *this;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::GPU, X, Y>& Matrix<DeviceType::GPU, X, Y>::operator=(Matrix<DeviceType::GPU, X, Y>&& other)
	{
		cudaError_t error = cudaFree(arr);

		if (error != cudaSuccess)
			throw std::exception("Can't free allocated memory.");

		arr = other.arr;
		other.arr = nullptr;

		return *this;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::GPU, X, Y> Matrix<DeviceType::GPU, X, Y>::operator+=(const Matrix<DeviceType::CPU, X, Y>& other)
	{
		double* tempMem;

		//1 - allocate memory
		cudaError_t err = cudaMalloc(&tempMem, X * Y * sizeof(double));
		if (err != cudaSuccess)
			throw std::exception("Can't allocate memory on device!");
		//2 - copy all array to device
		err = cudaMemcpy(tempMem, other.arr, X * Y * sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
			throw std::exception("Can't copy data!");
		//3 - call to the function add matrix.
		if (AddMatrix(arr, tempMem, X * Y) != cudaSuccess)
			throw std::exception("Can't make matrix addiction!");

		cudaFree(tempMem);

		return *this;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::GPU, X, Y> Matrix<DeviceType::GPU, X, Y>::operator+=(const Matrix<DeviceType::GPU, X, Y>& other)
	{
		//3 - call to the function add matrix.
		if (AddMatrix(arr, other.arr, X * Y) != cudaSuccess)
			throw std::exception("Can't make matrix addiction!");

		return *this;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::GPU, X, Y> Matrix<DeviceType::GPU, X, Y>::operator+(const Matrix<DeviceType::CPU, X, Y>& other)
	{
		std::shared_ptr<Matrix<DeviceType::GPU, X, Y>> mat = std::make_shared< Matrix<DeviceType::GPU, X, Y>>();
		*mat = *this;
		*mat += other;

		return *mat;
	}
	
	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::GPU, X, Y> Matrix<DeviceType::GPU, X, Y>::operator+(const Matrix<DeviceType::GPU, X, Y>& other)
	{
		std::shared_ptr<Matrix<DeviceType::GPU, X, Y>> mat = std::make_shared< Matrix<DeviceType::GPU, X, Y>>();
		*mat = *this;
		*mat += other;

		return *mat;
	}

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::GPU, X, Y>::ProxyCell Matrix<DeviceType::GPU, X, Y>::operator() (size_t row, size_t col)
	{
		return ProxyCell(row, col, *this);
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::GPU, X, Y>::ProxyCell& Matrix<DeviceType::GPU, X, Y>::ProxyCell::operator=(double value)
	{
		cudaError_t err = SetValue(m_hostMat.arr + X * m_row + m_column, value);
		if (err != cudaSuccess)
			throw std::exception("Can't assign value!");
		return *this;
	}

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::GPU, X, Y>::ProxyCell::operator double()
	{
		double val = 0; 
		cudaError_t err = GetValue(m_hostMat.arr + X * m_row + m_column, val);
		if (err != cudaSuccess)
			throw std::exception("Can't assign value!");

		return val;
	}

	template<size_t X, size_t Y>
	template<typename ...Args>
	void Matrix<DeviceType::GPU, X, Y>::Add(const shared_ptr<Matrix<DeviceType::CPU, X, Y>> other, Args ...args)
	{
		double* tempMem;

		//1 - allocate memory
		cudaError_t err = cudaMalloc(&tempMem, X * Y * sizeof(double));
		if (err != cudaSuccess)
			throw std::exception("Can't allocate memory on device!");
		//2 - copy all array to device
		err = cudaMemcpy(tempMem, other.arr, X * Y * sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
			throw std::exception("Can't copy data!");
		//3 - call to the function add matrix.
		if (AddMatrix(arr, tempMem, X * Y) != cudaSuccess)
			throw std::exception("Can't make matrix addiction!");

		cudaFree(tempMem);

		Add(args...);
	}

	template<size_t X, size_t Y>
	template<typename ...Args>
	void Matrix<DeviceType::GPU, X, Y>::Add(const shared_ptr<Matrix<DeviceType::GPU, X, Y>> other, Args ...args)
	{
		if (AddMatrix(arr, other->GetCArr(), X * Y) != cudaSuccess)
			throw std::exception("Can't make matrix addiction!");
		

		Add(args...);
	}

	template<size_t X, size_t Y>
	void Matrix<DeviceType::GPU, X, Y>::Add(const shared_ptr<Matrix<DeviceType::CPU, X, Y>> other)
	{
		double* tempMem;

		//1 - allocate memory
		cudaError_t err = cudaMalloc(&tempMem, X * Y * sizeof(double));
		if (err != cudaSuccess)
			throw std::exception("Can't allocate memory on device!");
		//2 - copy all array to device
		err = cudaMemcpy(tempMem, other.arr, X * Y * sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
			throw std::exception("Can't copy data!");
		//3 - call to the function add matrix.
		if (AddMatrix(arr, tempMem, X * Y) != cudaSuccess)
			throw std::exception("Can't make matrix addiction!");

		cudaFree(tempMem);
	}

	template<size_t X, size_t Y>
	void Matrix<DeviceType::GPU, X, Y>::Add(const shared_ptr<Matrix<DeviceType::GPU, X, Y>> other)
	{
		if (AddMatrix(arr, other->GetCArr(), X * Y) != cudaSuccess)
			throw std::exception("Can't make matrix addiction!");
	}


	//****************************************************************************************************************
	// ----------------------------------------------CPU MATRIX IMPLEMENTATION----------------------------------------
	//****************************************************************************************************************

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::CPU, X, Y>::Matrix(const Matrix<DeviceType::CPU, X, Y>& other) noexcept
	{
		InnerCopy(&other);
	}

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::CPU, X, Y>::Matrix(const Matrix<DeviceType::GPU, X, Y>& other)
	{
		cudaError_t error = cudaMemcpy(&arr, other.GetCArr(), X * Y * sizeof(double), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
			throw std::exception("Can't copy from device matrix to host matrix.");
	}

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::CPU, X, Y>::Matrix() : arr(), m_policy(MatrixExecutionPolicy::SEQUENCED_POLICY)
	{
		for (double& n : arr)
			n = 0;
	}

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::CPU, X, Y>::Matrix(const Matrix<DeviceType::CPU, X, Y>&& other) noexcept
	{
		InnerCopy(other);
	}

	template<size_t X, size_t Y>
	inline Matrix<DeviceType::CPU, X, Y>::Matrix(const Matrix<DeviceType::GPU, X, Y>&& other)
	{
		cudaError_t error = cudaMemcpy(&arr, other.GetCArr(), X * Y * sizeof(double), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
			throw std::exception("Can't copy from device matrix to host matrix.");
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::CPU, X, Y>& Matrix<DeviceType::CPU, X, Y>::operator=(const Matrix<DeviceType::CPU, X, Y>& other) noexcept
	{
		InnerCopy(&other);

		return *this;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::CPU, X, Y>& Matrix<DeviceType::CPU, X, Y>::operator=(const Matrix<DeviceType::GPU, X, Y>& other)
	{
		cudaError_t error = cudaMemcpy(arr, other.GetCArr(), X * Y * sizeof(double), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
			throw std::exception("Can't copy from device matrix to host matrix.");
		return *this;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::CPU, X, Y>& Matrix<DeviceType::CPU, X, Y>::operator=(const Matrix<DeviceType::CPU, X, Y>&& other) noexcept
	{
		InnerCopy(&other);

		return *this;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::CPU, X, Y>& Matrix<DeviceType::CPU, X, Y>::operator=(const Matrix<DeviceType::GPU, X, Y>&& other)
	{
		cudaError_t error = cudaMemcpy(arr, other.GetCArr(), X * Y * sizeof(double), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
			throw std::exception("Can't copy from device matrix to host matrix.");
		return *this;
	}

	template<size_t X, size_t Y>
	inline double* Matrix<DeviceType::CPU, X, Y>::GetCArr() noexcept
	{
		return arr;
	}
	
	template<size_t X, size_t Y>
	inline const double* Matrix<DeviceType::CPU, X, Y>::GetConstCArr() const noexcept
	{
		return arr;
	}

	template<size_t X, size_t Y>
	void Matrix<DeviceType::CPU, X, Y>::SetExecutionPolicy(MatrixExecutionPolicy pol)
	{
		m_policy = pol;
	}

	template<size_t X, size_t Y>
	void Matrix<DeviceType::CPU, X, Y>::InnerCopy(const Matrix<DeviceType::CPU, X, Y>* other)
	{
		switch (m_policy)
		{
		case MatrixExecutionPolicy::SEQUENCED_POLICY: copy(execution::seq, begin(other->arr), end(other->arr), begin(arr));
			break;
		case MatrixExecutionPolicy::PARALLEL_POLICY: copy(execution::par, begin(other->arr), end(other->arr), begin(arr));
			break;
		case MatrixExecutionPolicy::PARALLEL_UNSEQUENCED_POLICY: copy(execution::par_unseq, begin(other->arr), end(other->arr), begin(arr));
			break;
		case MatrixExecutionPolicy::UNSEQUENCED_POLICY: copy(execution::unseq, begin(other->arr), end(other->arr), begin(arr));
			break;
		default: copy(execution::seq, begin(other->arr), end(other->arr), begin(arr));
			break;
		}
	}

	template<size_t X, size_t Y>
	void Matrix<DeviceType::CPU, X, Y>::InnerAdd(const Matrix<DeviceType::CPU, X, Y>* other)
	{
		switch (m_policy)
		{
		case MatrixExecutionPolicy::SEQUENCED_POLICY: transform(execution::seq, begin(arr), end(arr), begin(other->arr), begin(arr), [&](double a, double b) { return a + b; });
			break;
		case MatrixExecutionPolicy::PARALLEL_POLICY: transform(execution::par, begin(arr), end(arr), begin(other->arr), begin(arr), [&](double a, double b) { return a + b; });
			break;
		case MatrixExecutionPolicy::PARALLEL_UNSEQUENCED_POLICY: transform(execution::par_unseq, begin(arr), end(arr), begin(other->arr), begin(arr), [&](double a, double b) { return a + b; });
			break;
		case MatrixExecutionPolicy::UNSEQUENCED_POLICY: transform(execution::unseq, begin(arr), end(arr), begin(other->arr), begin(arr), [&](double a, double b) { return a + b; });
			break;
		default: transform(execution::seq, begin(arr), end(arr), begin(other->arr), begin(arr), [&](double a, double b) { return a + b; });
			break;
		}
	}
	
	template<size_t X, size_t Y>
	void Matrix<DeviceType::CPU, X, Y>::InnerAddFromGPU(const Matrix<DeviceType::GPU, X, Y>* other)
	{
		double* tempMem;

		//1 - allocate memory
		cudaError_t err = cudaMalloc(&tempMem, X * Y * sizeof(double));
		if (err != cudaSuccess)
			throw std::exception("Can't allocate memory on device!");
		//2 - copy all array to device
		err = cudaMemcpy(tempMem, arr, X * Y * sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
			throw std::exception("Can't copy data!");
		//3 - call to the function add matrix.
		if (AddMatrix(tempMem, other->GetConstCArr(), X * Y) != cudaSuccess)
			throw std::exception("Can't make matrix addiction!");

		//4 - copy back to the given array.
		cudaMemcpy(arr, tempMem, X * Y * sizeof(double), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
			throw std::exception("Can't copy data!");

		cudaFree(tempMem);
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::CPU, X, Y> Matrix<DeviceType::CPU, X, Y>::operator+=(const Matrix<DeviceType::CPU, X, Y>& other) noexcept
	{
		InnerAdd(&other);

		return *this;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::CPU, X, Y> Matrix<DeviceType::CPU, X, Y>::operator+=(const Matrix<DeviceType::GPU, X, Y>& other)
	{
		InnerAddFromGPU(&other);

		return *this;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::CPU, X, Y> Matrix<DeviceType::CPU, X, Y>::operator+(const Matrix<DeviceType::CPU, X, Y>& other)
	{
		std::shared_ptr<Matrix<DeviceType::CPU, X, Y>> mat = std::make_shared< Matrix<DeviceType::CPU, X, Y>>();
		*mat = *this;
		return *mat += other;
	}

	template<size_t X, size_t Y>
	inline const Matrix<DeviceType::CPU, X, Y> Matrix<DeviceType::CPU, X, Y>::operator+(const Matrix<DeviceType::GPU, X, Y>& other)
	{
		std::shared_ptr<Matrix<DeviceType::CPU, X, Y>> mat = std::make_shared< Matrix<DeviceType::CPU, X, Y>>();
		*mat = *this;
		return *mat += other;
	}

	template<size_t X, size_t Y>
	inline double& Matrix<DeviceType::CPU, X, Y>::operator() (size_t row, size_t col)
	{
		return arr[row * X + col];
	}

	template<size_t X, size_t Y>
	template<typename ...Args>
	void Matrix<DeviceType::CPU, X, Y>::Add(const shared_ptr<Matrix<DeviceType::CPU, X, Y>> other, Args ...args)
	{
		InnerAdd(other.get());
		Add(args...);
	}

	template<size_t X, size_t Y>
	template<typename ...Args>
	void Matrix<DeviceType::CPU, X, Y>::Add(const shared_ptr<Matrix<DeviceType::GPU, X, Y>> other, Args ...args)
	{
		InnerAddFromGPU(other.get());

		Add(args...);
	}

	template<size_t X, size_t Y>
	void Matrix<DeviceType::CPU, X, Y>::Add(const shared_ptr<Matrix<DeviceType::CPU, X, Y>> other)
	{
		InnerAdd(other.get());
	}

	template<size_t X, size_t Y>
	void Matrix<DeviceType::CPU, X, Y>::Add(const shared_ptr<Matrix<DeviceType::GPU, X, Y>> other)
	{
		InnerAddFromGPU(other).get();
	}
}