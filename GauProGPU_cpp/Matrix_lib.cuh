#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "device_functions.h"
#include "Parameters.cuh"
#include<stdio.h>
#include<stdlib.h>
using namespace std;
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)


__global__ void TransposeKernel(const float* A, float* B, const int M, const int N);
void invert_device(float* src_d, float* dst_d, int n);

// Matrices are stored in col-major order:
// M(col, row) = *(M.elements + col * M.height + row)
class Matrix_v2 {
public:
	int height;
	int width;
	float* elements;
	Matrix_v2() { width = 0, height = 0, elements = NULL; }
	Matrix_v2(int h, int w) { width = w, height = h, elements = new float[h*w]; memset(elements, 0, h*w * sizeof(float)); }
	~Matrix_v2() {
		if (elements != NULL)
			delete[] elements;
	}
	void printMat();
	void MatTrsp(Matrix_v2 &res);  //transpose
	void MatTrsp();   //self transpose
	bool MatMul(const Matrix_v2 &A, const Matrix_v2 &B); //multiplication
	bool MatInv(Matrix_v2  &des);  //matrix inverse for this matrix
	void de_singular();
	void operator = (const Matrix_v2 &A) {
		this->width = A.width;
		this->height = A.height;
		if (this->elements != NULL) {
			delete[] this->elements;
		}
		this->elements = new float[width*height];
		for (int i = 0; i < width*height; i++) {
			this->elements[i] = A.elements[i];
		}
	}
};

void Matrix_v2::de_singular() {

	Matrix_v2 C;

	if (C.elements != NULL) {
		delete[] C.elements;
		C.elements = NULL;
	}
	C.height = this->width;
	C.width = this->height;
	C.elements = new float[C.height*C.width];

	for (int i = 0; i < this->height; i++) {
		for (int j = 0; j < this->width; j++) {
			if (i == j) {
				C.elements[i + j*this->height] = this->elements[i + j*this->height] + 1e-4;
			}
			else {
				C.elements[i + j*this->height] = this->elements[i + j*this->height];
			}
		}
		
	}

	*this = C;
}

__global__ void TransposeKernel(const float* A, float* B, const int M, const int N){
	//A: source matrix / vector
	//B: result matirx / vector
	//M: # column (width) of A
	//N: # row (height) of A

	int col = blockIdx.x * gridDim.x + threadIdx.x; // column
	int row = blockIdx.y * gridDim.y + threadIdx.y; // row

	for (int i = col; i < M; i += blockDim.x*gridDim.x) {
		for (int j = row; j < N; j += blockDim.y*gridDim.y) {
			B[j*M + i] = A[i*N + j];
		}
	}
}


void Matrix_v2::printMat() {
	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			printf("%f\t", elements[h + w*this->height]);
		}
		printf("\n");
	}
	printf("\n");
}

void Matrix_v2::MatTrsp(Matrix_v2 &res){
	//res: the matrix storing the result

	//CPU part
	if (res.elements != NULL) {
		delete[] res.elements;
		res.elements = NULL;
	}
	res.height = this->width;
	res.width = this->height;
	res.elements = new float[res.height*res.width];

	//GPU part
	cudaError_t cudaStat;  //Error status for cuda function

	//allocate GPU memory
	int const M = this->width;
	int const N = this->height;

	float*d_t;
	cudaStat = cudaMalloc(&d_t, N*M * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf("device memory allocation failed");
		goto Error;
	}
	cudaStat = cudaMemcpy(d_t, this->elements, N*M * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess) {
		printf("device memory copy failed");
		goto Error;
	}

	float*d_r;
	cudaStat = cudaMalloc(&d_r, N*M * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf("device memory allocation failed");
		goto Error;
	}

	//kernel function
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE, GRID_SIZE);
	TransposeKernel << <dimGrid, dimBlock >> >(d_t, d_r, M, N);

	// Check for any errors launching the kernel
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStat));
		goto Error;
	}

	//get result
	cudaStat = cudaMemcpy(res.elements, d_r, M*N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess) {
		printf("device memory copy failed");
		goto Error;
	}
	
Error:
	cudaFree(d_t);
	cudaFree(d_r);

	return;
}

void Matrix_v2::MatTrsp() {
	//res: the matrix storing the result
	Matrix_v2 res;

	//CPU part
	if (res.elements != NULL) {
		delete[] res.elements;
		res.elements = NULL;
	}
	res.height = this->width;
	res.width = this->height;
	res.elements = new float[res.height*res.width];

	//GPU part
	cudaError_t cudaStat;  //Error status for cuda function

						   //allocate GPU memory
	int const M = this->width;
	int const N = this->height;

	float*d_t;
	cudaStat = cudaMalloc(&d_t, N*M * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf("device memory allocation failed");
		goto Error;
	}
	cudaStat = cudaMemcpy(d_t, this->elements, N*M * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess) {
		printf("device memory copy failed");
		goto Error;
	}

	float*d_r;
	cudaStat = cudaMalloc(&d_r, N*M * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf("device memory allocation failed");
		goto Error;
	}

	//kernel function
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE, GRID_SIZE);
	TransposeKernel << <dimGrid, dimBlock >> >(d_t, d_r, M, N);

	// Check for any errors launching the kernel
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStat));
		goto Error;
	}

	//get result
	cudaStat = cudaMemcpy(res.elements, d_r, M*N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess) {
		printf("device memory copy failed");
		goto Error;
	}

	*this = res;

Error:
	cudaFree(d_t);
	cudaFree(d_r);

	return;
}

bool Matrix_v2::MatMul(const Matrix_v2 &A, const Matrix_v2 &B) {
	//A, B are two matrixes, the result will be stored in *this
	if (A.width != B.height) {
		printf("diemension inconsistency in MatMul\n");
		return EXIT_FAILURE;
	}

	//CPU part
	const int M = B.width;
	const int N = A.height;
	const int L = A.width;

	if (this->elements != NULL) {
		delete[] this->elements;
		this->elements = NULL;
	}
	this->height = N;
	this->width = M;
	this->elements = new float[N*M];

	cudaError_t cudaStat;  //Error status for cuda function
	cublasStatus_t stat;   //Error status for Cublas
	cublasHandle_t handle; //cublass object
	int i, j;
	float* d_A, *d_B, *d_C;  //device pointer for A, B, and C (*this)

	//prepare the GPU memory
	cudacall(cudaMalloc(&d_A, N*L * sizeof(float)));
	
	cudacall(cudaMalloc(&d_B, M*L * sizeof(float)));
	
	cudacall(cudaMemcpy(d_B, B.elements, M*L * sizeof(float), cudaMemcpyHostToDevice));

	cudacall(cudaMalloc(&d_C, M*N * sizeof(float)));

	//prepare the handle
	cublascall(cublasCreate(&handle));
	
	//copy the memory from host to device
	stat = cublasSetMatrix(N, L, sizeof(float), A.elements, N, d_A, N);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("data download failed");
		cudaFree(d_A);
		cublasDestroy(handle);
		return EXIT_FAILURE;
	}

	stat = cublasSetMatrix(L, M, sizeof(float), B.elements, L, d_B, L);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("data download failed");
		cudaFree(d_B);
		cublasDestroy(handle);
		return EXIT_FAILURE;
	}

	const float alpha = 1.0f;
	const float beta = 0.0f;
	
	//api kernel
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, L, &alpha, d_A, N, d_B, L, &beta, d_C, N);

	stat = cublasGetMatrix(N, M, sizeof(float), d_C, N, this->elements, N);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("data upload failed");
		cudaFree(d_C);
		cublasDestroy(handle);
		return EXIT_FAILURE;
	}

	//recycle the GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy(handle);

	return EXIT_SUCCESS;

}

void invert_device(float* src_d, float* dst_d, int n)
{
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));

	int batchSize = 1;

	int *P, *INFO;

	cudacall(cudaMalloc<int>(&P, n * batchSize * sizeof(int)));
	cudacall(cudaMalloc<int>(&INFO, batchSize * sizeof(int)));

	int lda = n;

	float *A[] = { src_d };
	float** A_d;
	cudacall(cudaMalloc<float*>(&A_d, sizeof(A)));
	cudacall(cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice));

	cublascall(cublasSgetrfBatched(handle, n, A_d, lda, P, INFO, batchSize));

	int INFOh = 0;
	cudacall(cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost));

	if (INFOh == n)
	{
		printf("Factorization Failed: Matrix is singular\n");
		fprintf(stderr, "Factorization Failed: Matrix is singular\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	float* C[] = { dst_d };
	float** C_d;
	cudacall(cudaMalloc<float*>(&C_d, sizeof(C)));
	cudacall(cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice));

	cublascall(cublasSgetriBatched(handle, n, A_d, lda, P, C_d, lda, INFO, batchSize));

	cudacall(cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost));

	if (INFOh != 0)
	{
		printf("Inversion Failed: Matrix is singular\n");
		fprintf(stderr, "Inversion Failed: Matrix is singular\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	cudaFree(P), cudaFree(INFO), cublasDestroy_v2(handle);
}

bool Matrix_v2::MatInv(Matrix_v2 &res) {
	//obtain the inverse matrix of *this
	if (this->height != this->width) {
		printf("*this is not square matrix in MatInv\n");
		return EXIT_FAILURE;
	}

	//CPU part
	if (res.elements != NULL) {
		delete[] res.elements;
	}
	res.height = res.width = this->height;
	res.elements = new float[this->height * this->width];

	cudaError_t cudaStat;  //Error status for cuda function
	cublasStatus_t stat;   //Error status for Cublas
	cublasHandle_t handle; //cublass object

	const int M = res.height; //the col and rows of the matrix
	const int num = 1;  //the number of arrays in the group
	int info_host = 0;
	int * info; //the info array, record the success / fail info
	int * pivo;  //the info of LU decomposition
	float *d_t;
	float *d_it;
	float ** mat = new float * [num]; // pointer to *this
	float ** matInv = new float *[num]; //pointer to results
	float ** gpuMat;   
	float **gpuMatInv;//the device pointer of inversed *this

	//prepare the handle

	//prepare the GPU memory
	cudacall(cudaMalloc(&info, sizeof(int)));
	
	cudacall(cudaMalloc(&pivo, sizeof(int) * M));
	
	cudacall(cudaMalloc(&d_t, sizeof(float) * M*M));
	
	cudacall(cudaMalloc(&d_it, sizeof(float) * M*M));
	
	cudacall(cudaMemcpy(d_t, this->elements, sizeof(float)*M*M, cudaMemcpyHostToDevice));

	//api kernel
	invert_device(d_t, d_it, M);
	
	//copy the memory from device to host
	cudacall(cudaMemcpy(res.elements, d_it, M*M * sizeof(float), cudaMemcpyDeviceToHost));

	Matrix_v2 test;
	test.MatMul(*this, res);

	for (int i = 0; i < M; i++) {
		if (test.elements[i*height + i] < 1 - 1e-2 || test.elements[i*height + i] > 1 + 1e-2) {
			printf("Caution: incorrect inversion\n");
			break;
			//cudaDeviceReset();
			//return EXIT_FAILURE;
		}
	}

	cudaFree(d_t);
	cudaFree(d_it);
	cudaFree(info);
	cudaFree(pivo);

	return EXIT_SUCCESS;
}