#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "Parameters.cuh"
#include<stdio.h>
using namespace std;

class Matrix;
class Matrix_d;

__global__ void MatMulKernel(Matrix_d *A, Matrix_d *B, Matrix_d *C);
__global__ void MatInvKernel(Matrix_d *A);
__global__ void getA(double *A, const int n, double*flag);
__global__ void getAStart(double *A, const int n, double *ans, double flag);


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
class Matrix {
public:
	int height;
	int width;
	double* elements;
	Matrix() { width = 0, height = 0, elements = NULL; }
	Matrix(int h, int w) { width = w, height = h, elements = new double[h*w]; memset(elements, 0, h*w * sizeof(double)); }
	~Matrix() {
		if (elements != NULL)
			delete[] elements;
	}
	void printMat();
	void MatT();  //transpose
	void MatMul(const Matrix &A, const Matrix &B); //multiplication
	bool MatInv(Matrix* des);  //matrix inverse for this matrix
	void operator = (const Matrix &A) {
		this->width = A.width;
		this->height = A.height;
		if (this->elements != NULL) {
			delete[] this->elements;
		}
		this->elements = new double[width*height];
		for (int i = 0; i < width*height; i++) {
			this->elements[i] = A.elements[i];
		}
	}
};

class Matrix_d{
public:
	int height;
	int width;
	double* elements;
	Matrix_d(){ width = 0, height = 0, elements = NULL; }
	Matrix_d(int h, int w) {
		height = h, width = w;
		size_t size = h * w * sizeof(double);
		cudaMalloc(&elements, size);
	}
	Matrix_d(const Matrix &A) {
		width = A.width, height = A.height;
		size_t size = A.width * A.height * sizeof(double);
		cudaMalloc(&elements, size);
		cudaMemcpy(elements, A.elements, size, cudaMemcpyHostToDevice);
	}
	~Matrix_d() { 
		if(elements!=NULL) 
			cudaFree(elements); 
	}
};

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix_d *A, Matrix_d *B, Matrix_d *C)
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	double Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int r = row; r < C->height; r += blockDim.y*gridDim.y) {
		for (int c = col; c < C->width; c += blockDim.x*gridDim.x) {
			for (int e = 0; e < A->width; ++e)
				Cvalue += A->elements[r * (A->width) + e] * B->elements[e * B->width + c];
			C->elements[r * C->width + c] = Cvalue;
		}
	}
}
//
//__global__ void getA(double *arcs, const int n, double *ans)
//{
//	if (n == 1)
//	{
//		*ans = arcs[0*n + 0];
//	}
//	*ans = 0;
//	double * temp = new double [n*n];
//	for (int i = 0; i<n; i++)
//	{
//		for (int j = 0; j<n - 1; j++)
//		{
//			for (int k = 0; k<n - 1; k++)
//			{
//				temp[j*n + k] = arcs[(j + 1)*n + ((k >= i) ? k + 1 : k)];
//			}
//		}
//		double t = 0;
//		getA(temp, n - 1, &t);
//		if (i % 2 == 0)
//		{
//			*ans += arcs[0 * n + i] * t;
//		}
//		else
//		{
//			*ans -= arcs[0*n + i] * t;
//		}
//	}
//	delete[] temp;
//}

//__global__ void  getAStart(double *arcs, const int n, double *ans, const double flag)
//{
//	if (n == 1)
//	{
//		ans[0*n + 0] = 1;
//		return;
//	}
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//	int i, j, k, t;
//	double *temp = new double [n*n];
//	for (i = row; i<n; i += blockDim.y*gridDim.y)
//	{
//		for (j = col; j<n; j += blockDim.x*gridDim.x)
//		{
//			for (k = 0; k<n - 1; k++)
//			{
//				for (t = 0; t<n - 1; t++)
//				{
//					temp[k*n + t] = arcs[(k >= i ? k + 1 : k)*n + (t >= j ? t + 1 : t)];
//				}
//			}
//
//
//			getA(temp, n - 1, ans+j*n+i);
//			if ((i + j) % 2 == 1)
//			{
//				ans[j*n + i] = -ans[j*n + i];
//			}
//
//			ans[j*n + i] = ans[j*n + i] / flag;
//		}
//	}
//	delete[] temp;
//}

void Matrix::printMat() {
	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			printf("%f\t", elements[h*width + w]);
		}
		printf("\n");
	}
}

void Matrix::MatMul(const Matrix &A, const Matrix &B)
{
	if (A.width != B.height) {
		printf("diemension inconsistency in MatMul\n");
		return;
	}
	// Load this matrix and A to device memory

	Matrix_d At(A);
	Matrix_d *d_A;
	cudaMalloc(&d_A, sizeof(Matrix_d));
	cudaMemcpy(d_A, &At, sizeof(Matrix_d), cudaMemcpyHostToDevice);

	Matrix_d Bt(B);
	Matrix_d *d_B;
	cudaMalloc(&d_B, sizeof(Matrix_d));
	cudaMemcpy(d_B, &Bt, sizeof(Matrix_d), cudaMemcpyHostToDevice);

	// Allocate C in device memory
	Matrix_d Ct(*this);
	Matrix_d *d_C;
	cudaMalloc(&d_C, sizeof(Matrix_d));
	cudaMemcpy(d_C, &Ct, sizeof(Matrix_d), cudaMemcpyHostToDevice);

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE, GRID_SIZE);
	MatMulKernel << <dimGrid, dimBlock >> >(d_A, d_B, d_C);

	// Read C from device memory
	size_t size = this->width * this->height * sizeof(double);
	cudaMemcpy(&Ct, d_C, sizeof(Matrix_d), cudaMemcpyDeviceToHost);
	cudaMemcpy(this->elements, Ct.elements, size, cudaMemcpyDeviceToHost);

}

void Matrix::MatT() {
	Matrix C;
	C.elements = new double[this->height*this->width];
	for (int i = 0; i < this->height; i++) {
		for (int j = 0; j < this->width; j++) {
			C.elements[j*this->height + i] = this->elements[i*this->width + j];
		}
	}
	C.height = this->width;
	C.width = this->height;

	*this = C;
}

//bool Matrix::MatInv(Matrix* des) {
//	if (this->height != this->width) {
//		printf("this matrix is not square matrix\n");
//		return false;
//	}
//	//prepare the GPU memory
//	Matrix_d src(*this);
//	Matrix_d *d_src;
//	cudaMalloc(&d_src, sizeof(Matrix_d));
//	cudaMemcpy(d_src, &src, sizeof(Matrix_d), cudaMemcpyHostToDevice);
//
//	double flag = 0;
//	double* d_flag;
//	cudaMalloc(&d_flag, sizeof(double));
//	cudaMemcpy(d_flag, &flag, sizeof(double), cudaMemcpyHostToDevice);
//
//	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//	dim3 dimGrid(GRID_SIZE, GRID_SIZE);
//	getA<<<dimGrid, dimBlock >>>(d_src->elements, this->height, d_flag);
//	cudaMemcpy(&flag, d_flag, sizeof(double), cudaMemcpyDeviceToHost);
//	double *des = new double [this->height*this->height];
//	if (flag == 0)
//	{
//		return false;
//	}
//	else
//	{
//		des->height = des->width = this->height;
//		double *d_des;
//		cudaMalloc(&d_des, sizeof(double));
//		getAStart << <dimGrid, dimBlock >> >(d_src->elements, this->height, d_des, flag);
//		cudaMemcpy(&des->elements, d_des, des->height*des->height * sizeof(double), cudaMemcpyDeviceToHost);
//		/*for (int i = 0; i<n; i++)
//		{
//			for (int j = 0; j<n; j++)
//			{
//				des[i][j] = t[i][j] / flag;
//			}
//
//		}*/
//
//	}
//
//
//	return true;
//}