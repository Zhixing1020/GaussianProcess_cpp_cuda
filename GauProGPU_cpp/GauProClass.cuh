#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "Parameters.cuh"
#include "Matrix_lib.cuh"
#include <stdio.h>
#include <vector>
using namespace std;

class GPR {
private:

	Matrix_v2 Kff;
	Matrix_v2 Kyy;
	Matrix_v2 Kfy;
	Matrix_v2 Kff_inv;
public:
	Matrix_v2 train_x;
	Matrix_v2 train_y;
	Matrix_v2 test_x;
	Matrix_v2 mu;
	Matrix_v2 cov;
	bool is_fit = 0;
	float L = c_L;
	float sigma_f = c_sf;
	int blockNum = 1024;
	int threadNum = 32;
	void fit(const vector<float> &X, const vector<float> &Y, float L = c_L, float sigma_f = c_sf);
	bool predict(const vector<float> &X_);
	bool Kernel(const Matrix_v2 &A, const Matrix_v2 &B, Matrix_v2 &C);

	void output2file();
};

void GPR::fit(const vector<float>&X, const vector<float> &Y, float L, float sigma_f) {
	train_x.elements = new float[X.size()];
	train_y.elements = new float[Y.size()];
	this->L = L;
	this->sigma_f = sigma_f;
	train_x.height = X.size() / dataDim;
	train_x.width = dataDim;
	train_y.height = Y.size() / outDataDim;
	train_y.width = outDataDim;
	int i = 0;
	for (vector<float>::const_iterator iter = X.begin(); iter != X.end(); ++iter, ++i) {
		train_x.elements[i] = *iter;
	}
	i = 0;
	for (vector<float>::const_iterator iter = Y.begin(); iter != Y.end(); ++iter, ++i) {
		train_y.elements[i] = *iter;
	}

	Kernel(train_x, train_x, Kff);
	Kff.de_singular();
	Kff.MatInv(Kff_inv);
}

bool GPR::predict(const vector<float>&X_) {
	test_x.elements = new float[X_.size()];
	test_x.height = X_.size() / dataDim;
	test_x.width = dataDim;
	int i = 0;
	for (vector<float>::const_iterator iter = X_.begin(); iter != X_.end(); ++iter, ++i) {
		test_x.elements[i] = *iter;
	}

	//Kernel(train_x, train_x, Kff);
	Kernel(test_x, test_x, Kyy);
	Kernel(train_x, test_x, Kfy);

	//Kff.de_singular();
	//Kff.MatInv(Kff_inv);

	Matrix_v2 tmp;
	Matrix_v2 KfyT;
	Kfy.MatTrsp(KfyT);
	tmp.MatMul(KfyT, Kff_inv);
	mu.MatMul(tmp, train_y);
	cov.MatMul(tmp, Kfy);
	for (int i = 0; i < cov.height*cov.width; i++) {
		cov.elements[i] = Kyy.elements[i] - cov.elements[i];
	}

	return true;
}

__global__ void GaussianKernel(float *x1, float *x2, float* dist_Matrix,
	float* result, float l, float sigma_f, int m, int n, int dim)
{
	const int tidx = threadIdx.x;
	const int bidx = blockIdx.x;

	for (int i = bidx; i<m; i += gridDim.x)
	{
		for (int j = tidx; j<n; j+=blockDim.x)
		{
			for (int k = 0; k < dim; k ++) {
				dist_Matrix[i+j*m] += pow((x1[i+k*m] - x2[j+k*n]), 2);
			}
			
			result[i+j*m] = pow(sigma_f, 2) * exp(-0.5 * pow(dist_Matrix[i+j*m] / (l), 2));
		}
	}
}

bool GPR::Kernel(const Matrix_v2 &A, const Matrix_v2 &B, Matrix_v2 &C) {
	int dim = A.width;
	if (dim != B.width) {
		printf("vectors in kernel have inconsistent dimension\n");
		return false;
	}

	const int M = A.height;
	const int N = B.height;
	Matrix_v2 distMat(M, N);
	Matrix_v2 res(M, N);

	if (C.elements != NULL) {
		delete[] C.elements;
	}
	C.height = M;
	C.width = N;
	C.elements = new float[M*N];

	float *d_x1, *d_x2, *d_dsm, *d_res;

	//prepare GPU memory
	cudacall(cudaMalloc(&d_x1, M*dim * sizeof(float)));
	cudacall(cudaMemcpy(d_x1, A.elements, M*dim * sizeof(float), cudaMemcpyHostToDevice));

	cudacall(cudaMalloc(&d_x2, N*dim * sizeof(float)));
	cudacall(cudaMemcpy(d_x2, B.elements, N*dim * sizeof(float), cudaMemcpyHostToDevice));

	cudacall(cudaMalloc(&d_dsm, M*N * sizeof(float)));
	cudacall(cudaMemcpy(d_dsm, distMat.elements, M*N * sizeof(float), cudaMemcpyHostToDevice));

	cudacall(cudaMalloc(&d_res, M*N * sizeof(float)));
	cudacall(cudaMemcpy(d_res, res.elements, M*N * sizeof(float), cudaMemcpyHostToDevice));

	//kernel
	GaussianKernel << <1024, 32 >> > (d_x1, d_x2, d_dsm, d_res, L, sigma_f, M, N, dim);

	//copy the memory back to host
	cudacall(cudaMemcpy(C.elements, d_res, M*N * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_x1);
	cudaFree(d_x2);
	cudaFree(d_dsm);
	cudaFree(d_res);

	return true;
}

#define WRITE_MAT(fn, tar) {												\
	FILE *f;																\
	char fname[100];														\
	sprintf(fname, "..\\%s.txt", fn);										\
	f = fopen(fname, "w");													\
	for (int i = 0; i < tar.height; i++) {									\
		for (int j = 0; j < tar.width; j++) {								\
			fprintf(f, "%f\t", tar.elements[i + j*tar.height]);				\
		}																	\
		fprintf(f, "\n");													\
	}																		\
	fclose(f);																\
}

void GPR::output2file() {
	WRITE_MAT("train_x", train_x);
	WRITE_MAT("train_y", train_y);
	WRITE_MAT("test_x", test_x);
	WRITE_MAT("mu", mu);
	WRITE_MAT("cov", cov);
}