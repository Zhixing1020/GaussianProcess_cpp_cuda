
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "GauProClass.cuh"
//#include "Matrix_lib.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include<algorithm>
#include<time.h>

vector<float> X, Y, X_, Y_;

void genData() {
	X.reserve(MaxData*dataDim);
	Y.reserve(MaxData*dataDim);
	for (int i = 0; i < numData; i++) {
		X.push_back(((double)rand() / RAND_MAX)*DataRng);
	}
	sort(X.begin(), X.end());
	for (auto x : X) Y.push_back(0.5*sin(3 * x));

	X_.reserve(MaxData*dataDim);
	Y_.reserve(MaxData*dataDim);
	for (int i = 0; i < test_numData; i++) {
		X_.push_back(((double)rand() / RAND_MAX)*DataRng);
	}
	sort(X_.begin(), X_.end());
	for (auto x:X_) Y_.push_back(0.5*sin(3 * x));
}


int main()
{
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
    
	genData();
	GPR gpr;

	clock_t start = clock();
	start = time(NULL);
	gpr.fit(X, Y);
	gpr.predict(X_);
	clock_t end = clock();
	printf("running time: %f s\n", (end - start)/CLOCKS_PER_SEC);
	gpr.output2file();

	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}
