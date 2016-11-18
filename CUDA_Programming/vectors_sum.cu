
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
// Length of vector
#define N 200 * 1024

#define ThreadsPerBlock 128
#define BlocksPerGrid 128

__global__ void vectorAddKernel(int *c, const int *a, const int *b)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int main()
{
    int *v_a, *v_b;
	int *v_c;
	int *dev_va, *dev_vb;
	int *dev_vc;
	// Open a file for future writing
	FILE *fp = fopen("result.txt", "w");

	// Allocate memory on CPU
	v_a = (int*)malloc(N * sizeof(int));
	v_b = (int*)malloc(N * sizeof(int));
	v_c = (int*)malloc(N * sizeof(int));
	// Allocate memory on GPU
	cudaMalloc((void**)&dev_va, N * sizeof(int));
	cudaMalloc((void**)&dev_vb, N * sizeof(int));
	cudaMalloc((void**)&dev_vc, N * sizeof(int));

	// Generate data of vectors
	for (int i = 0; i < N; ++i)
	{
		v_a[i] = 3;
		v_b[i] = 3;
	}

	// Copy data to device from host
	cudaMemcpy(dev_va, v_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vb, v_b, N * sizeof(int), cudaMemcpyHostToDevice);

	vectorAddKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_vc, dev_va, dev_vb);
    
	// Copy result to host from device
	cudaMemcpy(v_c, dev_vc, N * sizeof(int), cudaMemcpyDeviceToHost);

	// Print result
	for (int i = 0; i < N; ++i)
	{
		fprintf(fp, "%d + %d = %d\n", v_a[i], v_b[i], v_c[i]);
	}
	
	// Free memory
	free(v_a);
	free(v_b);
	free(v_c);
	cudaFree(dev_va);
	cudaFree(dev_vb);
	cudaFree(dev_vc);

	return 0;
}