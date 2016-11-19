
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 256

__constant__ int para[4];

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] * para[i % 4] + b[i];
}

int main()
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	int condata[4];
	for (int i = 0; i < N; ++i)
	{
		a[i] = 2;
		b[i] = 1;
	}
	for (int i = 0; i < 4; ++i)
	{
		condata[i] = 3;
	}
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));
	
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(para, condata, 4 * sizeof(int));
	
	addKernel<<<1, N>>>(dev_c, dev_a, dev_b);

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i)
	{
		printf("%d ", c[i]);
	}
	printf("\n");
    return 0;
}