
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

// Length of vectors
#define N 12 * 1024

#define ThreadsPerBlock 128
#define BlocksPerGrid (N + ThreadsPerBlock - 1) / ThreadsPerBlock

__global__ void productKernel(float *c, const float *a, const float *b)
{
	__shared__ float cache[ThreadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	// sum
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = temp;
	__syncthreads();
	// reduce algorithm
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
		{
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

int main()
{
	float *v_a, *v_b, *v_c;
	float *dev_va, *dev_vb, *dev_vc;

	// Allocate memory on CPU
	v_a = (float*)malloc(N * sizeof(float));
	v_b = (float*)malloc(N * sizeof(float));
	v_c = (float*)malloc(BlocksPerGrid * sizeof(float*));

	// Allocate memory on GPU
	cudaMalloc((void**)&dev_va, N * sizeof(float));
	cudaMalloc((void**)&dev_vb, N * sizeof(float));
	cudaMalloc((void**)&dev_vc, BlocksPerGrid * sizeof(float));

	// Generate data for vector
	for (int i = 0; i < N; ++i)
	{
		v_a[i] = 1;
		v_b[i] = 1; 
	}

	// Copy data to device from host
	cudaMemcpy(dev_va, v_a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vb, v_b, N * sizeof(float), cudaMemcpyHostToDevice);

	productKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_vc, dev_va, dev_vb);

	// Copy result array to host from device
	cudaMemcpy(v_c, dev_vc, BlocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 1; i < BlocksPerGrid; ++i)
	{
		v_c[0] += v_c[i];
	}

	printf("%lf\n", v_c[0]);

	// Free memory
	free(v_a);
	free(v_b);
	free(v_c);
	cudaFree(dev_va);
	cudaFree(dev_vb);
	cudaFree(dev_vc);
    return 0;
}