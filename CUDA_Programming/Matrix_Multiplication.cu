
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

// Size of matrix A
#define AW 256
#define AH 256
// Size of matrix B
#define BW 256
#define BH 256

#define BLOCKSIZE 16

#define BlocksPerGrid ()
typedef struct
{
	int *elements;
	int width;
	int height;
} matrix;

__global__ void matrix_mul(matrix c, const matrix a, const matrix b)
{
	__shared__ int Asub[BLOCKSIZE][BLOCKSIZE];
	__shared__ int Bsub[BLOCKSIZE][BLOCKSIZE];
	int tid_r = blockDim.x * blockIdx.x + threadIdx.x;
	int tid_c = blockDim.y * blockIdx.y + threadIdx.y;
	int row = threadIdx.x;	int col = threadIdx.y;
	int Cvalue = 0;
	for (int i = 0; i < (a.width / BLOCKSIZE); ++i)
	{
		Asub[row][col] = a.elements[(blockIdx.x *  BLOCKSIZE + row) + (i * BLOCKSIZE + col)];
		Bsub[row][col] = b.elements[(i * BLOCKSIZE + row) + (blockIdx.y * BLOCKSIZE + col)];
		Asub[row][col] = a.elements[tid_r * a.width + (i * BLOCKSIZE + col)];
		Bsub[row][col] = b.elements[(i * BLOCKSIZE + row) * b.width + tid_c];
		__syncthreads();
		for (int e = 0; e < BLOCKSIZE; ++e)
			Cvalue += Asub[row][e] * Bsub[e][col];
		__syncthreads();
	}
	c.elements[tid_r * c.width + tid_c] = Cvalue;
}

int main()
{
	matrix a, b, c;
	matrix dev_a, dev_b, dev_c;

	// Allocate memory for Matrices a, b and c on CPU
	a.elements = (int*)malloc(AH * AW * sizeof(int));
	b.elements = (int*)malloc(BH * BW * sizeof(int));
	c.elements = (int*)malloc(AH * BW * sizeof(int));
	// Allocate memory for Matrices dev_a, dev_b and dev_c
	cudaMalloc((void**)&dev_a.elements, AH * AW * sizeof(int));
	cudaMalloc((void**)&dev_b.elements, BH * BW * sizeof(int));
	cudaMalloc((void**)&dev_c.elements, AH * BW * sizeof(int));
	// Initialize Matrices a, b, and c
	a.height = dev_a.height = AH; a.width = dev_a.width = AW;
	b.height = dev_b.height = BH; b.width = dev_b.width = BW;
	c.height = dev_c.height = AH; c.width = dev_c.width = BW;
	for (int i = 0; i < AH * AW; ++i)
	{
		a.elements[i] = 1;
	}
	for (int i = 0; i < BH * BW; ++i)
	{
		b.elements[i]= 1;
	}
	// Copy data from host to device
	cudaMemcpy(dev_a.elements, a.elements, AH * AW * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b.elements, b.elements, BH * BW * sizeof(int), cudaMemcpyHostToDevice);
	// Block Size and Grid size
	dim3 Threads(BLOCKSIZE, BLOCKSIZE);
	dim3 Blocks(a.height / BLOCKSIZE, b.width / BLOCKSIZE);

	matrix_mul<<<Blocks, Threads>>>(dev_c, dev_a, dev_b);

	cudaMemcpy(c.elements, dev_c.elements, AH * BW * sizeof(int), cudaMemcpyDeviceToHost);
	
	// Output result
	for (int i = 0; i < c.height; ++i)
	{
		for (int j = 0; j < c.width; ++j)
		{
			printf("%d ", c.elements[i * c.width + j]);
		}
		printf("\n");
	}

    return 0;
}