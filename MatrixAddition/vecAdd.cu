#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// One thread computes one element of output matrix
__global__ void addOneElementPerThread(double* a, double* b, double* c, int n)
{
	// Get our global thread ID
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = idy * n + idx;
	// Make sure we do not go out of bounds
	if (idx < n && idy < n)
		c[id] = a[id] + b[id];
}

// One thread computes one row of output matrix
__global__ void addOneRowPerThread(double* a, double* b, double* c, int n)
{
	// Get the row for current thread
	int row = (blockIdx.y * blockDim.y + threadIdx.y);

	// Make sure we do not go out of bounds
	if (row < n)
	{
		int idx = row * n;
		for (int i = 0; i < n; i++)
		{
			c[idx + i] = a[idx + i] + b[idx + i];
		}
	}
}

// One thread computes one column of output matrix
__global__ void addOneColumnPerThread(double* a, double* b, double* c, int n)
{
	// Get the column for current thread
	int column = (blockIdx.x * blockDim.x + threadIdx.x);

	// Make sure we do not go out of bounds
	if (column < n)
	{
		for (int i = 0; i < n; i++)
		{
			c[i * n + column] = a[i * n + column] + b[i * n + column];
		}
	}
}

// Start the addition of two matrices of size 1024*1024
// One of the three types of kernels can be chosen to compute
// by passing the id of the kernel (a, b or c) as a command line parameter
//
// If no argumernt is provided it defaults to kernel 'a'
//
// Kernels:
//	a) Computes one element per thread
//	b) Computes one row per thread
//	c) Computes one colum per thread
int main(int argc, char* argv[])
{
	// matrix dimension
	const int n = 1024;
	char kernel = '\0';
	if (argc == 2)
	{
		char in = argv[1][0];
		if (in == 'a' || in == 'b' || in == 'c')
		{
			kernel = in;
		}
	}

	if (kernel)
	{
		printf("Choosing kernel %c.\n", kernel);
	}
	else
	{
		printf("Using default kernel a.\n");
	}

	// Host input matrices
	double* h_a;
	double* h_b;
	//Host output matrix
	double* h_c;

	// Device input matrices
	double* d_a;
	double* d_b;
	//Device output matrix
	double* d_c;

	// Size, in bytes, of each matrix
	size_t bytes = n * n * sizeof(double);

	// Allocate memory for each vector on host
	h_a = (double*)malloc(bytes);
	h_b = (double*)malloc(bytes);
	h_c = (double*)malloc(bytes);

	// Allocate memory for each vector on GPU
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	int i;
	// Initialize matrices on host
	for (i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			
			h_a[i * n + j] = i + 1;
			h_b[i * n + j] = 2 * (i + 1);
		}

	// Copy host vectors to device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	dim3 blockSize, gridSize;

	switch (kernel)
	{
		case 'b':
			printf("Calculating one row per thread\n");
			// Number of threads in each thread block
			blockSize = dim3(1, 1024);
			// Number of thread blocks in grid
			gridSize = dim3(1, (int)ceil((float)n / blockSize.y));
			// Execute the kernel
			addOneRowPerThread << <gridSize, blockSize >> >(d_a, d_b, d_c, n);
			break;
		case 'c':
			printf("Calculating one column per thread\n");
			// Number of threads in each thread block
			blockSize = dim3(1024, 1);
			// Number of thread blocks in grid
			gridSize = dim3((int)ceil((float)n / blockSize.x), 1);
			// Execute the kernel
			addOneColumnPerThread << <gridSize, blockSize >> >(d_a, d_b, d_c, n);
			break;
		default: //Kernel 'a'
			// Number of threads in each thread block
			blockSize = dim3(32, 32);
			// Number of thread blocks in grid
			gridSize = dim3((int)ceil((float)n / blockSize.x), (int)ceil((float)n / blockSize.y));
			// Execute the kernel
			addOneElementPerThread << <gridSize, blockSize >> >(d_a, d_b, d_c, n);
	}

	// Copy array back to host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	printf("Matrix h_a\n\n");
	for (i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
			printf("%f\t", h_a[i * n + j]);
		printf("\n");
	}
	printf("\nMatrix h_b\n\n");
	for (i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
			printf("%f\t", h_b[i * n + j]);
		printf("\n");
	}
	printf("\nResult h_c = h_a + h_b\n");

	// Print result
	for (i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
			printf("%f\t", h_c[i * n + j]);
		printf("\n");
	}

	// Release device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Release host memory
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}
