#ifndef __CUDACC__  
#define __CUDACC__
#endif


#include <cuda_runtime.h>
#include "page_rank.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <iostream>

texture<int> source_indices_tex;

__global__ void compute_tex_shared(const int* destination_offsets, const int* out_degrees, const int node_count, const float* input, float *output)
{
	int dest = blockDim.x*blockIdx.x + threadIdx.x;
	__shared__ int s_dest_off[BLOCK_SIZE+1];
	if (dest<node_count)
	{
		s_dest_off[threadIdx.x]= destination_offsets[dest];
		if(threadIdx.x == BLOCK_SIZE-1 || dest == node_count - 1)
		{
			s_dest_off[threadIdx.x+1] = destination_offsets[dest+1];
		}
		__syncthreads();
		int srcStart = s_dest_off[threadIdx.x];
		int srcEnd = s_dest_off[threadIdx.x + 1];
		int in_degree = srcEnd - srcStart;
		float rank = 0;
		if (in_degree>0)
		{
			for (int srcIdx = srcStart; srcIdx<srcEnd; ++srcIdx)
			{
				int src = tex1Dfetch(source_indices_tex, srcIdx);				
				float contrib = ((input[src] * DECAY) / out_degrees[src]);
				rank = rank + contrib;
			}
		}
		output[dest] = rank + (1 - DECAY);
	}
}
__global__ void compute_with_tex(const int* destination_offsets, const int* out_degrees, const int node_count, const float* input, float *output)
{
	int dest = blockDim.x*blockIdx.x + threadIdx.x;

	if (dest<node_count)
	{
		int srcStart = destination_offsets[dest];
		int srcEnd = destination_offsets[dest + 1];
		int in_degree = srcEnd - srcStart;
		float rank = 0;
		if (in_degree>0)
		{
			for (int srcIdx = srcStart; srcIdx<srcEnd; ++srcIdx)
			{
				int src = tex1Dfetch(source_indices_tex, srcIdx);
				float contrib =((input[src] * DECAY) / out_degrees[src]);
				rank = rank + contrib;
			}
		}
		output[dest] = rank + (1 - DECAY);
	}
}
__global__ void compute_shared(const int* destination_offsets, const int* source_indices, const int* out_degrees, const int node_count, const float* input, float *output)
{
	int dest = blockDim.x*blockIdx.x + threadIdx.x;
	__shared__ int s_dest_off[BLOCK_SIZE + 1];
	if (dest<node_count)
	{
		s_dest_off[threadIdx.x] = destination_offsets[dest];
		if (threadIdx.x == BLOCK_SIZE - 1 || dest == node_count - 1)
		{
			s_dest_off[threadIdx.x + 1] = destination_offsets[dest + 1];
		}
		__syncthreads();
		int srcStart = s_dest_off[threadIdx.x];
		int srcEnd = s_dest_off[threadIdx.x + 1];
		int in_degree = srcEnd - srcStart;
		float rank = 0;
		if (in_degree>0)
		{
			for (int srcIdx = srcStart; srcIdx<srcEnd; ++srcIdx)
			{
				int src = source_indices[srcIdx];
				float contrib = ((input[src] * DECAY) / out_degrees[src]);
				rank = rank + contrib;
			}
		}
		output[dest] = rank + (1 - DECAY);
	}
}
__global__ void compute(const int* destination_offsets, const int* source_indices, const int* out_degrees, const int node_count, const float* input, float *output)
{
	int dest = blockDim.x*blockIdx.x + threadIdx.x;
	if (dest<node_count)
	{
		int srcStart = destination_offsets[dest];
		int srcEnd = destination_offsets[dest + 1];
		int in_degree = srcEnd - srcStart;
		float rank = 0;
		if (in_degree>0)
		{
			for (int srcIdx = srcStart; srcIdx<srcEnd; ++srcIdx)
			{
				int src = source_indices[srcIdx];
				float contrib = ((input[src] * DECAY) / out_degrees[src]);
				rank = rank + contrib;
			}
		}
		output[dest] = rank + (1 - DECAY);
	}
}

__global__ void compute_tex_shared_inv(const int* destination_offsets, const float* out_degrees, const int node_count, const float* input, float *output)
{
	int dest = blockDim.x*blockIdx.x + threadIdx.x;
	__shared__ int s_dest_off[BLOCK_SIZE + 1];
	if (dest<node_count)
	{
		s_dest_off[threadIdx.x] = destination_offsets[dest];
		if (threadIdx.x == BLOCK_SIZE - 1 || dest == node_count - 1)
		{
			s_dest_off[threadIdx.x + 1] = destination_offsets[dest + 1];
		}
		__syncthreads();
		int srcStart = s_dest_off[threadIdx.x];
		int srcEnd = s_dest_off[threadIdx.x + 1];
		int in_degree = srcEnd - srcStart;
		float rank = 0;
		if (in_degree>0)
		{
			for (int srcIdx = srcStart; srcIdx<srcEnd; ++srcIdx)
			{
				int src = tex1Dfetch(source_indices_tex, srcIdx);
				float contrib = ((input[src] * DECAY) *out_degrees[src]);
				rank = rank + contrib;
			}
		}
		output[dest] = rank + (1 - DECAY);
	}
}

__global__ void compute_with_tex_inv(const int* destination_offsets, const float* out_degrees, const int node_count, const float* input, float *output)
{
	int dest = blockDim.x*blockIdx.x + threadIdx.x;

	if (dest<node_count)
	{
		int srcStart = destination_offsets[dest];
		int srcEnd = destination_offsets[dest + 1];
		int in_degree = srcEnd - srcStart;
		float rank = 0;
		if (in_degree>0)
		{
			for (int srcIdx = srcStart; srcIdx<srcEnd; ++srcIdx)
			{
				int src = tex1Dfetch(source_indices_tex, srcIdx);
				float contrib = ((input[src] * DECAY) * out_degrees[src]);
				rank = rank + contrib;
			}
		}
		output[dest] = rank + (1 - DECAY);
	}
}

__global__ void compute_shared_inv(const int* destination_offsets, const int* source_indices, const float* out_degrees, const int node_count, const float* input, float *output)
{
	int dest = blockDim.x*blockIdx.x + threadIdx.x;
	__shared__ int s_dest_off[BLOCK_SIZE + 1];
	if (dest<node_count)
	{
		s_dest_off[threadIdx.x] = destination_offsets[dest];
		if (threadIdx.x == BLOCK_SIZE - 1 || dest == node_count - 1)
		{
			s_dest_off[threadIdx.x + 1] = destination_offsets[dest + 1];
		}
		__syncthreads();
		int srcStart = s_dest_off[threadIdx.x];
		int srcEnd = s_dest_off[threadIdx.x + 1];
		int in_degree = srcEnd - srcStart;
		float rank = 0;
		if (in_degree>0)
		{
			for (int srcIdx = srcStart; srcIdx<srcEnd; ++srcIdx)
			{
				int src = source_indices[srcIdx];
				float contrib = ((input[src] * DECAY) * out_degrees[src]);
				rank = rank + contrib;
			}
		}
		output[dest] = rank + (1 - DECAY);
	}
}

__global__ void compute_inv(const int* destination_offsets, const int* source_indices, const float* out_degrees, const int node_count, const float* input, float *output)
{
	int dest = blockDim.x*blockIdx.x + threadIdx.x;
	if (dest<node_count)
	{
		int srcStart = destination_offsets[dest];
		int srcEnd = destination_offsets[dest + 1];
		int in_degree = srcEnd - srcStart;
		float rank = 0;
		if (in_degree>0)
		{
			for (int srcIdx = srcStart; srcIdx<srcEnd; ++srcIdx)
			{
				int src = source_indices[srcIdx];
				float contrib = ((input[src] * DECAY) * out_degrees[src]);
				rank = rank + contrib;
			}
		}
		output[dest] = rank + (1 - DECAY);
	}
}

__global__ void max_abs_diff(float* diff, const float* output1, const float* output2, const int size)
{
	extern __shared__ float s_max[];
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int tx = threadIdx.x;
	if (i<size)
	{
		float o1 = output1[i];
		if (o1 == -1)
		{
			s_max[tx] = -1;
		}
		else
		{
			s_max[tx] = fabsf(o1 - output2[i]);
		}
	}
	else
	{
		s_max[tx] = -1;
	}
	for (int j = blockDim.x / 2; j> 0; j >>= 1)
	{
		__syncthreads();
		if (tx<j)
		{
			s_max[tx] = fmaxf(s_max[tx], s_max[tx + j]);
		}
	}
	if (tx == 0)
	{
		diff[blockIdx.x] = s_max[0];
	}
}

__global__ void invert(float *output, int* input, const int size)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<size)
	{
		int d = input[i];
		if(d>0)
		{
			output[i] = __fdividef(1.f, d);
		}
	}
}

using namespace std;

void compute_gpu(CSC_t &h_data, bool useTexture, bool useShared)
{
	int* source_indices;
	int* destination_offsets;
	int* out_degrees;
	cudaMalloc((void**)&source_indices, h_data->nedges * sizeof(int));
	cudaMalloc((void**)&destination_offsets, (h_data->nvertices + 1) * sizeof(int));
	cudaMalloc((void**)&out_degrees, h_data->nvertices * sizeof(int));
	cudaMemcpy(source_indices, h_data->source_indices, h_data->nedges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(destination_offsets, h_data->destination_offsets, (h_data->nvertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(out_degrees, h_data->out_degrees, h_data->nvertices * sizeof(int), cudaMemcpyHostToDevice);
	if (useTexture)
	{
		cudaChannelFormatDesc d = cudaCreateChannelDesc<int>();
		size_t offset;
		cudaError_t error = cudaBindTexture(&offset, &source_indices_tex, source_indices, &d, h_data->nedges * sizeof(int));
		if (error != cudaSuccess)
		{
			printf("cudaBindTexture returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
			printf("Result = FAIL\n");
			exit(EXIT_FAILURE);
		}
	}
	int threads = BLOCK_SIZE;
	int blocks = (int)ceil((float)h_data->nvertices / threads);
	float* d_interim1;
	float* d_interim2;
	float* d_diff;
	float* h_diff = (float*)malloc(blocks * sizeof(float));

	cudaMalloc((void**)&d_diff, h_data->nvertices * sizeof(float));

	cudaMalloc((void**)&d_interim1, h_data->nvertices * sizeof(float));
	
	float* tmp = (float*)malloc(h_data->nvertices * sizeof(float));
	fill(tmp, tmp + h_data->nvertices, 1.f);	
	cudaMemcpy(d_interim1, tmp, h_data->nvertices * sizeof(float), cudaMemcpyHostToDevice);
	free(tmp);

	cudaMalloc((void**)&d_interim2, h_data->nvertices * sizeof(float));
	cudaMemset(d_interim2, 0, h_data->nvertices * sizeof(float));

	int i = 0;
	float elapsed = 0;
	float diff = 10000;
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	float *input, *output;

	while (diff >= THRESHOLD)
	{
		//cout << "GPU: " << "Compute Ranks Pass: " << i + 1 << endl;
		
		if (i % 2 == 0) {
			input = d_interim1;
			output = d_interim2;
		}
		else
		{
			input = d_interim2;
			output = d_interim1;
		}
		
		cudaEventRecord(start_event, 0);
		if (useTexture)
		{
			if (useShared)
			{
				compute_tex_shared << <blocks, threads >> > (destination_offsets, out_degrees, h_data->nvertices, input, output);
			}
			else
			{
				compute_with_tex << <blocks, threads >> > (destination_offsets, out_degrees, h_data->nvertices, input, output);
			}
		}
		else
		{
			if (useShared)
			{
				compute_shared << <blocks, threads >> > (destination_offsets, source_indices, out_degrees, h_data->nvertices, input, output);
			}
			else
			{
				compute << <blocks, threads >> > (destination_offsets, source_indices, out_degrees, h_data->nvertices, input, output);
			}
		}

		cudaDeviceSynchronize();
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&elapsed, start_event, stop_event);
		cout <<  "Pass " << i + 1 << ":" << elapsed << endl;

		if (i % 3 == 0)
		{
			//cout << "GPU: " << "Compute Rank Diff of Passes: " << i << " and " << i + 1 << endl;
			cudaEventRecord(start_event, 0);

			max_abs_diff << <blocks, threads, threads * sizeof(float) >> >(d_diff, input, output, h_data->nvertices);

			cudaMemcpy(h_diff, d_diff, blocks * sizeof(float), cudaMemcpyDeviceToHost);
			float max_diff = 0;
			// Max among every block's max
			for (int j = 0; j<blocks; j++)
			{
				max_diff = fmaxf(max_diff, h_diff[j]);
			}
			diff = max_diff;
			cudaEventRecord(stop_event, 0);
			cudaEventSynchronize(stop_event);
			cudaEventElapsedTime(&elapsed, start_event, stop_event);
			cout << "Diff:" << elapsed<< endl;
			//cout << "GPU: " << "Difference updates to: " << diff << endl;
		}
		cudaMemset(input, 0, h_data->nvertices * sizeof(float));
		i++;
	}
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	if (useTexture)
		cudaUnbindTexture(&source_indices_tex);
	cudaFree(source_indices);
	cudaFree(destination_offsets);
	cudaFree(out_degrees);
	cudaFree(d_diff);
	cudaFree(d_interim1);
	cudaFree(d_interim2);
}

void compute_gpu_invert(CSC_t &h_data, bool useTexture, bool useShared)
{
	int* source_indices;
	int* destination_offsets;
	float* out_degrees;
	cudaMalloc((void**)&source_indices, h_data->nedges * sizeof(int));
	cudaMalloc((void**)&destination_offsets, (h_data->nvertices + 1) * sizeof(int));
	cudaMalloc((void**)&out_degrees, h_data->nvertices * sizeof(float));
	cudaMemset(out_degrees, 0, h_data->nvertices * sizeof(float));
	cudaMemcpy(source_indices, h_data->source_indices, h_data->nedges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(destination_offsets, h_data->destination_offsets, (h_data->nvertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
	
	if (useTexture)
	{
		cudaChannelFormatDesc d = cudaCreateChannelDesc<int>();
		size_t offset;
		cudaError_t error = cudaBindTexture(&offset, &source_indices_tex, source_indices, &d, h_data->nedges * sizeof(int));
		if (error != cudaSuccess)
		{
			printf("cudaBindTexture returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
			printf("Result = FAIL\n");
			exit(EXIT_FAILURE);
		}
	}	

	int threads = BLOCK_SIZE;
	int blocks = (int)ceil((float)h_data->nvertices / threads);

	int* tmp_degrees;
	cudaMalloc((void**)&tmp_degrees, h_data->nvertices * sizeof(int));
	cudaMemcpy(tmp_degrees, h_data->out_degrees, h_data->nvertices * sizeof(int), cudaMemcpyHostToDevice);
	invert<<<blocks, threads>>>(out_degrees, tmp_degrees, h_data->nvertices);
	cudaFree(tmp_degrees);
	
	float* d_interim1;
	float* d_interim2;
	float* d_diff;
	float* h_diff = (float*)malloc(blocks * sizeof(float));

	cudaMalloc((void**)&d_diff, h_data->nvertices * sizeof(float));

	cudaMalloc((void**)&d_interim1, h_data->nvertices * sizeof(float));

	float* tmp = (float*)malloc(h_data->nvertices * sizeof(float));
	fill(tmp, tmp + h_data->nvertices, 1.f);
	cudaMemcpy(d_interim1, tmp, h_data->nvertices * sizeof(float), cudaMemcpyHostToDevice);
	free(tmp);

	cudaMalloc((void**)&d_interim2, h_data->nvertices * sizeof(float));
	cudaMemset(d_interim2, 0, h_data->nvertices * sizeof(float));

	int i = 0;
	float elapsed = 0;
	float diff = 10000;
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	float *input, *output;

	while (diff >= THRESHOLD)
	{
		//cout << "GPU: " << "Compute Ranks Pass: " << i + 1 << endl;

		if (i % 2 == 0) {
			input = d_interim1;
			output = d_interim2;
		}
		else
		{
			input = d_interim2;
			output = d_interim1;
		}

		cudaEventRecord(start_event, 0);
		if (useTexture)
		{
			if (useShared)
			{
				compute_tex_shared_inv << <blocks, threads >> > (destination_offsets, out_degrees, h_data->nvertices, input, output);
			}
			else
			{
				compute_with_tex_inv << <blocks, threads >> > (destination_offsets, out_degrees, h_data->nvertices, input, output);
			}
		}
		else
		{
			if (useShared)
			{
				compute_shared_inv << <blocks, threads >> > (destination_offsets, source_indices, out_degrees, h_data->nvertices, input, output);
			}
			else
			{
				compute_inv << <blocks, threads >> > (destination_offsets, source_indices, out_degrees, h_data->nvertices, input, output);
			}
		}

		cudaDeviceSynchronize();
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&elapsed, start_event, stop_event);
		cout << "Pass " << i + 1 << ":" << elapsed << endl;

		if (i % 3 == 0)
		{
			//cout << "GPU: " << "Compute Rank Diff of Passes: " << i << " and " << i + 1 << endl;
			cudaEventRecord(start_event, 0);

			max_abs_diff << <blocks, threads, threads * sizeof(float) >> >(d_diff, input, output, h_data->nvertices);

			cudaMemcpy(h_diff, d_diff, blocks * sizeof(float), cudaMemcpyDeviceToHost);
			float max_diff = 0;
			// Max among every block's max
			for (int j = 0; j<blocks; j++)
			{
				max_diff = fmaxf(max_diff, h_diff[j]);
			}
			diff = max_diff;
			cudaEventRecord(stop_event, 0);
			cudaEventSynchronize(stop_event);
			cudaEventElapsedTime(&elapsed, start_event, stop_event);
			cout << "Diff:" << elapsed << endl;
			//cout << "GPU: " << "Difference updates to: " << diff << endl;
		}
		cudaMemset(input, 0, h_data->nvertices * sizeof(float));
		i++;
	}
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	if (useTexture)
		cudaUnbindTexture(&source_indices_tex);
	cudaFree(source_indices);
	cudaFree(destination_offsets);
	cudaFree(out_degrees);
	cudaFree(d_diff);
	cudaFree(d_interim1);
	cudaFree(d_interim2);
}