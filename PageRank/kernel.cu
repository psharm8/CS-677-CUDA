#ifndef _KERNEL_CU_
#define _KERNEL_CU_

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include "device_functions.h"
#define DECAY 0.85f

__global__ void compute(const int* indices, const int* adj, const int* out_degrees, const int node_count, const int adj_length, const float* input, float *output)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<node_count)
	{
		int adj_start = indices[i];
		float out = 0;
		if (adj_start > -1)
		{
			int count;
			if (i == node_count - 1)
			{
				count = adj_length - adj_start;				
			}
			else
			{
				int next_index = i;
				while (indices[++next_index] < 0);
				count = indices[next_index] - adj_start;
			}
			
			if (count > 0)
			{
				for (int j = 0; j < count; ++j)
				{
					int src = adj[adj_start + j];
					float contrib=((input[src] * DECAY) / out_degrees[src]);
					out = out + contrib;			
				}
			}
		}
		output[i] = out + (1 - DECAY);
	}
}

__global__ void max_abs_diff(float* diff, float* output1, float* output2, const int size)
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

__global__ void abs_diff(float* diff, float* output1, float *output2, const int size)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i<size)
	{
		float o1 = output1[i];
		if(o1==-1)
		{
			diff[i] = -1;
		}
		else
		{
			diff[i] = fabsf(output1[i] - output2[i]);
		}		
	}
}

__global__ void diff_max(float* diff, const int size)
{
	extern __shared__ float s_max[];
	int i = (blockDim.x*blockIdx.x + threadIdx.x);
	int tx = threadIdx.x;
	if (i < size) {
		s_max[tx] = diff[i];
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

#endif
