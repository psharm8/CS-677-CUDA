#ifndef _KERNEL_CU_
#define _KERNEL_CU_

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "main.h"
#include "device_launch_parameters.h"
#include "device_functions.h"


__global__ void init_ranks(float *ranks, const int size)
{
    int i = blockDim.x*blockIdx.x+ threadIdx.x;
	if(i<size)
	{
		ranks[i] = 1;
	}
}

__global__ void compute(const Data data, const float* input, float *output)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<data.node_count)
	{
		int adj_start = data.indices[i];
		int out = 0;
		if (adj_start > -1)
		{
			int count;
			if (i == data.node_count - 1)
			{
				count = data.adj_length - adj_start;
			}
			else
			{
				int next_index = i;
				while (data.indices[++next_index] < 0);
				count = data.indices[next_index] - adj_start;
			}

			if (count > 0)
			{
				for (int j = 0; j < count; ++j)
				{
					int src = data.adj[adj_start + j];
					out += (input[src] * DECAY) / data.out_degrees[src];
				}
			}
		}
		output[i] = out + (1 - DECAY);
	}
}

__global__ void diff(float* output1, float *output2, const int size)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i<size)
	{
		output1[i] = fabsf(output1[i] - output2[i]);
	}
}

__global__ void max(float* diff, const int size, const int stride)
{
	extern __shared__ float s_max[];
	int i = (blockDim.x*blockIdx.x + threadIdx.x)*stride;
	int tx = threadIdx.x;
	if(i<size)
	{
		s_max[tx] = diff[i];
	}
	for (int j = blockDim.x / 2; j> 0; j >>= 1)
	{
		__syncthreads();
		if (tx<j)
		{
			s_max[tx] = fmaxf(s_max[tx],s_max[tx + j]);
		}
	}
	if(tx==0)
	{
		diff[0] = s_max[0];
	}	
}

#endif
