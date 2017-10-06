#include <algorithm>
#include <iostream>
#include <chrono>

#include "page_rank.h"

using namespace std;

void compute_contribution(CSC_t &data, float* input, float* output)
{
	for (int dest = 0; dest < data->nvertices; ++dest)
	{
		int srcStart = data->destination_offsets[dest];
		int srcEnd = data->destination_offsets[dest + 1];
		int in_degree = srcEnd - srcStart;
		float rank = 0;
		if (in_degree>0)
		{
			for (int srcIdx = srcStart; srcIdx<srcEnd; srcIdx++)
			{
				int src = data->source_indices[srcIdx];
				rank += input[src] * DECAY / data->out_degrees[src];
			}
		}
		output[dest] += (1 - DECAY) + rank;
	}
}

float compute_max_rank_diff(float* output1, float* output2, int size)
{
	float maxDiff = 0;
	for (int i = 0; i < size; ++i)
	{
		if (output1[i] == -1) continue;
		float diff = abs(output1[i] - output2[i]);
		maxDiff = max(maxDiff, diff);
	}
	return maxDiff;
}

void compute_cpu(CSC_t &data)
{
	int f_size = data->nvertices * sizeof(float);
	float *interim1 = (float*)malloc(f_size);
	float *interim2 = (float*)malloc(f_size);
	fill(interim1, interim1 + data->nvertices, 1.f);
	fill(interim2, interim2 + data->nvertices, 0.f);
	int i = 0;
	float diff = 10000;
	while (diff >= THRESHOLD)
	{
		//cout << "Compute Ranks Pass: " << i + 1 << endl;
		if (i % 2 == 0)
		{
			auto start = chrono::high_resolution_clock::now();
			compute_contribution(data, interim1, interim2);
			auto end = chrono::high_resolution_clock::now();
			cout << "Pass " << i + 1 << ":" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
		}
		else
		{
			auto start = chrono::high_resolution_clock::now();
			compute_contribution(data, interim2, interim1);
			auto end = chrono::high_resolution_clock::now();
			cout << "Pass " << i + 1 << ":" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
		}
		if (i % 3 == 0)
		{
			//cout << "Compute Rank Diff of Passes: " << i << " and " << i + 1 << endl;
			auto start = chrono::high_resolution_clock::now();
			diff = compute_max_rank_diff(interim1, interim2, data->nvertices);
			auto end = chrono::high_resolution_clock::now();
			//cout << "Difference updates to: " << diff << endl;
			cout << "Diff:" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
		}

		if (i % 2 == 0)
		{
			fill(interim1, interim1 + data->nvertices, 0.f);
		}
		else
		{
			fill(interim2, interim2 + data->nvertices, 0.f);
		}
		i++;
	}
	
	free(interim1);
	free(interim2);
}

