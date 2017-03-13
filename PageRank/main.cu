//#ifndef _CRT_SECURE_NO_WARNINGS
//#define _CRT_SECURE_NO_WARNINGS 1 
//#endif

#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

#include <fcntl.h>  // O_RDONLY

#include <vector>
#include <chrono>
#include <string.h>
#include <filesystem>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "kernel.cu"
#include "main.h"



using namespace std;
namespace fs = experimental::filesystem;


void computeContribution(Data &data, float* input, float* output);
float computeMaxRankDiff(float* output1, float* output2, int size);
void read_from_file(const char *fname, Data &data);
void read_data(const char* path, int node_count, Data &data);
void compute_cpu(Data &data);
Data allocate_on_device(const Data h_data);
void compute_gpu(const Data h_data);

int main(int argc, char** argv)
{
	int numNodes = 5716808;
	char* path = "C:\\Users\\only2\\links-simple-pairs\\";
	Data data;
	read_data(path, numNodes, data);
	//compute_cpu(data);
	compute_gpu(data);
}

Data allocate_on_device(const Data h_data)
{
	Data d_data = h_data;
	cudaMalloc((void**)&d_data.indices, h_data.node_count*sizeof(int));
	cudaMalloc((void**)&d_data.adj, h_data.adj_length * sizeof(int));
	cudaMalloc((void**)&d_data.out_degrees, h_data.node_count * sizeof(int));
	cudaMemcpy(d_data.indices, h_data.indices, h_data.node_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data.adj, h_data.adj, h_data.adj_length * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data.out_degrees, h_data.out_degrees, h_data.node_count * sizeof(int), cudaMemcpyHostToDevice);
	d_data.adj_length = h_data.adj_length;
	d_data.node_count = h_data.node_count;
	return d_data;
}
void compute_gpu(const Data h_data)
{
	chrono::time_point<chrono::steady_clock> compute_start = chrono::high_resolution_clock::now();
	Data d_data = allocate_on_device(h_data);
	int threads = 1024;
	int blocks = ceil((float)h_data.node_count / threads);
	float* d_interim1;
	float* d_interim2;
	cudaMalloc((void**)&d_interim1, h_data.node_count * sizeof(float));
	cudaMemset(d_interim1, 1, d_data.node_count);
	cudaMalloc((void**)&d_interim2, h_data.node_count * sizeof(float));
	cudaMemset(d_interim2, 0, d_data.node_count);

	chrono::time_point<chrono::steady_clock> start = chrono::high_resolution_clock::now();
	compute<<<blocks, threads>>>(d_data, d_interim1, d_interim2);
	cudaDeviceSynchronize();
	chrono::time_point<chrono::steady_clock> end = chrono::high_resolution_clock::now();	
	cout << "Pass " << 1 << " took: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds" << endl;
	
	cudaMemset(d_interim1, 0, d_data.node_count);

	start = chrono::high_resolution_clock::now();
	compute << <blocks, threads >> >(d_data, d_interim2, d_interim1);
	cudaDeviceSynchronize();
	end = chrono::high_resolution_clock::now();
	cout << "Pass " << 2 << " took: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds" << endl;
	
	cudaMemset(d_interim2, 0, d_data.node_count);

	start = chrono::high_resolution_clock::now();
	compute << <blocks, threads >> >(d_data, d_interim1, d_interim2);
	cudaDeviceSynchronize();
	
	end = chrono::high_resolution_clock::now();
	cout << "Pass " << 3 << " took: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds" << endl;

	
	cout << "GPU Compute took: " << chrono::duration_cast<chrono::milliseconds>(end - compute_start).count() << " milliseconds" << endl;
}
void compute_cpu(Data &data)
{
	chrono::time_point<chrono::steady_clock> compute_start = chrono::high_resolution_clock::now();
	float *interim1, *interim2;
	int f_size = data.node_count * sizeof(float);
	interim1 = (float*)malloc(f_size);
	interim2 = (float*)malloc(f_size);
	fill(interim1, interim1 + data.node_count, 1.f);
	fill(interim2, interim2 + data.node_count, 0.f);
	int i = 0;
	float diff = 10000;
	while (diff >= 30)
	{
		cout << "Compute Ranks Pass: " << i + 1 << endl;
		if (i % 2 == 0)
		{
			chrono::time_point<chrono::steady_clock> start = chrono::high_resolution_clock::now();
			computeContribution(data, interim1, interim2);
			chrono::time_point<chrono::steady_clock> end = chrono::high_resolution_clock::now();
			cout << "Pass " << i + 1 << " took: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds" << endl;
		}
		else
		{
			chrono::time_point<chrono::steady_clock> start = chrono::high_resolution_clock::now();
			computeContribution(data, interim2, interim1);
			chrono::time_point<chrono::steady_clock> end = chrono::high_resolution_clock::now();
			cout << "Pass " << i + 1 << " took: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds" << endl;
		}
		if (i % 3 == 0)
		{
			cout << "Compute Rank Diff of Passes: " << i << " and " << i + 1 << endl;
			chrono::time_point<chrono::steady_clock> start = chrono::high_resolution_clock::now();
			diff = computeMaxRankDiff(interim1, interim2, data.node_count);
			chrono::time_point<chrono::steady_clock> end = chrono::high_resolution_clock::now();
			cout << "Difference updates to: " << diff << endl;
			cout << "Diff took: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds" << endl;
		}
		if (i % 2 == 0)
		{
			fill(interim1, interim1 + data.node_count, 0.f);
		}
		else
		{
			fill(interim2, interim2 + data.node_count, 0.f);
		}
		i++;
	}

	free(interim1);
	free(interim2);
	chrono::time_point<chrono::steady_clock> compute_end = chrono::high_resolution_clock::now();
	cout << "Compute took: " << chrono::duration_cast<chrono::milliseconds>(compute_end - compute_start).count() << " milliseconds" << endl;
}

void computeContribution(Data &data, float* input, float* output)
{
	for (int i = 0; i < data.node_count; ++i)
	{
		if (data.indices[i] > -1)
		{
			int count;
			if (i == data.node_count - 1)
			{
				count = data.adj_length - data.indices[i];
			}
			else
			{
				int next_index = i;
				while (data.indices[++next_index] < 0);
				count = data.indices[next_index] - data.indices[i];
			}

			if (count > 0)
			{
				for (int j = 0; j < count; ++j)
				{
					int src = data.adj[data.indices[i] + j];
					output[i] += input[src] * DECAY / data.out_degrees[src];
				}
			}
		}
		output[i] += (1 - DECAY);
	}
}

float computeMaxRankDiff(float* output1, float* output2, int size)
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

void read_data(const char* path, int node_count, Data &data)
{
	data.node_count = node_count;
	data.indices = (int*)malloc(sizeof(int)*node_count);
	data.out_degrees = (int*)malloc(sizeof(int)*node_count);
	fill(data.indices, data.indices + node_count, -1);
	fill(data.out_degrees, data.out_degrees + node_count, 0);
	data.adj = nullptr;
	chrono::time_point<chrono::steady_clock> start_load = chrono::high_resolution_clock::now();

	for (fs::directory_entry p : fs::directory_iterator(path))
	{
		cout << "+-->Reading file: " << p.path().filename() << endl;
		read_from_file(p.path().string().data(), data);
	}
	chrono::time_point<chrono::steady_clock> end_load = chrono::high_resolution_clock::now();
	cout << "Loading took: " << chrono::duration_cast<chrono::milliseconds>(end_load - start_load).count() << " milliseconds" << endl;
}

void read_from_file(const char *fname, Data &data)
{
	chrono::time_point<chrono::steady_clock> start_load = chrono::high_resolution_clock::now();
	char buf[BUFFER_SIZE + 1];
	int fd = _open(fname, O_RDONLY);
	char* last = (char*)malloc(1);
	last[0] = '\0';
	int lines = 0;
	int size;
	if(data.adj_length ==-1)
	{
		data.adj = (int*)malloc(INT_SIZE);
		data.adj_length++;
	}
	size = data.adj_length + 1;
	while (size_t bytes_read = _read(fd, buf, BUFFER_SIZE))
	{
		char* b = buf;
		char* post;
		int read_count = 0;
		while (read_count<bytes_read)
		{
			post = strchr(b, '\n');
			int append_count = bytes_read - read_count;
			if (post != NULL)
			{
				append_count = post - b;
			}
			int len = strlen(last) + append_count + 1;
			last = (char*)realloc(last, len);
			strncat(last, b, append_count);
			read_count += append_count;
			b = post + 1;
			if (post != NULL)
			{
				read_count++;
				char* token;
				int dest = -1;
				
				token = strtok(last, TOEKNS);
				while (token != NULL)
				{
					if (dest==-1)
					{
						dest = atoi(token);
						data.indices[dest] = data.adj_length;
					}
					else
					{			
						int src = atoi(token);
						if (data.adj_length == size - 1)
						{
							size = size * 2 + 1;
							data.adj = (int*)realloc(data.adj, size * INT_SIZE);							
						}
						data.adj[data.adj_length++] = src;
						data.out_degrees[src]++;
					}					
					token = strtok(NULL, TOEKNS);
				}
				lines++;
				last = (char*)realloc(last, 1);
				last[0] = '\0';
			}
		}
	}
	free(last);
	chrono::time_point<chrono::steady_clock> end_load = chrono::high_resolution_clock::now();
	cout << "|   +--> " << lines << " lines Took: " << chrono::duration_cast<chrono::milliseconds>(end_load - start_load).count() << " milliseconds" << endl;
}
