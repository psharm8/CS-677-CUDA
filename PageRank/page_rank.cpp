#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

#include <fcntl.h>  // O_RDONLY

#include <vector>
#include <chrono>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "page_rank.h"
#include "page_rank_compute.h"

void read_from_file(const char *fname, CSC_t &data);
void read_data(const char* path, int node_count, CSC_t &data);
void free_data(CSC_t &data);

using namespace std;

int main(int argc, char** argv)
{
	int numNodes = 5716808;
	char* path = "C:\\Users\\only2\\links-simple-pairs\\";
	if (argc == 2)
	{
		path = argv[1];
	}
	int repeat = 3;
	cout << "Path: " << path << endl;
	//char* path = "/home/ubuntu/links-simple-pairs/";
	CSC_t d;

	char* runs[10];
	runs[0] = "CPU";
	runs[1] = "GPU Global Mem";
	runs[2] = "GPU Shared Mem";
	runs[3] = "GPU Texture Mem";
	runs[4] = "GPU Tex+Shared Mem";
	runs[5] = "GPU Global Invert";
	runs[6] = "GPU Shared Invert";
	runs[7] = "GPU Texture Invert";
	runs[8] = "GPU Tex+Shared Invert";
	long avg[9];
	for (int i = 0; i < 9; ++i)
	{
		avg[i] = 0;
	}
	cout << "=============== FILES ===============" << endl;
	read_data(path, numNodes, d);
	auto gpu_start = chrono::high_resolution_clock::now();
	auto gpu_end = chrono::high_resolution_clock::now();
	long gpu_time = 0;
	for (int i = 0; i<repeat; i++)
	{
		cout << "================ CPU ================" << endl;
		auto cpu_start = chrono::high_resolution_clock::now();
		compute_cpu(d);
		auto cpu_end = chrono::high_resolution_clock::now();
		long cpu_time = chrono::duration_cast<chrono::milliseconds>(cpu_end - cpu_start).count();
		avg[0] += cpu_time;
		//cout << "CPU: " << "Compute took: " << cpu_time << " milliseconds" << endl;

		cout << "============= GPU Global ==============" << endl;
		gpu_start = chrono::high_resolution_clock::now();
		compute_gpu(d, false, false);
		gpu_end = chrono::high_resolution_clock::now();
		gpu_time = chrono::duration_cast<chrono::milliseconds>(gpu_end - gpu_start).count();
		avg[1] += gpu_time;
		//cout << "GPU: " << "Compute took: " << gpu_time << " milliseconds" << endl;

		cout << "============== GPU S.Mem ==============" << endl;
		gpu_start = chrono::high_resolution_clock::now();
		compute_gpu(d, false, true);
		gpu_end = chrono::high_resolution_clock::now();
		gpu_time = chrono::duration_cast<chrono::milliseconds>(gpu_end - gpu_start).count();
		avg[2] += gpu_time;
		//cout << "GPU: " << "Compute took: " << gpu_time << " milliseconds" << endl;

		cout << "============= GPU Tex.Mem =============" << endl;
		gpu_start = chrono::high_resolution_clock::now();
		compute_gpu(d, true, false);
		gpu_end = chrono::high_resolution_clock::now();
		gpu_time = chrono::duration_cast<chrono::milliseconds>(gpu_end - gpu_start).count();
		avg[3] += gpu_time;
		//cout << "GPU: " << "Compute took: " << gpu_time << " milliseconds" << endl;

		cout << "============ GPU Tex+S.Mem ============" << endl;
		gpu_start = chrono::high_resolution_clock::now();
		compute_gpu(d, true, true);
		gpu_end = chrono::high_resolution_clock::now();
		gpu_time = chrono::duration_cast<chrono::milliseconds>(gpu_end - gpu_start).count();
		avg[4] += gpu_time;
		//cout << "GPU: " << "Compute took: " << gpu_time << " milliseconds" << endl;

		cout << "=========== GPU Global Invert =========" << endl;
		gpu_start = chrono::high_resolution_clock::now();
		compute_gpu_invert(d, false, false);
		gpu_end = chrono::high_resolution_clock::now();
		gpu_time = chrono::duration_cast<chrono::milliseconds>(gpu_end - gpu_start).count();
		avg[5] += gpu_time;
		//cout << "GPU: " << "Compute took: " << gpu_time << " milliseconds" << endl;

		cout << "=========== GPU S.Mem Invert ==========" << endl;
		gpu_start = chrono::high_resolution_clock::now();
		compute_gpu_invert(d, false, true);
		gpu_end = chrono::high_resolution_clock::now();
		gpu_time = chrono::duration_cast<chrono::milliseconds>(gpu_end - gpu_start).count();
		avg[6] += gpu_time;
		//cout << "GPU: " << "Compute took: " << gpu_time << " milliseconds" << endl;

		cout << "========== GPU Tex.Mem Invert =========" << endl;
		gpu_start = chrono::high_resolution_clock::now();
		compute_gpu_invert(d, true, false);
		gpu_end = chrono::high_resolution_clock::now();
		gpu_time = chrono::duration_cast<chrono::milliseconds>(gpu_end - gpu_start).count();
		avg[7] += gpu_time;
		//cout << "GPU: " << "Compute took: " << gpu_time << " milliseconds" << endl;

		cout << "========= GPU Tex+S.Mem Invert ========" << endl;
		gpu_start = chrono::high_resolution_clock::now();
		compute_gpu_invert(d, true, true);
		gpu_end = chrono::high_resolution_clock::now();
		gpu_time = chrono::duration_cast<chrono::milliseconds>(gpu_end - gpu_start).count();
		avg[8] += gpu_time;
		//cout << "GPU: " << "Compute took: " << gpu_time << " milliseconds" << endl;
	}
	for (int i=0;i<9;i++)
	{
		cout << "Average "<<runs[i]<<": " << avg[i] / repeat << " milliseconds" << endl;
	}
	free_data(d);
	cudaDeviceReset();
}

void free_data(CSC_t &data)
{
	free(data->destination_offsets);
	free(data->source_indices);
	free(data->out_degrees);
	free(data);
}

void read_data(const char* path, int node_count, CSC_t &data)
{
	data = (CSC_t)malloc(sizeof(struct CSC_st));
	data->source_indices = (int*)malloc(INT_SIZE);
	data->nedges = 0;
	data->nvertices = 0;
	data->destination_offsets = (int*)malloc((1 + node_count)*INT_SIZE);
	data->out_degrees = (int*)malloc(node_count*INT_SIZE);
	auto start_load = chrono::high_resolution_clock::now();


#ifdef _WIN32
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(path)) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			if (ent->d_type == DT_REG)
			{
				cout << "+-->Reading file: " << ent->d_name << endl;
				string s = path;
				s = s + ent->d_name;
				read_from_file(s.c_str(), data);
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("could not open directory");
		exit(1);
	}
#else
	struct dirent **namelist;
	int n;
	n = scandir(path, &namelist, NULL, alphasort);
	if (n == -1) {
		perror("scandir");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < n; ++i)
	{
		if (namelist[i]->d_type == DT_REG)
		{
			cout << "+-->Reading file: " << namelist[i]->d_name << endl;
			string s = path;
			s = s + namelist[i]->d_name;
			read_from_file(s.c_str(), data);
		}

		free(namelist[i]);
	}
	free(namelist);
#endif	
	for (int i = data->nvertices; i < node_count; ++i)
	{
		data->destination_offsets[data->nvertices++] = data->nedges;
	}
	data->destination_offsets[data->nvertices] = data->nedges;
	cout << "data->destination_offsets[" << data->nvertices << "]=" << data->destination_offsets[data->nvertices] << endl;
	auto end_load = chrono::high_resolution_clock::now();
	cout << "Loading took: " << chrono::duration_cast<chrono::milliseconds>(end_load - start_load).count() << " milliseconds" << endl;
}

void read_from_file(const char *fname, CSC_t &data)
{
	auto start_load = chrono::high_resolution_clock::now();
	char buf[BUFFER_SIZE + 1];
	int fd = open(fname, O_RDONLY);
	char* last = (char*)malloc(1);
	last[0] = '\0';
	int lines = 0;
	int size = data->nedges + 1;
	while (size_t bytes_read = read(fd, buf, BUFFER_SIZE))
	{
		char* b = buf;
		int read_count = 0;
		while (read_count<bytes_read)
		{
			char * post = strchr(b, '\n');
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
				int dest = -1;

				char * token = strtok(last, TOEKNS);
				while (token != NULL)
				{
					if (dest == -1)
					{
						dest = atoi(token);
						for (size_t i = data->nvertices; i <= dest; i++)
						{
							data->destination_offsets[data->nvertices++] = data->nedges;
						}
					}
					else
					{
						int src = atoi(token);
						if (data->nedges == size - 1)
						{
							size = size * 2 + 1;
							data->source_indices = (int*)realloc(data->source_indices, size * INT_SIZE);
						}

						data->source_indices[data->nedges++] = src;
						data->out_degrees[src]++;
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
	auto end_load = chrono::high_resolution_clock::now();
	cout << "|   +--> " << lines << " lines Took: " << chrono::duration_cast<chrono::milliseconds>(end_load - start_load).count() << " milliseconds" << endl;
}