#ifndef _MAIN_H_
#define _MAIN_H_

const int BUFFER_SIZE = (16 * 1024);
const char TOEKNS[3] = ": ";
const int INT_SIZE = sizeof(int);
#define DECAY 0.85f

typedef struct
{
	int* indices;
	int* adj;
	int* out_degrees;
	int node_count;
	int adj_length = -1;
} Data;


#endif
