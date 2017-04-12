#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEFAULT_THRESHOLD  8000

#define DEFAULT_FILENAME "BWstop-sign.ppm"
#define MASK_WIDTH 3
#define HALF_MASK_WIDTH (MASK_WIDTH/2)
#define TILE_WIDTH 16
#define INPUT_TILE_WIDTH (TILE_WIDTH+(2*HALF_MASK_WIDTH))

__constant__ int Mx[MASK_WIDTH*MASK_WIDTH];
__constant__ int My[MASK_WIDTH*MASK_WIDTH];

__global__ void filter(int* output, const unsigned int* const input, const int xsize, const int ysize, const int thresh)
{
	__shared__ int S[INPUT_TILE_WIDTH*INPUT_TILE_WIDTH];

	int2 global = make_int2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);
	unsigned int t = global.y*xsize + global.x;
	int2 tile = make_int2(threadIdx.x + HALF_MASK_WIDTH, threadIdx.y + HALF_MASK_WIDTH);


	if (global.x >= xsize || global.y >= ysize)
	{
		return;
	}

	// Offset by half tile width in shared memory becuase of halo, px(0,0) stored at s[1,1]
	S[(tile.y)*INPUT_TILE_WIDTH + tile.x] = input[t];

	// Border threads fetch the halo or ghost pixels
	// Left halo
	if (threadIdx.x < HALF_MASK_WIDTH)
	{
		S[(tile.y)*INPUT_TILE_WIDTH + threadIdx.x] = global.x < HALF_MASK_WIDTH ? 0 : input[t - HALF_MASK_WIDTH];
		// Top Left corner
		if (threadIdx.y < HALF_MASK_WIDTH)
		{
			S[(threadIdx.y)*INPUT_TILE_WIDTH + threadIdx.x] =
				(global.y < HALF_MASK_WIDTH || global.x < HALF_MASK_WIDTH) ? 0 : input[t - HALF_MASK_WIDTH - xsize];
		}
	}

	// Top Halo
	if (threadIdx.y < HALF_MASK_WIDTH)
	{
		S[(threadIdx.y)*INPUT_TILE_WIDTH + tile.x] = global.y < HALF_MASK_WIDTH ? 0 : input[t - xsize];
		// Top Right corner
		if (threadIdx.x >= blockDim.x - HALF_MASK_WIDTH)
		{
			S[(threadIdx.y)*INPUT_TILE_WIDTH + tile.x + HALF_MASK_WIDTH]
				= global.y < HALF_MASK_WIDTH || global.x >= xsize - HALF_MASK_WIDTH ? 0 : input[t - xsize + HALF_MASK_WIDTH];
		}
	}

	// Right Halo
	if (threadIdx.x >= blockDim.x - HALF_MASK_WIDTH)
	{
		S[(tile.y)*INPUT_TILE_WIDTH + tile.x + HALF_MASK_WIDTH]
			= global.x >= xsize - HALF_MASK_WIDTH ? 0 : input[t + HALF_MASK_WIDTH];
		// Bottom Right Corner
		if (threadIdx.y >= blockDim.y - HALF_MASK_WIDTH)
		{
			S[(tile.y + HALF_MASK_WIDTH)*INPUT_TILE_WIDTH + tile.x + HALF_MASK_WIDTH]
				= global.x >= xsize - HALF_MASK_WIDTH || global.y >= ysize - HALF_MASK_WIDTH ? 0 : input[t + HALF_MASK_WIDTH + xsize];
		}
	}

	// Bottom Halo
	if (threadIdx.y >= blockDim.y - HALF_MASK_WIDTH)
	{
		S[(tile.y + HALF_MASK_WIDTH)*INPUT_TILE_WIDTH + tile.x]
			= global.y >= ysize - HALF_MASK_WIDTH ? 0 : input[t + xsize];
		// Bottom Left corner
		if (threadIdx.x < HALF_MASK_WIDTH)
		{
			S[(tile.y + HALF_MASK_WIDTH)*INPUT_TILE_WIDTH + threadIdx.x]
				= global.x < HALF_MASK_WIDTH || global.y >= ysize - HALF_MASK_WIDTH ? 0 : input[t + xsize - HALF_MASK_WIDTH];
		}
	}

	__syncthreads();

	// HALF_MASK_WIDTH border
	if (global.x < HALF_MASK_WIDTH || global.x >= xsize - HALF_MASK_WIDTH || global.y <HALF_MASK_WIDTH || global.y >= ysize - HALF_MASK_WIDTH)
	{
		return;
	}

	int px = 0;
	int py = 0;
	int start_x = tile.x - HALF_MASK_WIDTH;
	int start_y = tile.y - HALF_MASK_WIDTH;
	for (int r = 0; r < MASK_WIDTH; r++)
	{
		int row = start_y + r;
		for (int c = 0; c < MASK_WIDTH; c++)
		{
			int col = start_x + c;
			px += S[row*INPUT_TILE_WIDTH + col] * Mx[r*MASK_WIDTH + c];
			py += S[row*INPUT_TILE_WIDTH + col] * My[r*MASK_WIDTH + c];
		}
	}

	if ((px*px + py*py) > thresh)
	{
		output[t] = 255;
	}
	else
	{
		output[t] = 0;
	}
}

unsigned int *read_ppm( char *filename, int * xsize, int * ysize, int *maxval ){
  
	if ( !filename || filename[0] == '\0') {
		fprintf(stderr, "read_ppm but no file name\n");
		return NULL;  // fail
	}

	FILE *fp;

	fprintf(stderr, "read_ppm( %s )\n", filename);
	fp = fopen( filename, "rb");
	if (!fp) 
	{
		fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
		return NULL; // fail 
	}

	char chars[1024];
	//int num = read(fd, chars, 1000);
	int num = fread(chars, sizeof(char), 1000, fp);

	if (chars[0] != 'P' || chars[1] != '6') 
	{
		fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
		return NULL;
	}

	unsigned int width, height, maxvalue;


	char *ptr = chars+3; // P 6 newline
	if (*ptr == '#') // comment line! 
	{
		ptr = 1 + strstr(ptr, "\n");
	}

	num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
	fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
	*xsize = width;
	*ysize = height;
	*maxval = maxvalue;
  
	unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
	if (!pic) {
		fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
		return NULL; // fail but return
	}

	// allocate buffer to read the rest of the file into
	int bufsize =  3 * width * height * sizeof(unsigned char);
	if ((*maxval) > 255) bufsize *= 2;
	unsigned char *buf = (unsigned char *)malloc( bufsize );
	if (!buf) {
		fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
		return NULL; // fail but return
	}

	// really read
	char duh[80];
	char *line = chars;

	// find the start of the pixel data. 
	sprintf(duh, "%d\0", *xsize);
	line = strstr(line, duh);
	//fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *ysize);
	line = strstr(line, duh);
	//fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *maxval);
	line = strstr(line, duh);
	
	fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
	line += strlen(duh) + 1;

	long offset = line - chars;
	//lseek(fd, offset, SEEK_SET); // move to the correct offset
	fseek(fp, offset, SEEK_SET); // move to the correct offset
	//long numread = read(fd, buf, bufsize);
	long numread = fread(buf, sizeof(char), bufsize, fp);
	fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize); 

	fclose(fp);
	
	int pixels = (*xsize) * (*ysize);
	for (int i=0; i<pixels; i++) 
		pic[i] = (int) buf[3*i];  // red channel
	
	return pic; // success
}

void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
	FILE *fp;
	int x,y;
  
	fp = fopen(filename, "wb");
	if (!fp) 
	{
		fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n", filename);
		exit(-1); 
	}
  
	fprintf(fp, "P6\n"); 
	fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
  
	int numpix = xsize * ysize;
	for (int i=0; i<numpix; i++) {
		unsigned char uc = (unsigned char) pic[i];
		fprintf(fp, "%c%c%c", uc, uc, uc); 
	}

	fclose(fp);
}

void compareImages(int* expected, int* actual, int xsize, int ysize)
{
	char* result = "PASSED";
	for(int i=0;i<ysize;i++)
	{
		for(int j=0; j<xsize;j++)
		{
			if(expected[i*xsize + j] !=actual[i*xsize+j])
			{
				printf("Mismatch at (%d, %d): Expected/Actual = %d/%d\n", i, j, expected[i*xsize + j], actual[i*xsize + j]);
				result = "FAILED";
			}
		}
	}
	printf("Test %s\n", result);
}

int main(int argc, char **argv)
{
	int thresh = DEFAULT_THRESHOLD;
	char *filename;
	filename = strdup(DEFAULT_FILENAME);

	if (argc > 1) {
		if (argc == 3) { // filename AND threshold
			filename = strdup(argv[1]);
			thresh = atoi(argv[2]);
		}
		if (argc == 2) { // default file but specified threshhold
			thresh = atoi(argv[1]);
		}
		fprintf(stderr, "file %s    threshold %d\n", filename, thresh);
	}

	int xsize, ysize, maxval;
	unsigned int *pic = read_ppm(filename, &xsize, &ysize, &maxval);

	printf("Compute Gold start\n");
	int numbytes = xsize * ysize * sizeof(int);
	int *result = (int *)malloc(numbytes);
	if (!result) {
		fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
		exit(-1); // fail
	}

	int i, j, magnitude, sum1, sum2;

	for (int row = 0; row < ysize; row++) {
		for (int col = 0; col < xsize; col++) {
			result[row*xsize + col] = 0;
		}
	}

	for (i = 1; i < ysize - 1; i++) {
		for (j = 1; j < xsize - 1; j++) {

			int offset = i*xsize + j;

			sum1 = pic[xsize * (i - 1) + j + 1] - pic[xsize*(i - 1) + j - 1]
				+ 2 * pic[xsize * (i)+j + 1] - 2 * pic[xsize*(i)+j - 1]
				+ pic[xsize * (i + 1) + j + 1] - pic[xsize*(i + 1) + j - 1];

			sum2 = pic[xsize * (i - 1) + j - 1] + 2 * pic[xsize * (i - 1) + j] + pic[xsize * (i - 1) + j + 1]
				- pic[xsize * (i + 1) + j - 1] - 2 * pic[xsize * (i + 1) + j] - pic[xsize * (i + 1) + j + 1];

			magnitude = sum1*sum1 + sum2*sum2;

			if (magnitude > thresh)
				result[offset] = 255;
			else
				result[offset] = 0;
		}
	}
	printf("Compute Gold Complete\n");
	write_ppm("result8000gold.ppm", xsize, ysize, 255, result);
	
	printf("Compute CUDA start\n");
	
	int h_Mx[MASK_WIDTH*MASK_WIDTH] =
	{	
		-1,  0,  1,
		-2,  0,  2,
		-1,  0,  1 
	};
	int h_My[MASK_WIDTH*MASK_WIDTH] =
	{ 
		-1, -2, -1,
		0,  0,  0,
		1,  2,  1
	};
	/*int h_Mx[MASK_WIDTH*MASK_WIDTH] =
	{
		-2,  -1,  0,  1,  2,
		-3,  -2,  0,  2,  3,
		-4,  -3,  0,  3,  4,
		-3,  -2,  0,  2,  3,
		-2,  -1,  0,  1,  2,
	};
	int h_My[MASK_WIDTH*MASK_WIDTH] =
	{
		-2, -3, -4, -3, -2,
		-1, -2, -3, -2, -1,
		 0,  0,  0,  0,  0,
		 1,  2,  3,  2,  1,
		 2,  3,  4,  3,  2
	};*/
	cudaMemcpyToSymbol(Mx, h_Mx, MASK_WIDTH*MASK_WIDTH * sizeof(int));
	cudaMemcpyToSymbol(My, h_My, MASK_WIDTH *MASK_WIDTH * sizeof(int));
	int channelBytes = xsize*ysize * sizeof(int);
	unsigned int *d_pic;
	int* h_result, *d_result;
	h_result = (int*)malloc(channelBytes);
	cudaMalloc(&d_pic, xsize*ysize * sizeof(unsigned int));
	cudaMalloc(&d_result, channelBytes);
	cudaMemcpy(d_pic, pic, xsize*ysize * sizeof(unsigned int), cudaMemcpyHostToDevice);
	dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
	dim3 gridSize((int)ceil((float)xsize / blockSize.x), (int)ceil((float)ysize / blockSize.y), 1);
	filter<<<gridSize, blockSize>>>(d_result, d_pic, xsize, ysize, thresh);	
	cudaMemcpy(h_result, d_result, channelBytes, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();	
	printf("Compute CUDA Complete\n");

	write_ppm("result_cuda.ppm", xsize, ysize, 255, h_result);
	compareImages(result, h_result, xsize, ysize);
	
	cudaFree(d_pic);
	cudaFree(d_result);
	free(pic);
	free(result);
	free(h_result);
	fprintf(stderr, "sobel done\n");
}

