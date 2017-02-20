/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_CU_
#define _MATRIXMUL_KERNEL_CU_

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "matrixmul.h"

#include "device_launch_parameters.h"
#include "device_functions.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	__shared__ float Mds[TILE_WIDTH*TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH*TILE_WIDTH];
	
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = blockIdx.y * blockDim.y + ty;
	int Col = blockIdx.x * blockDim.x + tx;
	
	double Pvalue = 0;
	for (int m = 0; m < (int)ceil((float)M.width / blockDim.x); ++m)
	{
		int mk = (m*blockDim.x + tx);
		if (mk < M.width && Row < P.height)
		{
			Mds[ty*blockDim.x + tx] = M.elements[Row*M.width + mk];
		}
		else
		{
			Mds[ty*blockDim.x + tx] = 0;
		}
		int nk = m*blockDim.y + ty;
		if (nk < N.height && Col < P.width)
		{
			Nds[ty*blockDim.x + tx] = N.elements[(m*blockDim.y + ty)*N.width + Col];
		}
		else
		{
			Nds[ty*blockDim.x + tx] = 0;
		}
		__syncthreads();

		for (int k = 0; k < blockDim.x; ++k)
		{
			Pvalue += (double)Mds[ty*blockDim.x + k] * (double)Nds[k*blockDim.x + tx];
		}
		__syncthreads();
	}
	if (Row < P.height && Col < P.width)
		P.elements[Row*P.width + Col] = Pvalue;

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
