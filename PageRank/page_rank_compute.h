#pragma once

#include "page_rank.h"

void compute_cpu(CSC_t &data);
void compute_gpu(CSC_t &h_data, bool useTexture, bool useShared);
void compute_gpu_invert(CSC_t &h_data, bool useTexture, bool useShared);
