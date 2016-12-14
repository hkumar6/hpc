#ifndef KERNELS_H
#define KERNELS_H

#include <stdbool.h>

void CSRmatvecmult(int* ptr, int* J, float* Val, int N, int nnz, float* x, float *y, bool bVectorized);
void ELLmatvecmult(int N, int num_cols_per_row , int * indices, float * data , float * x , float * y);

#endif
