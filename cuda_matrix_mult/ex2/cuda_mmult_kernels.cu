#include "cuda_mmult_kernels.h"

/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 4)
 *  -> Features: none (basic tiled version, using only global memory)
 */
__global__ void matrixMultKernel_global(float* Ad, float* Bd, float* Cd, int n)
{
   int i = blockIdx.x * TILE_SIZE + threadIdx.x;
   int k = blockIdx.y * TILE_SIZE + threadIdx.y;
   
   float Celem = 0;
   
   for(int j=0; j<n; j++) {
      float Aelem = Ad[i*n+j];
      float Belem = Bd[j*n+k];
      Celem += Aelem*Belem;
   }
   
   Cd[i*n+k] += Celem;
}

/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 5)
 *  -> Features:
 *     - tiled matrix multiplication with use of shared memory
 */
__global__ void matrixMultKernel_tiled(float* Ad, float* Bd, float* Cd, int n)
{
   __shared__ float Ads[TILE_SIZE][TILE_SIZE];
   __shared__ float Bds[TILE_SIZE][TILE_SIZE];

   int tx = threadIdx.x;
   int ty = threadIdx.y;
   
   int i = blockIdx.x * TILE_SIZE + tx;
   int k = blockIdx.y * TILE_SIZE + ty;
   
   float Celem = 0;
   
   for(int m=0; m < n/TILE_SIZE; m++) {
      Ads[tx][ty] = Ad[ i*n + m*TILE_SIZE+ty];
      Bds[tx][ty] = Bd[ (m*TILE_SIZE+tx)*n + k];
      __syncthreads();
      
      for(int j=0; j<TILE_SIZE; j++)
	     Celem += Ads[tx][j]*Bds[j][ty];
   
      __syncthreads();
   };

   Cd[i*n+k] += Celem;
}


/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 5)
 *  -> Features:
 *     - tiled matrix multiplication with use of shared memory
 *     - coalesced memory access
 */
__global__ void matrixMultKernel_coalesced(float* Ad, float* Bd, float* Cd, int n)
{
   __shared__ float Ads[TILE_SIZE][TILE_SIZE];
   __shared__ float Bds[TILE_SIZE][TILE_SIZE];

   int tx = threadIdx.x;
   int ty = threadIdx.y;
   
   int i = blockIdx.y * TILE_SIZE + ty;
   int k = blockIdx.x * TILE_SIZE + tx;
   
   float Celem = 0;
   
   for(int m=0; m < n/TILE_SIZE; m++) {
      Ads[ty][tx] = Ad[ i*n + m*TILE_SIZE+tx];
      Bds[ty][tx] = Bd[ (m*TILE_SIZE+ty)*n + k];
      __syncthreads();
      
      for(int j=0; j<TILE_SIZE; j++)
	     Celem += Ads[ty][j]*Bds[j][tx];
   
      __syncthreads();
   };
   Cd[i*n+k] += Celem;
}


/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 5)
 *  -> Features:
 *     - tiled matrix multiplication with use of shared memory
 *     - coalesced memory access
 *     - overlapping loads of subsequent tile pairs (using registers & shared memory)
 */
__global__ void matrixMultKernel_overlap(float* Ad, float* Bd, float* Cd, int n)
{
   __shared__ float Ads[TILE_SIZE][TILE_SIZE];
   __shared__ float Bds[TILE_SIZE][TILE_SIZE];

   float Adr;
   float Bdr;

   int tx = threadIdx.x;
   int ty = threadIdx.y;
   
   int i = blockIdx.y * TILE_SIZE + ty;
   int k = blockIdx.x * TILE_SIZE + tx;
   
   float Celem = 0;
   int m = 0;

   /* load the first tile into the registers */
   Adr = Ad[ i*n + m*TILE_SIZE+tx];
   Bdr = Bd[ (m*TILE_SIZE+ty)*n + k];

   for(m=1; m < n/TILE_SIZE; m++) {
      /* copy current tile from registers into shared memory */
      Ads[ty][tx] = Adr;
      Bds[ty][tx] = Bdr;
      __syncthreads();

      /* load the next tile into the registers */
      Adr = Ad[ i*n + m*TILE_SIZE+tx];
      Bdr = Bd[ (m*TILE_SIZE+ty)*n + k];

      /* compute from shared memory */
      #pragma unroll
      for(int j=0; j<TILE_SIZE; j++)
	     Celem += Ads[ty][j]*Bds[j][tx];
   
      __syncthreads();
   };

   /* compute final tile from register */
   Ads[ty][tx] = Adr;
   Bds[ty][tx] = Bdr;
   __syncthreads();
   #pragma unroll
   for(int j=0; j<TILE_SIZE; j++)
      Celem += Ads[ty][j]*Bds[j][tx];

   Cd[i*n+k] += Celem;
}
