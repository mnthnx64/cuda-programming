/*
* This file is part of the cuda-programming starter tutorial repository
* Author: Manthan C.S.
* Year: 2023
*
* This is script displays the parallelism in CUDA programming specifically Thread Parallelism
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1000

__global__ void add(int *a, int *b, int *c)
{
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(void)
{
    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add<<<N,1>>>(d_a, d_b, d_c);
    /*
    * What <<N,1>> means is that we are launching N blocks of 1 thread each
    * This is a very simple example of thread parallelism
    */

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}