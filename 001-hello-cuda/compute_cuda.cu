/*
* This file is part of the cuda-programming starter tutorial repository
* Author: Manthan C.S.
* Year: 2023
*
* This file contains the code to add two vectors together.
*/

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel function to add two vectors
__global__ void add(int *a, int *b, int *c, int n) {
    /*
    * This function will be running on seperate threads and blocks.
    * We have to get the index of the thread to access the elements of the vectors.
    *
    * The index of the thread is calculated as follows:
    * index = threadId + blockId * blockDim
    * The index is used to access the elements of the vectors.
    */
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}


int main() {
    int n = 100000;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    // Allocate memory on the host
    a = (int *) malloc(n * sizeof(int));
    b = (int *) malloc(n * sizeof(int));
    c = (int *) malloc(n * sizeof(int));

    // Allocate memory on the device
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));

    // Initialize the vectors
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Copy the vectors to the device
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    add<<<ceil(n / 256.0), 256>>>(d_a, d_b, d_c, n);

    // Copy the result back to the host
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < n/100; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Free the memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}