/*
* This file is part of the cuda-programming starter tutorial repository
* Author: Manthan C.S.
* Year: 2023
*
* This is a very basic hello world script in cuda. It is used to demonstrate the basic syntax of a cuda program.
*/
#include <iostream>
/*
* This is a kernel function. It is a function that is executed on the GPU.
* @param: void
* @return: void
*/
__global__ void helloFromGPU(void) {
    printf("Hello World from GPU!\n");
}

/*
* This is the main function. It is executed on the CPU.
* @param: void
* @return: int
*/
int main(void) {
    printf("Hello World from CPU!\n");
    /*
    * The below syntax is called the triple chevron syntax. It is used to call a kernel function from the host.
    * The first argument is the number of blocks and the second argument is the number of threads per block.
    * ex: function<<<blocks, threads per block>>>()
    */
    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}