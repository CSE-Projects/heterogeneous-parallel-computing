#include<stdio.h>
#include<cuda.h>


// setting array size
#define ARRAY_SIZE 20000
#define BLOCK_SIZE 1024

// helper function for calculating upper ceil of division
int upper_ceil(int numerator, int denominator) {
    if(numerator % denominator == 0){
        return numerator/denominator;
    }
    return (numerator/denominator) + 1;
}


/**
 * Kernel to compute the addition of two vectors
 */
__global__ void vector_addition(float *device_arrA, float *device_arrB, float *device_arrC) {
    // get index for which this thread works on
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    // check boundary constraint
    if(index < ARRAY_SIZE) {
        device_arrC[index] = device_arrA[index] + device_arrB[index];
    }
}


// Main function
int main() {
    // host arrays
    float* host_arrA;
    float* host_arrB;
    float* host_arrC;

    // device variables
    float* device_arrA;
    float* device_arrB;
    float* device_arrC;

    // allocate memory on host
    host_arrA = (float *) malloc(ARRAY_SIZE * sizeof(float));
    host_arrB = (float *) malloc(ARRAY_SIZE * sizeof(float));
    host_arrC = (float *) malloc(ARRAY_SIZE * sizeof(float));

    // initialization of host arrays
    for(int i = 0; i < ARRAY_SIZE; ++i) {
        host_arrA[i] = i + 3;
        host_arrB[i] = 2*i + 1;
    }
    printf("Input A: \n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", host_arrA[i]);
    }
    printf("\n");
    printf("Input B: \n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", host_arrB[i]);
    }
    printf("\n");

    // allocate memory on device with error handling
    cudaError_t err  = cudaMalloc((void **)&device_arrA,  ARRAY_SIZE * sizeof(float));                                              \
    if(err != cudaSuccess) {                                                 
        printf( "\nError:  %s ", cudaGetErrorString(err));
        return 0;                   
    }
    err  = cudaMalloc((void **)&device_arrB,  ARRAY_SIZE * sizeof(float)); 
    if(err != cudaSuccess) {                                                 
        printf( "\nError:  %s ", cudaGetErrorString(err));
        return 0;                   
    }
    err  = cudaMalloc((void **)&device_arrC,  ARRAY_SIZE * sizeof(float)); 
    if(err != cudaSuccess) {                                                 
        printf( "\nError:  %s ", cudaGetErrorString(err));
        return 0;                   
    }

    // copy host memory data to device
    cudaMemcpy(device_arrA, host_arrA, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_arrB, host_arrB, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // number of blocks
    int blocks = upper_ceil(ARRAY_SIZE, BLOCK_SIZE);

    // invoke CUDA kernel
    vector_addition<<<blocks, BLOCK_SIZE>>>(device_arrA, device_arrB, device_arrC);

    // copy results from device to host
    cudaMemcpy(host_arrC, device_arrC, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result: \n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", host_arrC[i]);
    }

    // free device and host memory
    cudaFree(device_arrA);
    cudaFree(device_arrB);
    cudaFree(device_arrC);
    free(host_arrA);
    free(host_arrB);
    free(host_arrC);

    return 0;
}