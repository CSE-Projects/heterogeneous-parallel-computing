#include <stdio.h>
#include <cuda.h>

// setting array and block size
#define ARRAY_SIZE 1048576
#define BLOCK_SIZE 1024


// helper function for calculating upper ceil of division
int upper_ceil(int numerator, int denominator) {
    if(numerator % denominator == 0){
        return numerator/denominator;
    }
    return (numerator/denominator) + 1;
}


/**
 *  Kernel code to compute the vector reduction sum
 */
__global__ void vector_sum_reduction(float *device_arr, float *device_sum) {

    // array in shared memory declaration
    __shared__ float shared_data[BLOCK_SIZE];
    // thread id
    unsigned int thread_id = threadIdx.x;
    // index of host array mapped to the thread
    unsigned int index = blockDim.x * blockIdx.x + thread_id;

    // initializing shared memory in the block corresponding to this thread  
    shared_data[thread_id] = device_arr[index];
    // wait for the all threads to complete filling the array in shared memory
    __syncthreads();

    // setting offsets in multiple of 2
    for(int offset = 1; offset < blockDim.x; offset *= 2) {
        // finding the idx to be incremented
        unsigned int idx = 2 * thread_id * offset;
        // check boundary of idx
        if(idx < blockDim.x) {
            // incrementing the shared data at index idx by the shared data at an offset
            /* shared data at idx and shared data at an offset hold the cumulative sum of fixed no of 
            elements to the right and values at these indices */
            // refer diagram in the report 
            shared_data[idx] += shared_data[idx + offset];
            // now shared_data[idx] contains the sum from element at idx to index of the rightmost element taken into account by shared_data at offset 
        }
        // making sure all adds at one stage are done
        __syncthreads();
    }
    
    // adding the block sums (present at the first index of array in shared data for each block) to the device sum
    if(thread_id == 0) {
        atomicAdd(device_sum, shared_data[0]);
    }
}


// Main function
int main() {
    
    // host variables
    float *host_arr;
    float *host_sum;
    
    // device variables
    float *device_arr;
    float *device_sum;

    // allocate space in host
    host_arr = (float *) malloc(ARRAY_SIZE * sizeof(float));
    host_sum = (float *) malloc(sizeof(float));
    (*host_sum) = 0;

    // initialize host array elements
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        host_arr[i] = (float)2;
    } 
    
    // allocate device memory with error handling
    cudaError_t err  = cudaMalloc((void **)&device_arr,  ARRAY_SIZE * sizeof(float));;                                               \
    if(err != cudaSuccess) {                                                 
        printf( "\nError:  %s ", cudaGetErrorString(err));
        return 0;                   
    }  
    err = cudaMalloc((void **)&device_sum , sizeof(float));
    if(err != cudaSuccess) {                                                
        printf( "\nError:  %s ", cudaGetErrorString(err));
        return 0;                   
    }

    // copy host memory data to device
    cudaMemcpy(device_arr, host_arr, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_sum, host_sum, sizeof(float), cudaMemcpyHostToDevice);

    // initialize thread block and kernel grid dimensions
    int blocks = upper_ceil(ARRAY_SIZE, BLOCK_SIZE);

    // invoke CUDA kernel
    vector_sum_reduction<<<blocks, BLOCK_SIZE>>>(device_arr, device_sum);

    // copy results from device to host
    cudaMemcpy(host_sum, device_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // print result
    printf("Sum = %f\n", *host_sum);

    // free device and host memory
    cudaFree(device_arr);
    cudaFree(device_sum);
    free(host_arr);
    free(host_sum);

    return 0;
}