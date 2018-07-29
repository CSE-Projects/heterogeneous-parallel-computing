#include <stdio.h>

int main(){

        printf("\n\n");
        
        // CUDA device properties struct
	cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        // Device name 
        printf("Device Name: %s\n",prop.name);

        // Compute capability
        printf("Compute capability: major: %d\tminor: %d\n\n", prop.major, prop.minor);

        // Maximum block dimension
        printf("Maximum block dimension in x: %d\n", prop.maxThreadsDim[0]);
        printf("Maximum block dimension in y: %d\n", prop.maxThreadsDim[1]);
        printf("Maximum block dimension in z: %d\n\n", prop.maxThreadsDim[2]);

        // Maximum block dimension
        printf("Maximum grid dimension in x: %d\n", prop.maxGridSize[0]);
        printf("Maximum grid dimension in y: %d\n", prop.maxGridSize[1]);
        printf("Maximum grid dimension in z: %d\n\n", prop.maxGridSize[2]);

        // Shared Memory Per Block
        printf("Shared memory per block: %zu\n", prop.sharedMemPerBlock);

        // Total Global Memory
        printf("Total global memory: %zu\n", prop.totalGlobalMem);

        // Total Constant Memory
        printf("Total constant memory: %zu\n\n", prop.totalConstMem);

        // Warp size
        printf("Warp size: %d\n", prop.warpSize);
        
        printf("\n\n");
}
