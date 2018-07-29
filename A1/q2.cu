#include<stdio.h>
#include<cuda.h>


// setting matrix size
#define ROW_SIZE 1024
#define COL_SIZE 1024


// host matrices
int host_matrix1[ROW_SIZE][COL_SIZE];
int host_matrix2[ROW_SIZE][COL_SIZE];
int host_matrix3[ROW_SIZE][COL_SIZE];


// helper function for calculating upper ceil of division
int upper_ceil(int numerator, int denominator) {
	if(numerator%denominator==0) return numerator/denominator;
	return numerator/denominator + 1;
}


/**
 * Kernel to compute the addition of two matrices
 */
__global__ void matrix_add(int host_matrix1[][COL_SIZE], int host_matrix2[][COL_SIZE], int host_matrix3[][COL_SIZE]) {
	// get index for which this thread works on
  	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	// check boundary constraint  
  	if( x < ROW_SIZE && y < COL_SIZE )
    	host_matrix3[x][y] = host_matrix2[x][y] + host_matrix1[x][y];
}


int main() {

	int i, j;

	// device variables
	int (*deviceM1)[COL_SIZE];
	int (*deviceM2)[COL_SIZE];
	int (*deviceM3)[COL_SIZE];

	// initialization of host arrays
    for(i=0; i<ROW_SIZE; i++) {
    	for(j=0; j<COL_SIZE; j++) {
        	host_matrix1[i][j] = i;
			host_matrix2[i][j] = j;
			host_matrix3[i][j] = 0;
		  }
	}

	// allocate memory on device with error handling
	cudaError_t err = cudaMalloc((void **)&deviceM1,  ROW_SIZE * COL_SIZE * sizeof(int));
	if(err != cudaSuccess) {                                                 
        printf( "\nError:  %s ", cudaGetErrorString(err));
        return 0;                   
    }
	err = cudaMalloc((void **)&deviceM2,  ROW_SIZE * COL_SIZE * sizeof(int));
	if(err != cudaSuccess) {                                                 
        printf( "\nError:  %s ", cudaGetErrorString(err));
        return 0;                   
    }
	err = cudaMalloc((void **)&deviceM3,  ROW_SIZE * COL_SIZE * sizeof(int));
	if(err != cudaSuccess) {                                                 
        printf( "\nError:  %s ", cudaGetErrorString(err));
        return 0;                   
	}
	
	// copy host memory data to device
    cudaMemcpy(deviceM1, host_matrix1, ROW_SIZE * COL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceM2, host_matrix2, ROW_SIZE * COL_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	// number of blocks and threads
	dim3 thread_number(16, 16);
	dim3 block_number(upper_ceil(ROW_SIZE,16.0), upper_ceil(COL_SIZE,16.0));

	// invoke CUDA kernel
    matrix_add<<<block_number, thread_number>>>(deviceM1, deviceM2, deviceM3);

	// copy results from device to host
    cudaMemcpy(host_matrix3, deviceM3, ROW_SIZE * COL_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	// printing initial 25 values for verification
	printf("Result: \n");
	for(i=0;i<5;++i){
		for(j=0;j<5;++j){
			printf("%d ",host_matrix3[i][j]);
		}
		printf("\n");
	}

	// free device memory
    cudaFree(deviceM1);
    cudaFree(deviceM2);
    cudaFree(deviceM3);
}