#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include<time.h>

// Setting block size
#define BLOCK_SIZE 16

// Helper function for calculating upper ceil of division
int upper_ceil(int numerator, int denominator) {
    if(numerator % denominator == 0){
        return numerator/denominator;
    }
    return (numerator/denominator) + 1;
}

// Function to check if space can be allocated or not
#define printError(func){                                               \
    cudaError_t E  = func;                                              \
    if(E != cudaSuccess){                                               \
        printf( "\nError at line: %d ", __LINE__);                      \
        printf( "\nError:  %s ", cudaGetErrorString(E));                \
    }                                                                   \
}                                                                       


// Kernel for matrix multiplication
__global__ void MatrixMultiplication(int *device_A, int *device_B, int *device_C, int m, int n, int k){
    
    // Calculating row and col
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int i, sum;

    // Checking for validity
    if(Row<0 || Col<0 || Row>=m || Col>=k){
        return;
    }
    else{
        sum = 0;
        for(i=0;i<n;i++){
            // Calculating sum
            sum = sum + (device_A[Row*n + i] * device_B[k*i + Col]);
        }
        // Assigning value
        device_C[Row*k + Col] = sum;
    }
}

// Function to check if result is correct
int check(int m, int n, int k, int *host_A, int *host_B, int *host_C)
{
    int flag=1, row, col, sum, i;   

    for(row= 0;row<m;row++){
        for(col=0;col<k;col++){
            sum=0;
            for(i=0;i<n;i++){
                sum = sum + host_A[row*n + i] * host_B[col + i*k];
            }

            // Checking if the answer is shared_A expected
            if(host_C[row*k + col] != sum){
                flag=0;
                break;	   
	        }
        }
        if(!flag) break;
    }
    
    // Returning flag
    return flag;
}

int main(){

    // Seeding PRNG
    srand(time(NULL));

    int i;

    // Host Matrices
    int *host_A;
    int *host_B;
    int *host_C;

    // Matrix host_A of size (m,n) and Matrix host_B of size (n,k)
    int m = 512;
    int n = 1024;
    int k = 512;
    
    // Device matrices
    int *device_A;
    int *device_B;
    int *device_C;

    // Allocating memory
    host_A = (int *)malloc(m * n * sizeof(int));
    host_B = (int *)malloc(n * k * sizeof(int));
    host_C = (int *)malloc(m * k * sizeof(int));

    for(i=0;i<m*n;i++){
        // Assigning values
        host_A[i] = rand()%100;
    }
    
    for(i=0;i<n*k;i++){
        // Assigning values
        host_B[i] = rand()%100;
    }
    
    // Allocating memory with error checking
    printError(cudaMalloc((void **)&device_A,  m * n * sizeof(int)));
    printError(cudaMalloc((void **)&device_B,  n * k * sizeof(int)));
    printError(cudaMalloc((void **)&device_C,  m * k * sizeof(int)));

    // Copying values
    cudaMemcpy(device_A, host_A, m * n *  sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, n * k *  sizeof(int), cudaMemcpyHostToDevice);

    // Initializing grid size and block size
    dim3 dimGrid(upper_ceil(k,BLOCK_SIZE), upper_ceil(m,BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);


    // Matrix multiplication
    MatrixMultiplication<<<dimGrid, dimBlock>>>(device_A, device_B, device_C, m, n, k);

    // Copying results
    cudaMemcpy(host_C, device_C, m * k * sizeof(int), cudaMemcpyDeviceToHost);

    // Checking results
    if(check(m, n, k, host_A, host_B, host_C))
      printf("Correct\n");

    else
       printf("Incorrect\n");


    // Freeing memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    free(host_A);
    free(host_B);
    free(host_C);
    
    return 0;
}
