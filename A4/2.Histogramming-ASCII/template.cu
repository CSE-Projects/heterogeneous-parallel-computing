#include <stdio.h>
#include <cuda.h>
#include "wb.h"

#define NUM_BINS 128

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

/**
 * Kernel to perform the histogramming of the input data of ASCII characters
 */
__global__
void HistogramAscii(unsigned *deviceInput, unsigned *deviceBins, int inputLength) {
  
  // calculate index at which this thread will function
  unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
  
  // create a private array of bins for each thread block
  __shared__ unsigned shared_bins[NUM_BINS];

  // each thread will handle a bin initialization starting at its index in a block
  // and then at offsets of (number of threads in a block)
  unsigned i = threadIdx.x;
  while (i < NUM_BINS) {
    shared_bins[i] = 0;
    i += blockDim.x; 
  }
  // wait for all threads to complete initialization
  __syncthreads();

  // perform histogramming of the input data
  i = index;  
  // stide block length is all the threads generated  
  int stride = blockDim.x * gridDim.x;
  // considering the input data to be divide into divisions of stride length
  // here each thread handles a input data starting from its index overall 
  // and then will skip over stride length and take in the next input data from next division
  while (i < inputLength) {
    atomicAdd(&shared_bins[deviceInput[i]], 1);
    i += stride;
  }
  // wait for all threads to complete
  __syncthreads();

  // each thread will handle a private bin transfer to global memory starting at its index in a block
  // and then at offsets of (number of threads in a block)
  i = threadIdx.x;
  while (i < NUM_BINS) {
    atomicAdd(&deviceBins[i], shared_bins[i]);
    i += blockDim.x; 
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  // hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
  //                                      &inputLength);
  // reading data from input file as the above line was not working for spaces
  FILE *handle = fopen(wbArg_getInputFile(args, 0), "r");
  char c;
  fscanf (handle, "%u", &inputLength);
  // consume new line
  c = fgetc(handle); 
  hostInput = (unsigned *)malloc(sizeof(unsigned)*inputLength);
  int i = 0;
  while ((c = fgetc(handle)) != EOF) {
    hostInput[i++] = (unsigned int) c;
  }
  fflush(handle);
  fclose(handle);

  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  printf("First 10 input values:\n");
  for (int i = 0; i < 10; i++) {
    printf("(%c : %u)", hostInput[i], hostInput[i]);
  }
  printf("\n");

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof (unsigned int), cudaMemcpyHostToDevice);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Perform kernel computation here
  HistogramAscii<<<2, 2>>>(deviceInput, deviceBins, inputLength);
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof (unsigned int), cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");
  
  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  wbTime_stop(GPU, "Freeing GPU Memory");

  // Verify correctness
  // -----------------------------------------------------
  // wbSolution(args, hostBins, NUM_BINS);
  int num;
  unsigned *eOutput = (unsigned int *)wbImport(wbArg_getInputFile(args, 1),
                                       &num);
  bool diff = false;
  for (int i = 0; i < NUM_BINS; i++) {
      if (eOutput[i] != hostBins[i]) {
        printf("%d: %u %u\n", i, hostBins[i], eOutput[i]);
        diff = true;
        break;
      }
  }
  if (!diff) {
    printf("Solution is correct");
  }
  else {
    printf("Solution doesn't match");
  }
  
  free(hostBins);
  free(hostInput);
  return 0;
}
