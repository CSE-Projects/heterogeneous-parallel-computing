#include<stdio.h> 
#include<cuda.h>
#include "wb.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Clamp(a, start, end) (max(min(a, end), start))
#define value(arry, i, j, k) (arry[((i)*width + (j)) * depth + (k)])
#define output(i, j, k) value(deviceOutputData, i, j, k)
#define input(i, j, k) value(deviceInputData, i, j, k)
#define shared_data(i, j, k) shared_data[i*121 + j*11 + k]

__global__ void stencil(float *deviceOutputData, float *deviceInputData, int width, int height,
                         int depth, int k) {
  //@@ INSERT CODE HERE

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  int x = threadIdx.x;
  int y = threadIdx.y;

  __shared__ float shared_data[11*11*3];

  if (i < height && j < width) {

    shared_data(x+1, y+1, 0) = input(i, j, k - 1);
    shared_data(x+1, y+1, 1) = input(i, j, k);
    shared_data(x+1, y+1, 2) = input(i, j, k + 1);

    if (x == 0 && i - 1 >= 0) {
      shared_data(x, y+1, 0) = input(i-1, j, k - 1);
      shared_data(x, y+1, 1) = input(i-1, j, k);
      shared_data(x, y+1, 2) = input(i-1, j, k + 1);
    }
    if (x == 8 && i + 1 < height) {
      shared_data(10, y+1, 0) = input(i+1, j, k - 1);
      shared_data(10, y+1, 1) = input(i+1, j, k);
      shared_data(10, y+1, 2) = input(i+1, j, k + 1);
    }
    if (y == 0 && j - 1 >= 0) {
      shared_data(x+1, y, 0) = input(i, j-1, k - 1);
      shared_data(x+1, y, 1) = input(i, j-1, k);
      shared_data(x+1, y, 2) = input(i, j-1, k + 1);
    }
    if (y == 8 && j + 1 < width) {
      shared_data(x+1, 10, 0) = input(i, j+1, k - 1);
      shared_data(x+1, 10, 1) = input(i, j+1, k);
      shared_data(x+1, 10, 2) = input(i, j+1, k + 1);
    }
  }
  
  __syncthreads();

  if(i < 1 || i >= height -1 || j < 1 || j >= width -1) {
    return;
  }

  float res = shared_data(x, y, 0) + shared_data(x, y, 2) + shared_data(x, y + 1, 1) + shared_data(x, y - 1, 1) + shared_data(x + 1, y, 1) + shared_data(x - 1, y, 1) - 6 * shared_data(x, y, 1);
  res = Clamp(res, 0.0, 1.0);
  output(i, j, k) = res;
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData,
                           int width, int height, int depth) {
  //@@ INSERT CODE HERE
  dim3 dimGrid(ceil(height/9.0), ceil(width/9.0), 1);
  dim3 dimBlock(9, 9, 1);

  for (int i = 1; i <= depth - 2; i++) {
    stencil<<<dimGrid, dimBlock>>>(deviceOutputData, deviceInputData, width, height, depth, i);
  }
}


// bool checkSoln(float* a, float* b, int h, int w, int d, float *in)
// {     
//   const float tolerance = 1.5f;
//   for(int i=0; i< w; i++){
//     for (int j = 0; j < h; j++) {
//       for (int k = 0; k < d; k++) {
//         int dex = (j * w + i) * d + k;
//         int error = a[dex] - b[dex];
//         if (error > (1.0f /  wbInternal::kImageColorLimit * tolerance)) {
//           if (error != 0) {
//               printf("(%d, %d, %d): %f %f\n", i, j, k, a[i], b[i]);
//               // return false;
//           }
//         }
//       }
//     }
//   }
//   return true;
// }

int main(int argc, char *argv[]) {

  wbArg_t args;
  int width;
  int height;
  int depth;
  char *inputFile;
  wbImage_t input;
  wbImage_t output;
  float *hostInputData;
  float *hostOutputData;
  float *deviceInputData;
  float *deviceOutputData;

  args = wbArg_read(argc, argv);

  inputFile = wbArg_getInputFile(args, 0);
  input = wbImport(inputFile);

  width  = wbImage_getWidth(input);
  height = wbImage_getHeight(input);
  depth  = wbImage_getChannels(input);

  output = wbImage_new(width, height, depth);

  hostInputData  = wbImage_getData(input);
  hostOutputData = (float*)malloc(height*width*depth);

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
  cudaMalloc((void **)&deviceOutputData, width * height * depth * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputData, hostInputData, width * height * depth * sizeof(float),
      cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(output.data, deviceOutputData, width * height * depth * sizeof(float),
      cudaMemcpyDeviceToHost);
    cudaMemcpy(hostOutputData, deviceOutputData, width * height * depth * sizeof(float),
    cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbSolution(args, output);

  char * eOutputFile = wbArg_getInputFile(args, 1);
  wbImage_t eOutput = wbImport(eOutputFile);
  float * eOutputData  = wbImage_getData(eOutput);

  // if (checkSoln(hostOutputData, eOutputData, height, width, depth, hostInputData)) {
  //   printf("YESS\n");
  // }
                                      
  cudaFree(deviceInputData);
  cudaFree(deviceOutputData);

  wbImage_delete(output);
  wbImage_delete(input);

  return 0;
}