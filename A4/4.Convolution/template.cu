#include "wb.h"
#include <stdio.h>
#include "cuda.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

// Function to check if space can be allocated or not
#define printError(func){                                               \
	cudaError_t E  = func;                                                \
	if(E != cudaSuccess){                                              	  \
		printf( "\nError at line: %d ", __LINE__);                          \
		printf( "\nError:  %s ", cudaGetErrorString(E));                    \
	}                                                                     \
} 

//@@ INSERT CODE HERE
__global__ 
void Convolution(float * deviceInputImageData, const float * __restrict__ deviceMaskData,
  float * deviceOutputImageData, int imageChannels,
  int imageWidth, int imageHeight){

  __shared__ float sharedImageTile[TILE_WIDTH][TILE_WIDTH];

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int row = TILE_WIDTH * blockIdx.y + tid_y;
  int col = TILE_WIDTH * blockIdx.x + tid_x;
  int row_start = row-tid_y;
  int col_start = col-tid_x;

  if(row>=imageHeight || col>=imageWidth) return;

  int i, j, k;

  for(i=0;i<imageChannels;++i){

    for(j=0;j<9;++j) for(k=0;k<9;++k) sharedImageTile[j][k] = 0.0;

    sharedImageTile[tid_y][tid_x] = deviceInputImageData[3*(row*imageWidth+col)+i];

    __syncthreads();

    float value = 0.0;

    for(j=-Mask_radius;j<=Mask_radius;++j){
      for(k=-Mask_radius;k<=Mask_radius;++k){

        if(row+j<0 || row+j>=imageHeight || col+k<0 || col+k>=imageWidth) continue;

        if(row+j>=row_start && col+k>=col_start && row+j<row_start+TILE_WIDTH && col+k<col_start+TILE_WIDTH){
          value+=(deviceMaskData[(j+Mask_radius)*Mask_width+k+Mask_radius]*sharedImageTile[tid_y+j][tid_x+k]);
        }
        else{
          value+=(deviceMaskData[(j+Mask_radius)*Mask_width+k+Mask_radius]*deviceInputImageData[3*((row+j)*imageWidth+(col+k))+i]);
        }

      }
    }

    deviceOutputImageData[3*(row*imageWidth+col)+i] = (unsigned char)(floor(clamp(value)*255.0));

  }
    
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float hostMaskData[25] = {0.088000, 0.184000, 0.112000, 0.080000, 0.080000, 0.168000, 0.032000, 0.088000, 0.032000, 0.056000, 0.040000, 0.016000, 0.200000, 0.144000, 0.136000, 0.144000, 0.064000, 0.152000, 0.008000, 0.160000, 0.200000, 0.144000, 0.176000, 0.024000, 0.184000 };
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  // hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  // assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  // assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");

  float ImageArraySize = imageWidth * imageHeight * imageChannels * sizeof(float);
  
  //@@ INSERT CODE HERE
  printError(cudaMalloc((void **)&deviceInputImageData, ImageArraySize));
  printError(cudaMalloc((void **)&deviceOutputImageData, ImageArraySize));
  printError(cudaMalloc((void **)&deviceMaskData, Mask_width * Mask_width * sizeof(float)));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(deviceInputImageData, hostInputImageData, ImageArraySize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutputImageData, hostOutputImageData, ImageArraySize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, Mask_width * Mask_width * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 dimGrid(ceil(imageHeight/16.0),ceil(imageWidth/16.0),1);
  dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
  Convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
                                     deviceOutputImageData, imageChannels,
                                     imageWidth, imageHeight);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ Insert code here
  free(hostInputImageData);
  free(hostOutputImageData);
  free(hostMaskData);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}