#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "wb.h"

#define BLUR_SIZE 1

//@@ INSERT CODE HERE
// Kernel to implement gaussian blur to convert input_image to output_image  
__global__ void gauss_blur_image(float* input, float* output, int image_height, int image_width) {
	// calculating x coordinate and the y coordinate
	// calculated on the basis of block index and thread index
	int X = blockIdx.y * blockDim.y + threadIdx.y;
    int Y = blockIdx.x * blockDim.x + threadIdx.x;

	// compute blurred value in the blur size grid
    float blurred_val1=0, blurred_val2 = 0, blurred_val3 = 0;
	int count=0;

	// considering 9 values (consecutive in the input)
    for(int i =  X - BLUR_SIZE; i <= X + BLUR_SIZE; i++) {
      	for(int j = Y - BLUR_SIZE; j <= Y + BLUR_SIZE; j++) {
			// checking if the index is valid or not
			if(i >= 0 && j >= 0 && i < image_height && j < image_width) { 
				blurred_val1 += input[3 * (i*image_width + j)];
				blurred_val2 += input[3 * (i*image_width + j) + 1];
				blurred_val3 += input[3 * (i*image_width + j) + 2];
				// incrementing the count of valid indices
				count++;  
			}
      	}	
	}

	// average
    blurred_val1 = blurred_val1/ count;
    blurred_val2 = blurred_val2/ count;
    blurred_val3 = blurred_val3/ count;
	// update output
	blurred_val1 = blurred_val1/ count;
    blurred_val2 = blurred_val2/ count;
    blurred_val3 = blurred_val3/ count;

    output[ 3 * (X*image_width + Y)] = blurred_val1;
    output[ 3 * (X*image_width + Y) + 1] = blurred_val2;
    output[ 3 * (X*image_width + Y) + 2] = blurred_val3;
}

int main(int argc, char *argv[]) {

	// declare arguments
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *deviceInputImageData;
	float *deviceOutputImageData;

	
	/* parse the input arguments */
	//@@ Insert code here
	wbArg_t args = wbArg_read(argc, argv);

	inputImageFile = wbArg_getInputFile(args, 1);

	inputImage = wbImport(inputImageFile);

	// The input image is in grayscale, so the number of channels
	// is 1
	imageWidth  = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);

	// Since the image is monochromatic, it only contains only one channel
	outputImage = wbImage_new(imageWidth, imageHeight, 0);

	hostInputImageData  = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData,
				imageWidth * imageHeight * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData,
				imageWidth * imageHeight * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData,
				imageWidth * imageHeight * sizeof(float),
				cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");
	//  initialize thread block and kernel grid dimensions
	dim3 threads(16, 16, 1);
	dim3 blocks(imageHeight/16 + 1, imageWidth/16 + 1, 1);
  
	// invoke CUDA kernel
	gauss_blur_image<<<threads, blocks>>> (deviceInputImageData, deviceOutputImageData, imageHeight, imageWidth);

	wbTime_stop(Compute, "Doing the computation on the GPU");
	///////////////////////////////////////////////////////

	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
				imageWidth * imageHeight * sizeof(float),
				cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(args, outputImage);

	// free memory
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
