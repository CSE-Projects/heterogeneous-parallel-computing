#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include "wb.h"

//@@ define error checking macro here.
#define errCheck(stmt)                                                     			\
	do {                                                                    		\
		cudaError_t err = stmt;                                               		\
		if (err != cudaSuccess) {                                             		\
		printErrorLog(ERROR, "Failed to run stmt ", #stmt);                         \
		printErrorLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
		return -1;                                                          		\
		}                                                                     		\
	} while (0)

//@@ INSERT CODE HERE
__global__ void ConvertRgbtoGray(float *deviceInputImageData, float *deviceOutputImageData, int imageHeight, int imageWidth) {
	// get index for thread to work on
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < imageHeight*imageWidth) {
		// get R, G, B components
		float r = deviceInputImageData[3*index];
		float g = deviceInputImageData[3*index + 1];
		float b = deviceInputImageData[3*index + 2];
		// store gray scale pixel value in output image 
		deviceOutputImageData[index] = (0.21f * r + 0.71f * g + 0.07f * b);
	} 
}

int main(int argc, char *argv[]) {

	int imageChannels;
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

	// get input file
	inputImageFile = wbArg_getInputFile(args, 0);
	inputImage = wbImport(inputImageFile);

	imageWidth  = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	// For this lab the value is always 3
	imageChannels = wbImage_getChannels(inputImage);

	// Since the image is monochromatic, it only contains one channel
	outputImage = wbImage_new(imageWidth, imageHeight, 1);

	hostInputImageData  = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData,
				imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData,
				imageWidth * imageHeight * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData,
				imageWidth * imageHeight * imageChannels * sizeof(float),
				cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ INSERT CODE HERE
	// 1024 threads and no of blocks are calculated hence
	int blocks = ((imageWidth*imageHeight)/1024) + 1; 
	// invoked kernel
	ConvertRgbtoGray<<<blocks, 1024>>> (deviceInputImageData, deviceOutputImageData, imageHeight, imageWidth);

	wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,	
				imageWidth * imageHeight * sizeof(float),
				cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
		
	// int solnItems;
    // float* soln = wbImport(wbArg_getInputFile(args, 1), &solnItems);
	// printf("%d %d", imageHeight*imageWidth, solnItems);
	// This will check for MP6 format but output image is in MP5 format
	// wbSolution(args, outputImage);
	// this is throwing an error 
	// wbSolution(args, hostOutputImageData, imageHeight*imageWidth);

	// Serial computation and checking
	bool gray_scale = true;
	for (int i = 0; i < imageHeight; i++) {
		for (int j = 0; j < imageWidth; j++) {
			int index = i*imageWidth + j;
			float r = hostInputImageData[3*index];
			float g = hostInputImageData[3*index + 1];
			float b = hostInputImageData[3*index + 2];
			float val = (0.21f * r + 0.71f * g + 0.07f * b);
			// check for difference
			if (!wbInternal::wbFPCloseEnough(hostOutputImageData[index], val)) {
                printf("%f %f", hostOutputImageData[index], val);
				gray_scale = false;
				break;
            }
			if (!gray_scale) {
				break;
			}
		}
	}
	if (gray_scale) {
		printf("Solution is Correct\n");
	}
	else {
		printf("Solutions differs\n");
	}

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
