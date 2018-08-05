#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>

int main(int argc, char *argv[]) {

	// declare variables

    // float *hostInput1 = nullptr;
    // float *hostInput2 = nullptr;
	// float *hostOutput = nullptr;
	thrust::host_vector<float> hostInput1;
	thrust::host_vector<float> hostInput2;
	thrust::host_vector<float> hostOutput;
    int inputLength;

    /* parse the input arguments */
    //@@ Insert code here
    // file pointers
    FILE *input_file1 = fopen(argv[1],"r");
  	FILE *input_file2 = fopen(argv[2],"r");
	FILE *e_output_file = fopen(argv[3],"r");
	FILE *output_file = fopen(argv[4],"w");
	// input length
    fscanf(input_file1, "%d", &inputLength);
	fscanf(input_file2, "%d", &inputLength);

    // Import host input data
    //@@ Read data from the raw files here
    //@@ Insert code here
    // hostInput1 = (float*) malloc(inputLength * sizeof(float));
    // hostInput2 = (float*) malloc(inputLength * sizeof(float));

	// fill host arrays
	float readVal;
    for (int i = 0; i < inputLength; ++i) {
		fscanf(input_file1, "%f", &readVal);
		hostInput1.push_back(readVal);
		fscanf(input_file2, "%f", &readVal);
		hostInput2.push_back(readVal);
		hostOutput.push_back(0);
    }

    // Declare and allocate host output
    //@@ Insert code here
    // hostOutput = (float *) malloc(inputLength * sizeof(float));

    // Declare and allocate thrust device input and output vectors
	//@@ Insert code here
	thrust::device_vector<float> deviceInput1(inputLength);
  	thrust::device_vector<float> deviceInput2(inputLength);
  	thrust::device_vector<float> deviceOutput(inputLength);

    // Copy to device
	//@@ Insert code here
	thrust::copy(hostInput1.begin(), hostInput1.end(), deviceInput1.begin());
	thrust::copy(hostInput2.begin(), hostInput2.end(), deviceInput2.begin());

    // Execute vector addition
    //@@ Insert Code here
	thrust::transform(deviceInput1.begin(), deviceInput1.end(), deviceInput2.begin(), deviceOutput.begin(), thrust::plus<float>());
    /////////////////////////////////////////////////////////

    // Copy data back to host
    //@@ Insert code here
	thrust::copy(deviceOutput.begin(), deviceOutput.end(), hostOutput.begin());

	 // write result to expected output file
	fprintf(output_file, "%d", inputLength);
	for (int i = 0; i < inputLength; ++i) {
		fprintf(output_file, "\n%.2f", hostOutput[i]);
	}

	// close file pointers
	fclose(input_file1);
	fclose(input_file2);
	fclose(output_file);
	fclose(e_output_file);

	// check output difference
	e_output_file = fopen(argv[3],"r");
	output_file = fopen(argv[4],"r");
	bool flag = false;
	float read1, read2;
	while (fscanf(e_output_file, "%f", &read1) != EOF) {
		fscanf(output_file, "%f", &read2);
		if (read1 != read2) {
			flag = true;
			break;
		}
	}
	if (flag) {
		printf("Outputs are different\n");
	}
	else {
		printf("Outputs are the same\n");
	}
	fclose(output_file);
	fclose(e_output_file);
	
    // free(hostInput1);
    // free(hostInput2);
    // free(hostOutput);
    return 0;
}
