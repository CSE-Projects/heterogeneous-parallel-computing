#include <stdio.h>
#include <cuda.h>
#include "wb.h"

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength, num_bins;
  unsigned int *hostInput, *hostBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  // Copy the input to the GPU
  wbTime_start(GPU, "Allocating GPU memory");
  //@@ Insert code here
  thrust::device_vector<unsigned int> deviceInput(hostInput, hostInput + inputLength);  
  wbTime_stop(GPU, "Allocating GPU memory");

  // Determine the number of bins (num_bins) and create space on the host
  //@@ insert code here
  // sort the array  
  thrust::sort(deviceInput.begin(), deviceInput.end());
  // max element (at the end) will determine the bin size 
  num_bins = deviceInput.back() + 1;
  hostBins = (unsigned int *)malloc(num_bins * sizeof(unsigned int));

  // Allocate a device vector for the appropriate number of bins
  //@@ insert code here
  thrust::device_vector<unsigned int> deviceBins(num_bins);

  // Create a cumulative histogram. Use thrust::counting_iterator and
  // thrust::upper_bound
  //@@ Insert code here
  thrust::counting_iterator<unsigned int> c_it(0);
  thrust::upper_bound(deviceInput.begin(), deviceInput.end(), c_it, c_it + num_bins, deviceBins.begin());

  // Use thrust::adjacent_difference to turn the culumative histogram
  // into a histogram.
  //@@ insert code here.
  thrust::adjacent_difference(deviceBins.begin(), deviceBins.end(), deviceBins.begin());

  // Copy the histogram to the host
  //@@ insert code here
  thrust::copy(deviceBins.begin(), deviceBins.end(), hostBins);

  // Check the solution is correct
  // wbSolution(args, hostBins, num_bins);
  int num;
  unsigned *eOutput = (unsigned int *)wbImport(wbArg_getInputFile(args, 1),
                                       &num);
  bool diff = false;
  for (int i = 0; i < num_bins; i++) {
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

  // Free space on the host
  //@@ insert code here
  free(hostBins);

  return 0;
}
