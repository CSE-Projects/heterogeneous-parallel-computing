#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(int argc, char *argv[]) {

  float *hostInput1 = nullptr;
  float *hostInput2 = nullptr;
  float *hostOutput = nullptr;
  int inputLength;

  /* parse the input arguments */
  //@@ Insert code here

  // Import host input data
  //@@ Read data from the raw files here
  //@@ Insert code here
  hostInput1 =
  hostInput2 =

  // Declare and allocate host output
  //@@ Insert code here

  // Declare and allocate thrust device input and output vectors
  //@@ Insert code here

  // Copy to device
  //@@ Insert code here

  // Execute vector addition
  //@@ Insert Code here

  /////////////////////////////////////////////////////////

  // Copy data back to host
  //@@ Insert code here

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  return 0;
}
