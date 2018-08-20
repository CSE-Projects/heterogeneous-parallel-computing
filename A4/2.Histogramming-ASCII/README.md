## Usage

```bash
$ g++ dataset_generator.cpp
$ nvcc template.cu
$ ./a input.txt output.raw
```

### Info
* Input is been read directly from file as wbImport function won't consider spaces as input
* Output is being checked sequentially 
* wbImport function (line 414) in wb.h has been modified to handle unsigned integer input rather than float
* And line 971 had to be changed accordingly
* dataset generator has been modified to remove wb.h dependency