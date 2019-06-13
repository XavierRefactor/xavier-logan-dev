COMPILER = g++
COMPILER_GPU = nvcc
CC = gcc
CFLAGS = -I. -O3 -Wall -Wextra -pedantic -ansi -c -Wno-write-strings
SEQINCLUDE = -Iseqan
CUB = -Icub

optlist.o:	../optlist/optlist.c ../optlist/optlist.h
	$(CC) $(CFLAGS) $<

test: test-performance-nosimd.cpp

	$(COMPILER) -std=c++14 $(INCLUDE) -O3 -march=native $(SEQINCLUDE) -o test test-performance-nosimd.cpp

gpu_c: test-performance-gpu-temp_cpp.cpp
	
	$(COMPILER) -std=c++14 $(INCLUDE) -O3 -march=native $(SEQINCLUDE) -o test test-performance-gpu-temp_cpp.cpp 

gpu: test-performance-gpu.cu

	$(COMPILER_GPU) -std=c++14 -maxrregcount=32 $(INCLUDE) $(SEQINCLUDE) test-performance-gpu.cu -o test

gpu_dbg: test-performance-gpu.cu

	$(COMPILER_GPU) -G -std=c++14 $(INCLUDE) $(SEQINCLUDE) test-performance-gpu.cu -o test

gpu_cori: test-performance-gpu.cu

	$(COMPILER_GPU) -arch=sm_70 -O3 -maxrregcount=32 -std=c++14 $(INCLUDE) $(SEQINCLUDE) test-performance-gpu.cu -o test

gpu_cori_dbg: test-performance-gpu.cu

	$(COMPILER_GPU) -G -arch=sm_70 -std=c++14 $(INCLUDE) $(SEQINCLUDE) test-performance-gpu.cu -o test

clean:
	rm -f *.o
	rm -f test
	rm -f benchmark.txt
	
