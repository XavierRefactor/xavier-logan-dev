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

	$(COMPILER_GPU) -Xcompiler "-std=c++11" $(INCLUDE) $(SEQINCLUDE) test-performance-gpu.cu -o test

gpu_temp: test-performance-gpu-temp.cu

	$(COMPILER_GPU) -std=c++11 $(INCLUDE) $(SEQINCLUDE) $(CUB) test-performance-gpu-temp.cu -o test

gpu_temp_dfstream:  test-performance-gpu-temp.cu

	$(COMPILER_GPU) --default-stream per-thread -std=c++11 $(INCLUDE) $(SEQINCLUDE) $(CUB) ./test-performance-gpu-temp.cu -o test
 

clean:
	rm -f *.o
	rm -f test
	rm -f benchmark.txt
	
