COMPILER = g++
COMPILER_GPU = nvcc
CC = gcc
CFLAGS = -I. -O3 -Wall -Wextra -pedantic -ansi -c -Wno-write-strings
SEQINCLUDE=	-Iseqan

optlist.o:	../optlist/optlist.c ../optlist/optlist.h
	$(CC) $(CFLAGS) $<

test: test-performance-nosimd.cpp

	$(COMPILER) -std=c++14 $(INCLUDE) -O3 -march=native $(SEQINCLUDE) -o test test-performance-nosimd.cpp

gpu: test-performance-gpu.cpp

	$(COMPILER_GPU) -std=c++14 $(INCLUDE) -O3 -march=native $(SEQINCLUDE) -o test test-performance-gpu.cpp

clean:
	rm -f *.o
	rm -f test
	rm -f benchmark.txt
	