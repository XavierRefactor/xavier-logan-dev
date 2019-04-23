COMPILER = g++
COMPILER_GPU = nvcc
CC = gcc
CFLAGS = -I. -O3 -Wall -Wextra -pedantic -ansi -c -Wno-write-strings
SEQINCLUDE=	-I../utils/seqan

test: simulator.cpp

	$(COMPILER) -std=c++14 $(INCLUDE) -O3 -march=native $(SEQINCLUDE) -o test simulator.cpp

clean:
	rm -f *.o
	rm -f test