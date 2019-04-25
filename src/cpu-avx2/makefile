COMPILER = g++
COMPILER_GPU = nvcc
CC = gcc
CFLAGS = 		-I. -O3 -Wall -Wextra -pedantic -ansi -c -Wno-write-strings
INCLUDE=		-I../utils/seqan -Ilibgaba
LIBS= 			-Llibgaba -lgaba

#gabalib:
#	$(MAKE) -C libgaba all

test: simulator.cpp #gabalib
	$(COMPILER) -std=c++14 $(INCLUDE) -O3 -march=native -fpermissive -o test simulator.cpp #${LIBS}

clean:
	rm -f *.o
	rm -f test
	$(MAKE) -C libgaba clean