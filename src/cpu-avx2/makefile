COMPILER = g++
COMPILER_GPU = nvcc
CC = gcc
CFLAGS = 		-I. -O3 -Wall -Wextra -pedantic -ansi -c -Wno-write-strings
INCLUDE=		-I../utils/seqan -Ilibgaba -Iksw2
#LIBS= 			-Lksw2 #-Llibgaba -lgaba

test: simulator.cpp ksw2 #gabalib
	$(COMPILER) -mavx2 -std=c++14 $(INCLUDE) -O3 -fpermissive -o test simulator.cpp

clean:
	rm -f *.o
	rm -f test
