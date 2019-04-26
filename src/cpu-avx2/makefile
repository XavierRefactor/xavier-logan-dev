CC = gcc
CXX = g++
CFLAGS  = 	-I. -O3 -Wall -Wextra -pedantic -ansi -c -Wno-write-strings
SEQANPATH=../utils/seqan
SSWPATH=Complete-Striped-Smith-Waterman-Library/src
KSW2PATH=ksw2
INCLUDE =	-I../utils/seqan -Iparasail -I edlib/include#-Ilibgaba
LIBS= 	-ledlib#-Llibgaba -lgaba

test: $(SSWPATH)/ssw_cpp.cpp $(SSWPATH)/ssw.c simulator.cpp ksw2 
	$(CXX) -march=native -std=c++14 $(INCLUDE) -O3 -fpermissive -o test $(SSWPATH)/ssw_cpp.cpp $(SSWPATH)/ssw.c simulator.cpp $(LIBS)

clean:
	rm -f *.o
	rm -f test
