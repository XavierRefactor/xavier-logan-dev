CC = gcc
CXX = g++
CFLAGS  = 	-I. -O3 -Wall -Wextra -pedantic -ansi -c -Wno-write-strings
#SEQANPATH=seqan2.4/seqan/include
SEQANPATH=../utils/seqan
SSWPATH=Complete-Striped-Smith-Waterman-Library/src
KSW2PATH=ksw2
INCLUDE=	-I$(SEQANPATH) -Ilibgaba
LIBS= #-Llibgaba -lgaba

test: $(SSWPATH)/ssw_cpp.cpp $(SSWPATH)/ssw.c simulator.cpp ksw2 
	$(CXX) -march=native -std=c++14 $(INCLUDE) -O3 -fpermissive -o test $(SSWPATH)/ssw_cpp.cpp $(SSWPATH)/ssw.c simulator.cpp $(LIBS)

mc: $(SSWPATH)/ssw_cpp.cpp $(SSWPATH)/ssw.c simulator-mc.cpp ksw2 
	$(CXX) -march=native -std=c++14 $(INCLUDE) -O3 -fopenmp -fpermissive -o mc $(SSWPATH)/ssw_cpp.cpp $(SSWPATH)/ssw.c simulator-mc.cpp $(LIBS)

demo: demo.cpp
	$(CXX) -march=native -std=c++14 -O3 -fpermissive -o demo demo.cpp

clean:
	rm -f *.o
	rm -f test
	rm -f mc
	rm -f demo

