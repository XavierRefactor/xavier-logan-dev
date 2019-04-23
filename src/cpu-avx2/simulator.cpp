//==================================================================
// Title:  LOGAN: X-Drop Adaptive Banded Alignment
// Author: G. Guidi, E. Younis
// Date:   23 April 2019
//==================================================================

#include<vector>
#include<iostream>
#include<string>
#include<omp.h>
#include<algorithm>
#include<stdlib.h>
#include<inttypes.h>
#include<assert.h>
#include<iterator>
#include"logan-v2.cpp"
#include <immintrin.h>

//======================================================================================
// READ SIMULATOR
//======================================================================================

#define LEN (10000)	// read length (this is going to be a distribution of length in
						// the adaptive version)
#define MAT	( 1)
#define MIS	(-1)
#define GAP	(-1)

void 
readSimulator (std::string& readh, std::string& readv)
{
	char bases[4] = {'A', 'T', 'C', 'G'}; 

	for (int i = 0; i < LEN; i++)
	{
		// two identical sequences
		readh = readh + bases[rand() % 4]; 
		readv = readv + bases[rand() % 4];
	}
}

//======================================================================================
// BENCHMARK CODE
//======================================================================================

int main(int argc, char const *argv[])
{
	std::string targetSeg, querySeg;
	ScoringSchemeL scoringScheme(MAT, MIS, GAP);

	// 1st prototype without seed and x-drop termination (not adaptive band so sequences 
	// have same length)
	// sim a pair of read
	readSimulator(targetSeg, querySeg);
	// calling Logan
	LoganAVX2(targetSeg, querySeg, scoringScheme);

	return 0;
}
