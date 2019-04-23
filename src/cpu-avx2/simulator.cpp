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
#include <immintrin.h>
#include <seqan/sequence.h>
#include <seqan/align.h>
#include <seqan/seeds.h>
#include <seqan/score.h>
#include <seqan/modifier.h>
#include"logan-v2.cpp"

//======================================================================================
// READ SIMULATOR
//======================================================================================

#define LEN (200)		// read length (this is going to be a distribution of length in
						// the adaptive version)
#define MAT	( 1)		// match score
#define MIS	(-1)		// mismatch score
#define GAP	(-1)		// gap score
#define XDROP (50)	// so high so it won't be triggered in SeqAn

void 
readSimulator (std::string& readh, std::string& readv)
{
	char bases[4] = {'A', 'T', 'C', 'G'}; 

	for (int i = 0; i < LEN; i++)
	{
		// two identical sequences
		int test = rand();
		readh = readh + bases[rand() % 4]; 
		readv = readv + bases[test % 4];
	}
}

//======================================================================================
// BENCHMARK CODE
//======================================================================================

int main(int argc, char const *argv[])
{
	std::string targetSeg, querySeg;

	// Simulate pair of read
	readSimulator(targetSeg, querySeg);
	std::cout << std::endl;
	std::cout << targetSeg << std::endl;
	std::cout << std::endl;
	std::cout << querySeg  << std::endl;
	std::cout << std::endl;

	// Logan
	ScoringSchemeL scoringSchemeLogan(MAT, MIS, GAP);
	// 1st prototype without seed and x-drop termination (not adaptive band so sequences 
	// have same length)
	LoganAVX2(targetSeg, querySeg, scoringSchemeLogan);

	// SeqAn
	seqan::Score<int, seqan::Simple> scoringSchemeSeqAn(MAT, MIS, GAP);
	seqan::Seed<seqan::Simple> seed(0, 0, 0);
	int score = seqan::extendSeed(seed, targetSeg, querySeg, seqan::EXTEND_RIGHT, 
		scoringSchemeSeqAn, XDROP, seqan::GappedXDrop(), 0);

	std::cout << "" << score << std::endl;

	return 0;
}
