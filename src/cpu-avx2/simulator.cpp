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
#include<x86intrin.h>
#include<seqan/sequence.h>
#include<seqan/align.h>
#include<seqan/seeds.h>
#include<seqan/score.h>
#include<seqan/modifier.h>
#include"logan.cpp"

//======================================================================================
// READ SIMULATOR
//======================================================================================

#define LEN1 (10000)		// read length (this is going to be a distribution of length in
						// the adaptive version)
#define LEN2 (10000)		// 2nd read length
#define MAT	( 1)		// match score
#define MIS	(-1)		// mismatch score
#define GAP	(-1)		// gap score
#define XDROP (LEN1)	// so high so it won't be triggered in SeqAn

void 
readSimulator (std::string& readh, std::string& readv)
{
	char bases[4] = {'A', 'T', 'C', 'G'}; 

	// reads are currently identical
	// read horizontal
	for (int i = 0; i < LEN1; i++)
	{
		readh = readh + bases[rand() % 4];
		readv = readv + bases[rand() % 4];
	}
	//for (int i = LEN1; i < LEN2; i++)
	//{
	//	int test = rand();
	//	readv = readv + bases[test % 4];
	//}

	// read vertical
	//for (int i = 0; i < LEN2; i++)
	//	readv = readv + bases[test % 4];
}

//======================================================================================
// BENCHMARK CODE
//======================================================================================

int main(int argc, char const *argv[])
{
	std::string targetSeg, querySeg;

	// Simulate pair of read
	readSimulator(targetSeg, querySeg);
	std::cout << targetSeg << "\n" << std::endl;
	std::cout << querySeg << "\n" << std::endl;

	// Logan
	ScoringSchemeL scoringSchemeLogan(MAT, MIS, GAP);
	// 1st prototype without seed and x-drop termination (not adaptive band so sequences 
	// have same length)
	std::chrono::duration<double> diff1;
	auto start1 = std::chrono::high_resolution_clock::now();
	LoganAVX2(targetSeg, querySeg, scoringSchemeLogan);
	auto end1 = std::chrono::high_resolution_clock::now();
	diff1 = end1-start1;
	// score off by factor of 5
	std::cout << " in " << diff1.count() << " sec " << std::endl;

	// SeqAn
	seqan::Score<int, seqan::Simple> scoringSchemeSeqAn(MAT, MIS, GAP);
	seqan::Seed<seqan::Simple> seed(0, 0, 0);
	std::chrono::duration<double> diff2;
	auto start2 = std::chrono::high_resolution_clock::now();
	int score = seqan::extendSeed(seed, targetSeg, querySeg, seqan::EXTEND_RIGHT, 
		scoringSchemeSeqAn, XDROP, seqan::GappedXDrop(), 0);
	auto end2 = std::chrono::high_resolution_clock::now();
	diff2 = end2-start2;

	// SeqAn is doing more computation
	std::cout << "SeqAn's best " << score << " in " << diff2.count() << " sec " << std::endl;

	return 0;
}
