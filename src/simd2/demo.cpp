//==================================================================
// Title:  LOGAN: X-Drop Adaptive Banded Alignment
// Author: G. Guidi, E. Younis
// Date:   29 April 2019
//==================================================================

#include <vector>
#include <iostream>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <iterator>
#include <x86intrin.h>
// #include "logan_xa.h"

#ifdef DEBUG
	#define log( var ) do { std::cerr << "LOG: " << __FILE__ << "(" << __LINE__ << ") " << #var << " = " << (var) << std::endl; } while(0)
#else
	#define log( var )
#endif

//======================================================================
// DEMO
//======================================================================

int main(int argc, char const *argv[])
{
	srand(0);
	/* Declarations */
	std::string seq1, seq2;

	/* Bandwidth (the alignment path of the input sequence and the
	result does not go out of the band) */
	unsigned short bw = 32;

	/* Penalties (LOGAN temporarily supports only linear gap penalty) */
	short match    =  1;
	short mismatch = -1;
	short gap 	   = -1;

	/* Initialize scoring scheme */
	ScoringSchemeL penalties(match, mismatch, gap);

	/* Generate pair of sequences */
	seq1 = "ACCAATTTGGGACTCCAAAGCTTGGGT";
	seq2 = "ACGAAAAAATTTGGGGGGACTCCCAAAAAGGTTGGTT";

	/* x-drop value */
	unsigned short x = 100;

	/* seed/k-mer length */
	unsigned short k = 17;

	/* seed starting position on seq1, seed starting position on seq2,
	k-mer length */
	SeedL seed(0, 0, k);

	/* result.first = best score, result.second = exit score when (if)
	x-drop termination is satified */
	std::pair<short, short> result_x;

	//==================================================================
	// LOGAN (X-Drop Adaptive Banded Alignment)
	//==================================================================

	result_x = LoganXDrop(seed, LOGAN_EXTEND_BOTH, seq1, seq2, penalties, x);
	std::cout << "Best score : " << result_x.first <<
	"\tExit score : " << result_x.second << std::endl;

	return 0;
}
