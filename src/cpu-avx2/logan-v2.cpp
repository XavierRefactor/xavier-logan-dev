//==================================================================
// Title:  LOGAN: X-Drop Adaptive Banded Alignment
// Author: G. Guidi, E. Younis
// Date:   22 April 2019
//==================================================================

#include<vector>
#include<iostream>
#include<omp.h>
#include<algorithm>
#include<inttypes.h>
#include<assert.h>
#include<iterator>
#include"logan.h"
#include"score.h"
#include <immintrin.h>

#define DEBUG

//======================================================================================
// GLOBAL DEFINITION
//======================================================================================

#define VECTORWIDTH  (16)
#define LOGICALWIDTH (VECTORWIDTH - 1)
#define RIGHT (0)
#define DOWN  (1)
#define NINF  (std::numeric_limits<short>::min())

//======================================================================================
// UTILS
//======================================================================================

typedef __m256i vector_t;
typedef int16_t element_t;

typedef union {
	vector_t  simd;
	element_t elem[VECTORWIDTH];
} vector_union_t;

void
print_vector_c(vector_t a) {

	vector_union_t tmp;
	tmp.simd = a;

	printf("{");
	for (int i = 0; i < VECTORWIDTH-1; ++i)
		printf("%c,", tmp.elem[i]);
	printf("%c}\n", tmp.elem[VECTORWIDTH]);
}

void
print_vector_d(vector_t a) {

	vector_union_t tmp;
	tmp.simd = a;

	printf("{");
	for (int i = 0; i < VECTORWIDTH-1; ++i)
		printf("%d,", tmp.elem[i]);
	printf("%d}\n", tmp.elem[VECTORWIDTH]);
}

inline vector_union_t
leftShift (const vector_union_t& a) {

	vector_union_t b; 

	for(short i = 0; i < (VECTORWIDTH - 1); i++)	// data are saved in reversed order
		b.elem[i] = a.elem[i + 1];

	// replicating last element
	b.elem[VECTORWIDTH - 1] = NINF;
	return b;
}

inline vector_union_t
rightShift (const vector_union_t& a) {

	vector_union_t b; 

	for(short i = 0; i < (VECTORWIDTH - 1); i++)	// data are saved in reversed order
		b.elem[i + 1] = a.elem[i];

	// replicating last element
	b.elem[0] = NINF;
	return b;
}

void
rightAfterDown (vector_union_t& antiDiag1, vector_union_t& antiDiag2, 
	vector_union_t& antiDiag3, int& hoffset, int& voffset, vector_union_t& vqueryh, 
		vector_union_t& vqueryv, const short queryh[], const short queryv[])
{
	// (a) shift to the left on query horizontal
	leftShift (vqueryh);
	vqueryh.elem[LOGICALWIDTH-1] = queryh[hoffset++];
	// (b) shift left on updated vector 1 (this places the right-aligned vector 2 as a left-aligned vector 1)
	antiDiag1.simd = antiDiag2.simd;
	leftShift (antiDiag1);
	antiDiag2.simd = antiDiag3.simd;
}

void
downAfterRight (vector_union_t& antiDiag1, vector_union_t& antiDiag2, 
	vector_union_t& antiDiag3, int& hoffset, int& voffset, vector_union_t& vqueryh, 
		vector_union_t& vqueryv, const short queryh[], const short queryv[])
{
	//(a) shift to the right on query vertical
	rightShift (vqueryv);
	vqueryh.elem[0] = queryv[voffset++];
	//(b) shift to the right on updated vector 2 (this places the left-aligned vector 3 as a right-aligned vector 2)
	antiDiag1.simd = antiDiag2.simd;
	antiDiag2.simd = antiDiag3.simd;
	rightShift (antiDiag2);
}

void
rightAfterRight (vector_union_t& antiDiag1, vector_union_t& antiDiag2, 
	vector_union_t& antiDiag3, int& hoffset, int& voffset, vector_union_t& vqueryh, 
		vector_union_t& vqueryv, const short queryh[], const short queryv[])
{
	// TODO
}

void
downAfterDown (vector_union_t& antiDiag1, vector_union_t& antiDiag2, 
	vector_union_t& antiDiag3, int& hoffset, int& voffset, vector_union_t& vqueryh, 
		vector_union_t& vqueryv, const short queryh[], const short queryv[])
{
	// TODO
}

void
move (const short& prevDir, const short& nextDir, vector_union_t& antiDiag1, vector_union_t& antiDiag2, 
	vector_union_t& antiDiag3, int& hoffset, int& voffset, vector_union_t& vqueryh, vector_union_t& vqueryv,
		const short queryh[], const short queryv[])
{
	if(prevDir == RIGHT && nextDir == DOWN)
	{
		rightAfterDown (antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);
	}
	else if(prevDir == DOWN && nextDir == RIGHT)
	{
		downAfterRight (antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);
	}
	else if(prevDir == RIGHT && nextDir == RIGHT)
	{
		rightAfterRight (antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);
	}
	else if(prevDir == DOWN && nextDir == DOWN)
	{
		downAfterDown (antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);
	}
}

//======================================================================================
// X-DROP (not yet) ADAPTIVE (not yet) BANDED ALIGNMENT
//======================================================================================

//int
//LoganAVX2(
//		SeedL & seed,
//		std::string const & querySeg,
//		std::string const & databaseSeg,
//		ExtensionDirectionL const & direction,
//		ScoringSchemeL &scoringScheme,
//		short const &scoreDropOff)
//{

// 1st prototype
void
LoganAVX2(
		std::string const& targetSeg,
		std::string const& querySeg,
		ScoringSchemeL& scoringScheme)
{
//int main(int argc, char const *argv[])
//{
	// TODO: check scoring scheme correctness/input parameters
	// TODO: chop sequences in left and right extension

	unsigned short hlength = targetSeg.length() + 1;
	unsigned short vlength = querySeg.length()  + 1;

	if (hlength <= 1 || vlength <= 1)
		return;

	// Convert from string to int array
	// This is the entire sequences 	
	short* queryh = new short[hlength]; 
	short* queryv = new short[vlength];
	std::copy(targetSeg.begin(), targetSeg.end(), queryh); 	
	std::copy(querySeg.begin(), querySeg.end(), queryv); 

	//Redundant piece of code
	//setScoreGap(scoringScheme, scoreGap(scoringScheme));
	//setScoreMismatch(scoringScheme, scoreMismatch(scoringScheme));

	short matchCost    = scoreMatch(scoringScheme   );
	short mismatchCost = scoreMismatch(scoringScheme);
	short gapCost      = scoreGap(scoringScheme     );

	vector_t vmatchCost    = _mm256_set1_epi16 (matchCost   );
	vector_t vmismatchCost = _mm256_set1_epi16 (mismatchCost);
	vector_t vgapCost      = _mm256_set1_epi16 (gapCost     );

	//======================================================================================
	// PHASE I (initial values load using dynamic programming)
	//======================================================================================

	// we need one more space for the off-grid values and one more space for antiDiag2
	short phase1_data[LOGICALWIDTH + 2][LOGICALWIDTH + 2];

	// phase1_data initialization
	phase1_data[0][0] = 0;
	for (int i = 1; i < LOGICALWIDTH + 2; i++)
	{
		phase1_data[0][i] = -i;
		phase1_data[i][0] = -i;
	}

	// dynamic programming loop to fill phase1_data[][]
	for(int i = 1; i < LOGICALWIDTH + 2; i++)
		for(int j = 1; j < LOGICALWIDTH + 2; j++)
		{
			short onef = phase1_data[i-1][j-1];
			if(queryh[i-1] == queryv[j-1])
				onef += matchCost;
			else
				onef += mismatchCost;

			short twof = std::max(phase1_data[i-1][j], phase1_data[i][j-1]) + gapCost;
			phase1_data[i][j] = std::max(onef, twof);
		}

#ifdef DEBUG
	// print phase1_data[][]
	for(int i = 1; i < LOGICALWIDTH + 2; i++)
	{
		for(int j = 1; j < LOGICALWIDTH + 2; j++)
			std::cout << phase1_data[i][j] << '\t';
		std::cout << std::endl;
	}
#endif

	vector_union_t antiDiag1; 	// 16 (vector width) 16-bit integers
	vector_union_t antiDiag2; 	// 16 (vector width) 16-bit integers
	vector_union_t antiDiag3; 	// 16 (vector width) 16-bit integers

	vector_union_t vqueryh;
	vector_union_t vqueryv;

	// Initialize vqueryh and vqueryv
	for ( int i = 0; i < LOGICALWIDTH; ++i )
	{
		vqueryh.elem[i] = queryh[i + 1];
		vqueryv.elem[i] = queryv[LOGICALWIDTH - i];
	}

	vqueryh.elem[LOGICALWIDTH] = NINF;
	vqueryv.elem[LOGICALWIDTH] = NINF;

#ifdef DEBUG
	print_vector_c(vqueryh.simd);
	print_vector_c(vqueryv.simd);
#endif

	// this should point to the next value to be loaded into vqueryh and vqueryv
	int hoffset = LOGICALWIDTH;
	int voffset = LOGICALWIDTH;

	// load phase1_data into antiDiag1 vector
	for (int i = 1; i <= LOGICALWIDTH; ++i)
		antiDiag1.elem[i-1] = phase1_data[i][LOGICALWIDTH - i + 1];
	antiDiag1.elem[LOGICALWIDTH] = NINF;

	// load phase1_data into antiDiag2 vector going RIGHT (our arbitrary decision)
	// the first antiDiag3 computation is going DOWN
	// shift to the right on updated vector 2 (This places the left-aligned vector 3 as a right-aligned vector 2)
	for (int i = 1; i <= LOGICALWIDTH; ++i)
		antiDiag2.elem[i] = phase1_data[i + 1][LOGICALWIDTH - i + 1];
	antiDiag2.elem[0] = NINF;

	// initialize antiDia3 to -inf
	antiDiag3.simd = _mm256_set1_epi16(NINF);

	//======================================================================================
	// PHASE II (core vectorized computation)
	//======================================================================================

	short prevDir = RIGHT;
	short antiDiagNo = 1;

	// TODO: Phase III in adaptive version begins when both hoffset < hlength and voffset < vlength are verified 
	// This loop should looks different in the adaptive version
	while(hoffset < hlength && voffset < vlength)
	{
		// antiDiagBest initialization
		antiDiagNo++;
		short antiDiagBest = antiDiagNo * gapCost;

		// antiDiag1F (final)
		// POST-IT: -1 for a match and 0 for a mismatch
		vector_t m = _mm256_cmpeq_epi16 (vqueryh.simd, vqueryv.simd);
		m = _mm256_blendv_epi8 (vmismatchCost, vmatchCost, m);
		vector_t antiDiag1F = _mm256_adds_epi16 (m, antiDiag1.simd);

	//#ifdef DEBUG
	//	printf("antiDiag1: ");
	//	print_vector_d(antiDiag1.simd);
	//	printf("antiDiag1F: ");
	//	print_vector_d(antiDiag1F);
	//#endif

		// antiDiag2S (shift)
		vector_union_t antiDiag2S = leftShift (antiDiag2);
	//#ifdef DEBUG
	//	printf("antiDiag2S: ");
	//	print_vector_d(antiDiag2S.simd);
	//#endif
		// antiDiag2M (pairwise max)
		vector_t antiDiag2M = _mm256_max_epi16 (antiDiag2S.simd, antiDiag2.simd);
	//#ifdef DEBUG
	//	printf("antiDiag2M: ");
	//	print_vector_d(antiDiag2M);
	//#endif
		// antiDiag2F (final)
		vector_t antiDiag2F = _mm256_adds_epi16 (antiDiag2M, vgapCost);
	//#ifdef DEBUG
	//	printf("antiDiag2F: ");
	//	print_vector_d(antiDiag2F);
	//#endif

		// Compute antiDiag3
		antiDiag3.simd = _mm256_max_epi16 (antiDiag1F, antiDiag2F);
	//#ifdef DEBUG
	//	printf("antiDiag3: ");
	//	print_vector_d(antiDiag3.simd);
	//	printf("\n");
	//#endif

		// TODO: x-drop termination

		// antiDiag swap, offset updates, and new base load
		short nextDir = prevDir ^ 1;
		move (prevDir, nextDir, antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);

		// direction update
		prevDir = nextDir;
	}

	// TODO: update best
	//antiDiagBest = *std::max_element(antiDiag3.elem + offset3, antiDiag3.elem + antiDiag3size);
	//best = (best > antiDiagBest) ? best : antiDiagBest;

	//======================================================================================
	// PHASE III (reaching end of sequences)
	//======================================================================================

	for (int i = 0; i < (LOGICALWIDTH - 1); i++)
	{
		// antiDiag1F (final)
		// POST-IT: -1 for a match and 0 for a mismatch
		vector_t m = _mm256_cmpeq_epi16 (vqueryh.simd, vqueryv.simd);
		m = _mm256_blendv_epi8 (vmismatchCost, vmatchCost, m);
		vector_t antiDiag1F = _mm256_adds_epi16 (m, antiDiag1.simd);

	//#ifdef DEBUG
	//	printf("antiDiag1: ");
	//	print_vector_d(antiDiag1.simd);
	//	printf("antiDiag1F: ");
	//	print_vector_d(antiDiag1F);
	//#endif

		// antiDiag2S (shift)
		vector_union_t antiDiag2S = leftShift (antiDiag2);
	//#ifdef DEBUG
	//	printf("antiDiag2S: ");
	//	print_vector_d(antiDiag2S.simd);
	//#endif
		// antiDiag2M (pairwise max)
		vector_t antiDiag2M = _mm256_max_epi16 (antiDiag2S.simd, antiDiag2.simd);
	//#ifdef DEBUG
	//	printf("antiDiag2M: ");
	//	print_vector_d(antiDiag2M);
	//#endif
		// antiDiag2F (final)
		vector_t antiDiag2F = _mm256_adds_epi16 (antiDiag2M, vgapCost);
	//#ifdef DEBUG
	//	printf("antiDiag2F: ");
	//	print_vector_d(antiDiag2F);
	//#endif

		// Compute antiDiag3
		antiDiag3.simd = _mm256_max_epi16 (antiDiag1F, antiDiag2F);

		// TODO: x-drop termination

		// antiDiag swap, offset updates, and new base load
		short nextDir = prevDir ^ 1;
		move (prevDir, nextDir, antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);

		// direction update
		prevDir = nextDir;
	}

		// TODO: update best
		//antiDiagBest = *std::max_element(antiDiag3.elem + offset3, antiDiag3.elem + antiDiag3size);
		//best = (best > antiDiagBest) ? best : antiDiagBest;

	#ifdef DEBUG
		printf("antiDiag1: ");
		print_vector_d(antiDiag1.simd);
		printf("antiDiag2: ");
		print_vector_d(antiDiag2.simd);
		// -15 should be off (it will be shifted off next iteration)
		printf("antiDiag3: ");
		print_vector_d(antiDiag3.simd);
		printf("\n");
	#endif

	// TODO: find positions of longest extension
	// TODO: update seed
	// return 0;
}