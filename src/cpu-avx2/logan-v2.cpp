//==================================================================
// Title:  C++ x-drop seed-and-extend alignment algorithm
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

#define VECTORWIDTH  (16)
#define LOGICALWIDTH (VECTORWIDTH - 1)
#define RIGHT (0)
#define DOWN  (1)
#define NINF  (std::numeric_limits<short>::min())

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

//int
//extendSeedLGappedXDropAVX2(
//		SeedL & seed,
//		std::string const & querySeg,
//		std::string const & databaseSeg,
//		ExtensionDirectionL const & direction,
//		ScoringSchemeL &scoringScheme,
//		short const &scoreDropOff)
//{
int main(int argc, char const *argv[])
{
	// TODO : check scoring scheme correctness/input parameters

	//unsigned short cols = querySeg.length() + 1;
	//unsigned short rows = databaseSeg.length() + 1;

	//if (rows <= 1 || cols <= 1)
	//	return 0;

	// convert from string to __m256i* array
	// this is the entire sequences 	
	//short* queryh = new short[cols]; 
	//short* queryv = new short[rows];
	//std::copy(querySeg.begin(), querySeg.end(), queryh); 	
	//std::copy(databaseSeg.begin(), databaseSeg.end(), queryv); 

	// test hardcoded
	//short queryh[32] = {'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'};
	//short queryv[32] = {'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'};
	short queryh[32] = {'A', 'C', 'T', 'G', 'A', 'A', 'T', 'C', 'A', 'C', 'T', 'G', 'A', 'A', 'T', 'C', 'A', 'C', 'T', 'G', 'A', 'A', 'T', 'C', 'A', 'C', 'T', 'G', 'A', 'A', 'T', 'C'};
	short queryv[32] = {'G', 'C', 'T', 'A', 'A', 'A', 'G', 'C', 'G', 'C', 'T', 'A', 'A', 'A', 'G', 'C', 'G', 'C', 'T', 'A', 'A', 'A', 'G', 'C', 'G', 'C', 'T', 'A', 'A', 'A', 'G', 'C'};
	int hlength = 32;
	int vlength = 32;

	if (hlength <= 1 || vlength <= 1)
		return 0;

	//setScoreGap(scoringScheme, scoreGap(scoringScheme));
	//setScoreMismatch(scoringScheme, scoreMismatch(scoringScheme));

	short matchCost    =  1; //scoreMatch(scoringScheme);
	short mismatchCost = -1; //scoreMismatch(scoringScheme);
	short gapCost      = -1; //scoreGap(scoringScheme);

	vector_t vmatchCost    = _mm256_set1_epi16 (matchCost   );
	vector_t vmismatchCost = _mm256_set1_epi16 (mismatchCost);
	vector_t vgapCost      = _mm256_set1_epi16 (gapCost     );

	//======================================================================================
	// PHASE I (initial values load using dynamic programming)
	//======================================================================================

	// we need one more space for the off-grid values and one more space for antiDiag2
	short phase1_data[LOGICALWIDTH + 3][LOGICALWIDTH + 3];

	// phase1_data initialization
	phase1_data[0][0] = 0;
	for (int i = 1; i < LOGICALWIDTH + 3; i++)
	{
		phase1_data[0][i] = -i;
		phase1_data[i][0] = -i;
	}

	// dynamic programming loop to fill phase1_data
	for(int i = 1; i < LOGICALWIDTH + 3; i++)
		for(int j = 1; j < LOGICALWIDTH + 3; j++)
		{
			short onef = phase1_data[i-1][j-1];
			if(queryh[i-1] == queryv[j-1])
				onef += matchCost;
			else
				onef += mismatchCost;

			short twof = std::max(phase1_data[i-1][j], phase1_data[i][j-1]) + gapCost;
			phase1_data[i][j] = std::max(onef, twof);
		}

	//for(int i = 1; i < LOGICALWIDTH + 3; i++)
	//{
	//	for(int j = 1; j < LOGICALWIDTH + 3; j++)
	//	{
	//	std::cout << phase1_data[i][j] << '\t';
	//	}
	//std::cout << std::endl;
	//}

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

	print_m256i_16c(vqueryh.simd);
	print_m256i_16c(vqueryv.simd);

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

	//print_m256i_16d(antiDiag1.simd);
	//print_m256i_16d(antiDiag2.simd);

	//======================================================================================
	// PHASE II (core vectorized computation)
	//======================================================================================

	// compute
	short prevDir = RIGHT;
	//short count = 0;

	// phase III will begin when both hoffset < hlength and voffset < vlength are verified 
	// in the adaptive version so this loop will be different
	while(hoffset < hlength && voffset < vlength)
	{
		// ONEF
		//count++;
		// -1 for a match and 0 for a mismatch
		vector_t m = _mm256_cmpeq_epi16 (vqueryh.simd, vqueryv.simd);
		//print_m256i_16d(m);
		m = _mm256_blendv_epi8 (vmismatchCost, vmatchCost, m);
		//print_m256i_16d(m);
		vector_t vonef = _mm256_adds_epi16 (m, antiDiag1.simd);
		//print_m256i_16d(antiDiag1.simd);
		//printf("vonef ");
		//print_m256i_16d(vonef);

		// TWOF
		//print_m256i_16d(antiDiag2.simd);
		vector_union_t vtwos = leftShift (antiDiag2);
		//printf("vtwos ");
		//print_m256i_16d(vtwos.simd);
		vector_t vtwom = _mm256_max_epi16 (vtwos.simd, antiDiag2.simd);
		//printf("vtwom ");
		//print_m256i_16d(vtwom);
		vector_t vtwof = _mm256_adds_epi16 (vtwom, vgapCost);
		//printf("vtwof ");
		//print_m256i_16d(vtwof);

		// THREE
		antiDiag3.simd = _mm256_max_epi16 (vonef, vtwof);
		//printf("1 ");
		//print_m256i_16d(antiDiag1.simd);
		//printf("2 ");
		//print_m256i_16d(antiDiag2.simd);
		//printf("3 ");
		//// -15 should be off (it will be shifted off next iteration)
		//print_m256i_16d(antiDiag3.simd);
		//printf("\n");

		// x-drop termination

		short nextDir = prevDir ^ 1;
		move (prevDir, nextDir, antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);

		prevDir = nextDir;
	}

	//printf("count %d\n", count);

	// update best
	//antiDiagBest = *std::max_element(antiDiag3.elem + offset3, antiDiag3.elem + antiDiag3size);
	//best = (best > antiDiagBest) ? best : antiDiagBest;

	//======================================================================================
	// PHASE III (reaching end of sequences)
	//======================================================================================

	for (int i = 0; i < (LOGICALWIDTH - 1); i++)
	{
		// ONEF
		//count++;
		// -1 for a match and 0 for a mismatch
		vector_t m = _mm256_cmpeq_epi16 (vqueryh.simd, vqueryv.simd);
		//print_m256i_16d(m);
		m = _mm256_blendv_epi8 (vmismatchCost, vmatchCost, m);
		//print_m256i_16d(m);
		vector_t vonef = _mm256_adds_epi16 (m, antiDiag1.simd);
		//print_m256i_16d(antiDiag1.simd);
		//printf("vonef ");
		//print_m256i_16d(vonef);

		// TWOF
		//print_m256i_16d(antiDiag2.simd);
		vector_union_t vtwos = leftShift (antiDiag2);
		//printf("vtwos ");
		//print_m256i_16d(vtwos.simd);
		vector_t vtwom = _mm256_max_epi16 (vtwos.simd, antiDiag2.simd);
		//printf("vtwom ");
		//print_m256i_16d(vtwom);
		vector_t vtwof = _mm256_adds_epi16 (vtwom, vgapCost);
		//printf("vtwof ");
		//print_m256i_16d(vtwof);

		// THREE
		antiDiag3.simd = _mm256_max_epi16 (vonef, vtwof);
		//printf("1 ");
		//print_m256i_16d(antiDiag1.simd);
		//printf("2 ");
		//print_m256i_16d(antiDiag2.simd);
		//printf("3 ");
		//// -15 should be off (it will be shifted off next iteration)
		//print_m256i_16d(antiDiag3.simd);
		//printf("\n");

		// x-drop termination

		short nextDir = prevDir ^ 1;
		move (prevDir, nextDir, antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);

		prevDir = nextDir;
	}

		printf("1 ");
		print_m256i_16d(antiDiag1.simd);
		printf("2 ");
		print_m256i_16d(antiDiag2.simd);
		printf("3 ");
		// -15 should be off (it will be shifted off next iteration)
		print_m256i_16d(antiDiag3.simd);
		printf("\n");

	// find positions of longest extension
	// update seed
	return 0;
}