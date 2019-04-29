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
#include<x86intrin.h>
#include"logan.h"
#include"score.h"

//======================================================================================
// GLOBAL FUNCTION DECLARATION
//======================================================================================

#ifdef __AVX2__ 	// Compile flag: -mavx2
#define VECTORWIDTH  (16)
#define LOGICALWIDTH (VECTORWIDTH - 1)
#define vector_t    __m256i
#define add_func    _mm256_adds_epi16  // saturated arithmetic
#define max_func    _mm256_max_epi16   // max
#define set1_func   _mm256_set1_epi16  // set1 operation
#define blendv_func _mm256_blendv_epi8 // blending operation
#define cmpeq_func  _mm256_cmpeq_epi16 // compare equality operation
//#define slli_func 	// AVX2 does not have proper slli (sad) TODO: code here our function
//#define srli_func 	// AVX2 does not have proper srli (sad) TODO: code here our function
#elif __SSE4_2__ 	// Compile flag: -msse4.2
#define VECTORWIDTH  (8)
#define LOGICALWIDTH (VECTORWIDTH - 1)
#define vector_t    __m128i
#define add_func    _mm_adds_epi16 // saturated arithmetic
#define max_func    _mm_max_epi16  // max
#define set1_func   _mm_set1_epi16  // set1 operation
#define blendv_func _mm_blendv_epi8 // blending operation
#define cmpeq_func  _mm_cmpeq_epi16 // compare equality operation
#define slli_func _mm_slli_epi16 // left shift
#define srli_func _mm_srli_epi16 // right shift
//#elif __AVX512F__ 	// Compile flag: -march=skylake-avx512
//#define VECTORWIDTH  (32)
//#define LOGICALWIDTH (VECTORWIDTH - 1)
//#define vector_t    __m512i
//#define add_func    _mm512_adds_epi16 // saturated arithmetic
//#define max_func    _mm512_max_epi16  // max
//#define set1_func   _mm512_set1_epi16  // set1 operation
//#define blendv_func _mm256_blendv_epi8 // blending operation
//#define cmpeq_func  _mm256_cmpeq_epi16 // compare equality operation
//#define slli_func _mm512_slli_epi16 // left shift
//#define srli_func _mm512_srli_epi16 // right shift
#endif

//From: https://github.com/ocxtal/adaptivebandbench
//#define VEC_SHIFT_R(a) { \
//	__m256i tmp1 = _mm256_permute2x128_si256((a##1), (a##2), 0x21); \
//	__m256i tmp2 = _mm256_permute2x128_si256((a##1), (a##2), 0x83); \
//	(a##1) = _mm256_alignr_epi8(tmp1, (a##1), sizeof(short)); \
//	(a##2) = _mm256_alignr_epi8(tmp2, (a##2), sizeof(short)); \
//}
//
//#define VEC_SHIFT_L(a) { \
//	__m256i tmp1 = _mm256_permute2x128_si256((a##2), (a##1), 0x28); \
//	__m256i tmp2 = _mm256_permute2x128_si256((a##2), (a##1), 0x03); \
//	(a##1) = _mm256_alignr_epi8((a##1), tmp1, sizeof(__m128i) - sizeof(short)); \
//	(a##2) = _mm256_alignr_epi8((a##2), tmp2, sizeof(__m128i) - sizeof(short)); \
//}

//======================================================================================
// GLOBAL VARIABLE DEFINITION
//======================================================================================

//#define DEBUG
#define NINF  	(std::numeric_limits<short>::min())
#define myRIGHT 	(0)
#define myDOWN  	(1)
#define MIDDLE 	(LOGICALWIDTH / 2)

//======================================================================================
// UTILS
//======================================================================================

typedef short element_t;

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
	printf("%c}\n", tmp.elem[VECTORWIDTH-1]);
}

void
print_vector_d(vector_t a) {

	vector_union_t tmp;
	tmp.simd = a;

	printf("{");
	for (int i = 0; i < VECTORWIDTH-1; ++i)
		printf("%d,", tmp.elem[i]);
	printf("%d}\n", tmp.elem[VECTORWIDTH-1]);
}

// TODO: optimize with intrinsics
inline vector_union_t
leftShift (const vector_union_t& a) { // this work for avx2

	vector_union_t b;
	// https://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avxhttps://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avx
	b.simd = _mm256_alignr_epi8(_mm256_permute2x128_si256(a.simd, a.simd, _MM_SHUFFLE(2, 0, 0, 1)), a.simd, 2);

	//for(short i = 0; i < (VECTORWIDTH - 1); i++)	// data are saved in reversed order
	//	b.elem[i] = a.elem[i + 1];
	// replicating last element
	b.elem[VECTORWIDTH - 1] = NINF;
	//print_vector_d(a.simd);
	//print_vector_d(b.simd);
	//printf("\n");
	return b;
}

// TODO: optimize with intrinsics
inline vector_union_t
rightShift (const vector_union_t& a) { // this work for avx2

	vector_union_t b;
	// https://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avx
	b.simd = _mm256_alignr_epi8(a.simd, _mm256_permute2x128_si256(a.simd, a.simd, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 2);

	//for(short i = 0; i < (VECTORWIDTH - 1); i++)	// data are saved in reversed order
	//	b.elem[i + 1] = a.elem[i];
	// replicating last element
	b.elem[0] = NINF;
	//print_vector_d(a.simd);
	//print_vector_d(b.simd);
	//printf("\n");
	return b;
}

static inline void
moveRight (vector_union_t& antiDiag1, vector_union_t& antiDiag2,
	vector_union_t& antiDiag3, int& hoffset, int& voffset, vector_union_t& vqueryh,
		vector_union_t& vqueryv, const short queryh[], const short queryv[])
{
	// (a) shift to the left on query horizontal
	vqueryh = leftShift (vqueryh);
	vqueryh.elem[LOGICALWIDTH-1] = queryh[hoffset++];
	// (b) shift left on updated vector 1 (this places the right-aligned vector 2 as a left-aligned vector 1)
	antiDiag1.simd = antiDiag2.simd;
	antiDiag1 = leftShift (antiDiag1);
	antiDiag2.simd = antiDiag3.simd;
}

static inline void
moveDown (vector_union_t& antiDiag1, vector_union_t& antiDiag2,
	vector_union_t& antiDiag3, int& hoffset, int& voffset, vector_union_t& vqueryh,
		vector_union_t& vqueryv, const short queryh[], const short queryv[])
{
	//(a) shift to the right on query vertical
	vqueryv = rightShift (vqueryv);
	vqueryv.elem[0] = queryv[voffset++];
	//(b) shift to the right on updated vector 2 (this places the left-aligned vector 3 as a right-aligned vector 2)
	antiDiag1.simd = antiDiag2.simd;
	antiDiag2.simd = antiDiag3.simd;
	antiDiag2 = rightShift (antiDiag2);
}

//======================================================================================
// X-DROP ADAPTIVE BANDED ALIGNMENT
//======================================================================================

enum ExtensionDirectionLMC
{
    EXTEND_MC_NONE  = 0,
    EXTEND_MC_LEFT  = 1,
    EXTEND_MC_RIGHT = 2,
    EXTEND_MC_BOTH  = 3
};

template <typename T,typename U>
std::pair<T,U> operator+(const std::pair<T,U> & l,const std::pair<T,U> & r) {
    return {l.first+r.first,l.second+r.second};
}

std::pair<short, short>
LoganXDrop
(
	SeedL & seed,
	std::string const& targetSeg,
	std::string const& querySeg,
	ScoringSchemeL& scoringScheme,
	unsigned short const &scoreDropOff
)
{
	unsigned int hlength = targetSeg.length() + 1;
	unsigned int vlength = querySeg.length()  + 1;

	if (hlength <= 1 || vlength <= 1)
	{
		printf("Error: read length == 0\n");
		exit(1);
	}

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

	vector_t vmatchCost    = set1_func (matchCost   );
	vector_t vmismatchCost = set1_func (mismatchCost);
	vector_t vgapCost      = set1_func (gapCost     );

	//======================================================================================
	// PHASE I (initial values load using dynamic programming)
	//======================================================================================

#ifdef DEBUG
	printf("Phase I\n");
#endif
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

	// this should point to the next value to be loaded into vqueryh and vqueryv
	int hoffset = LOGICALWIDTH;
	int voffset = LOGICALWIDTH;

	// load phase1_data into antiDiag1 vector
	for (int i = 1; i <= LOGICALWIDTH; ++i)
		antiDiag1.elem[i-1] = phase1_data[i][LOGICALWIDTH - i + 1];
	antiDiag1.elem[LOGICALWIDTH] = NINF;

	// load phase1_data into antiDiag2 vector going myRIGHT (our arbitrary decision)
	// the first antiDiag3 computation is going myDOWN
	// shift to the right on updated vector 2 (This places the left-aligned vector 3 as a right-aligned vector 2)
	for (int i = 1; i <= LOGICALWIDTH; ++i)
		antiDiag2.elem[i] = phase1_data[i + 1][LOGICALWIDTH - i + 1];
	antiDiag2.elem[0] = NINF;

	// initialize antiDia3 to -inf
	antiDiag3.simd = set1_func (NINF);

	//======================================================================================
	// PHASE II (core vectorized computation)
	//======================================================================================

	short antiDiagNo = 1;
	short antiDiagBest = antiDiagNo * gapCost;
	short best = 0;

#ifdef DEBUG
	printf("Phase II\n");
#endif

	while(hoffset < hlength && voffset < vlength)
	{

#ifdef DEBUG
	printf("\n");
	print_vector_c(vqueryh.simd);
	print_vector_c(vqueryv.simd);
#endif

		// antiDiagBest initialization
		antiDiagNo++;
		antiDiagBest = antiDiagNo * gapCost;

		// antiDiag1F (final)
		// POST-IT: -1 for a match and 0 for a mismatch
		vector_t m = cmpeq_func (vqueryh.simd, vqueryv.simd);
		m = blendv_func (vmismatchCost, vmatchCost, m);
		vector_t antiDiag1F = add_func (m, antiDiag1.simd);

	#ifdef DEBUG
		printf("antiDiag1: ");
		print_vector_d(antiDiag1.simd);
		printf("antiDiag1F: ");
		print_vector_d(antiDiag1F);
	#endif

		// antiDiag2S (shift)
		vector_union_t antiDiag2S = leftShift (antiDiag2);
	#ifdef DEBUG
		printf("antiDiag2S: ");
		print_vector_d(antiDiag2S.simd);
	#endif
		// antiDiag2M (pairwise max)
		vector_t antiDiag2M = max_func (antiDiag2S.simd, antiDiag2.simd);
	#ifdef DEBUG
		printf("antiDiag2M: ");
		print_vector_d(antiDiag2M);
	#endif
		// antiDiag2F (final)
		vector_t antiDiag2F = add_func (antiDiag2M, vgapCost);
	#ifdef DEBUG
		printf("antiDiag2F: ");
		print_vector_d(antiDiag2F);
	#endif
	#ifdef DEBUG
		printf("antiDiag2: ");
		print_vector_d(antiDiag2.simd);
	#endif
		// Compute antiDiag3
		antiDiag3.simd = max_func (antiDiag1F, antiDiag2F);
		// we need to have always antiDiag3 left-aligned
		antiDiag3.elem[LOGICALWIDTH] = NINF;
	#ifdef DEBUG
		printf("antiDiag3: ");
		print_vector_d(antiDiag3.simd);
	#endif

		// TODO: x-drop termination
		antiDiagBest = *std::max_element(antiDiag3.elem, antiDiag3.elem + VECTORWIDTH);
		if(antiDiagBest < best - scoreDropOff)
		{
			delete [] queryh;
			delete [] queryv;
			return std::make_pair(best, antiDiagBest);
		}

		// update best
		best = (best > antiDiagBest) ? best : antiDiagBest;

		// antiDiag swap, offset updates, and new base load
		int maxpos, max = 0;
		for(int i = 0; i < VECTORWIDTH; ++i)
			if(antiDiag3.elem[i] > max)
			{
				maxpos = i;
				max = antiDiag3.elem[i];
			}

		if(maxpos > MIDDLE)
		//if(antiDiag3.elem[MIDDLE] < antiDiag3.elem[MIDDLE + 1])
		{
			#ifdef DEBUG
			printf("myRIGHT\n");
			#endif
			moveRight (antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);
		}
		else
		{
			#ifdef DEBUG
			printf("myDOWN\n");
			#endif
			moveDown (antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);
		}
	}

	// Phase III (we are on one edge)
	int dir = hoffset >= hlength ? myDOWN : myRIGHT;

#ifdef DEBUG
	printf("Phase III\n");
#endif

	while(hoffset < hlength || voffset < vlength)
	{

	#ifdef DEBUG
		printf("\n");
		print_vector_c(vqueryh.simd);
		print_vector_c(vqueryv.simd);
	#endif
		// antiDiagBest initialization
		antiDiagNo++;
		antiDiagBest = antiDiagNo * gapCost;

		// antiDiag1F (final)
		// POST-IT: -1 for a match and 0 for a mismatch
		vector_t m = cmpeq_func (vqueryh.simd, vqueryv.simd);
		m = blendv_func (vmismatchCost, vmatchCost, m);
		vector_t antiDiag1F = add_func (m, antiDiag1.simd);

	#ifdef DEBUG
		printf("antiDiag1: ");
		print_vector_d(antiDiag1.simd);
		printf("antiDiag1F: ");
		print_vector_d(antiDiag1F);
	#endif

		// antiDiag2S (shift)
		vector_union_t antiDiag2S = leftShift (antiDiag2);
	#ifdef DEBUG
		printf("antiDiag2S: ");
		print_vector_d(antiDiag2S.simd);
	#endif
		// antiDiag2M (pairwise max)
		vector_t antiDiag2M = max_func (antiDiag2S.simd, antiDiag2.simd);
	#ifdef DEBUG
		printf("antiDiag2M: ");
		print_vector_d(antiDiag2M);
	#endif
		// antiDiag2F (final)
		vector_t antiDiag2F = add_func (antiDiag2M, vgapCost);
	#ifdef DEBUG
		printf("antiDiag2F: ");
		print_vector_d(antiDiag2F);
	#endif
	#ifdef DEBUG
		printf("antiDiag2: ");
		print_vector_d(antiDiag2.simd);
	#endif
		// Compute antiDiag3
		antiDiag3.simd = max_func (antiDiag1F, antiDiag2F);
		// we need to have always antiDiag3 left-aligned
		antiDiag3.elem[LOGICALWIDTH] = NINF;
	#ifdef DEBUG
		printf("antiDiag3: ");
		print_vector_d(antiDiag3.simd);
	#endif

		// x-drop termination
		antiDiagBest = *std::max_element(antiDiag3.elem, antiDiag3.elem + VECTORWIDTH);
		if(antiDiagBest < best - scoreDropOff)
		{
			delete [] queryh;
			delete [] queryv;
			return std::make_pair(best, antiDiagBest);
		}

		// update best
		best = (best > antiDiagBest) ? best : antiDiagBest;

		// antiDiag swap, offset updates, and new base load
		if (dir == myRIGHT)
		{
		#ifdef DEBUG
			printf("myRIGHT\n");
		#endif
			moveRight (antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);
		}
		else
		{
		#ifdef DEBUG
			printf("myDOWN\n");
		#endif
			moveDown (antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);
		}
	}

	//======================================================================================
	// PHASE IV (reaching end of sequences)
	//======================================================================================

#ifdef DEBUG
	printf("Phase IV\n");
#endif
	for (int i = 0; i < (LOGICALWIDTH - 3); i++)
	{
		// antiDiag1F (final)
		// POST-IT: -1 for a match and 0 for a mismatch
		vector_t m = cmpeq_func (vqueryh.simd, vqueryv.simd);
		m = blendv_func (vmismatchCost, vmatchCost, m);
		vector_t antiDiag1F = add_func (m, antiDiag1.simd);

	#ifdef DEBUG
		printf("\n");
		printf("antiDiag1: ");
		print_vector_d(antiDiag1.simd);
		printf("antiDiag1F: ");
		print_vector_d(antiDiag1F);
	#endif

		// antiDiag2S (shift)
		vector_union_t antiDiag2S = leftShift (antiDiag2);
	#ifdef DEBUG
		printf("antiDiag2S: ");
		print_vector_d(antiDiag2S.simd);
	#endif
		// antiDiag2M (pairwise max)
		vector_t antiDiag2M = max_func (antiDiag2S.simd, antiDiag2.simd);
	#ifdef DEBUG
		printf("antiDiag2M: ");
		print_vector_d(antiDiag2M);
	#endif

		// antiDiag2F (final)
		vector_t antiDiag2F = add_func (antiDiag2M, vgapCost);
	#ifdef DEBUG
		printf("antiDiag2F: ");
		print_vector_d(antiDiag2F);
	#endif
		// Compute antiDiag3
		antiDiag3.simd = max_func (antiDiag1F, antiDiag2F);
		// we need to have always antiDiag3 left-aligned
		antiDiag3.elem[LOGICALWIDTH] = NINF;
	#ifdef DEBUG
		printf("antiDiag2: ");
		print_vector_d(antiDiag2.simd);
	#endif
	#ifdef DEBUG
		printf("antiDiag3: ");
		print_vector_d(antiDiag3.simd);
	#endif

		// x-drop termination
		antiDiagBest = *std::max_element(antiDiag3.elem, antiDiag3.elem + VECTORWIDTH);
		if(antiDiagBest < best - scoreDropOff)
		{
			delete [] queryh;
			delete [] queryv;
			return std::make_pair(best, antiDiagBest);
		}

		// update best
		best = (best > antiDiagBest) ? best : antiDiagBest;

		// antiDiag swap, offset updates, and new base load
		short nextDir = dir ^ 1;
		// antiDiag swap, offset updates, and new base load
		if (nextDir == myRIGHT)
		{
		#ifdef DEBUG
			printf("myRIGHT\n");
		#endif
			moveRight (antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);
		}
		else
		{
		#ifdef DEBUG
			printf("myDOWN\n");
		#endif
			moveDown (antiDiag1, antiDiag2, antiDiag3, hoffset, voffset, vqueryh, vqueryv, queryh, queryv);
		}
		// direction update
		dir = nextDir;
	}

	// find positions of longest extension and update seed
	setBeginPositionH(seed, 0);
	setBeginPositionV(seed, 0);
	// this is wrong
	setEndPositionH(seed, hoffset);
	setEndPositionV(seed, voffset);

	delete [] queryh;
	delete [] queryv;

	return std::make_pair(best, antiDiagBest);
}

// 1st prototype
std::pair<short, short>
LoganAVX2
(
	SeedL& seed,
	ExtensionDirectionLMC direction,
	std::string const& target,
	std::string const& query,
	ScoringSchemeL& scoringScheme,
	unsigned short const &scoreDropOff
)
{
	// TODO: check scoring scheme correctness/input parameters
	// Left extension
    if (direction == EXTEND_MC_LEFT)
    {
    	// string substr (size_t pos = 0, size_t len = npos) const;
		// returns a newly constructed string object with its value initialized to a copy of a substring of this object
		std::string targetPrefix = target.substr(0, getBeginPositionH(seed));	// from read start til start seed (seed not included)
		std::string queryPrefix = query.substr(0, getBeginPositionV(seed));	// from read start til start seed (seed not included)
		std::reverse( targetPrefix.begin(), targetPrefix.end() );
		std::reverse( queryPrefix.begin(), queryPrefix.end() );
		return LoganXDrop( seed, targetPrefix, queryPrefix, scoringScheme, scoreDropOff );
    }

    else if (direction == EXTEND_MC_RIGHT)
    {
    	// Do not extend to the right if we are already at the beginning of an
		// infix or the sequence itself.
		std::string targetSuffix = target.substr(getEndPositionH(seed), target.length()); 	// from end seed until the end (seed not included)
		std::string querySuffix = query.substr(getEndPositionV(seed), query.length());		// from end seed until the end (seed not included)
		return LoganXDrop( seed, targetSuffix, querySuffix, scoringScheme, scoreDropOff );
	}

	else
	{
		std::pair<short, short> left;
		std::pair<short, short> right;

		std::string targetPrefix = target.substr(0, getBeginPositionH(seed));	// from read start til start seed (seed not included)
		std::string queryPrefix = query.substr(0, getBeginPositionV(seed));	// from read start til start seed (seed not included)
		std::reverse( targetPrefix.begin(), targetPrefix.end() );
		std::reverse( queryPrefix.begin(), queryPrefix.end() );
		left = LoganXDrop( seed, targetPrefix, queryPrefix, scoringScheme, scoreDropOff );

		std::string targetSuffix = target.substr(getEndPositionH(seed), target.length()); 	// from end seed until the end (seed not included)
		std::string querySuffix = query.substr(getEndPositionV(seed), query.length());		// from end seed until the end (seed not included)
		right = LoganXDrop( seed, targetSuffix, querySuffix, scoringScheme, scoreDropOff );

		return left + right;
	}
}



