
//==================================================================
// Title:  C++ x-drop seed-and-extend alignment algorithm
// Author: G. Guidi, A. Zeni
// Date:   6 March 2019
//==================================================================

// -----------------------------------------------------------------
// Function extendSeedL                         [GappedXDrop, noSIMD]
// -----------------------------------------------------------------

//remove asserts to speedup 

//#define DEBUG

#include<vector>
#include<iostream>
#include<omp.h>
#include<algorithm>
#include<inttypes.h>
#include<assert.h>
#include<iterator>
#include"logan.h"
#include"score.h"
#include <immintrin.h> // For AVX instructions
// #include <bits/stdc++.h>

enum ExtensionDirectionL
{
	EXTEND_NONEL  = 0,
	EXTEND_LEFTL  = 1,
	EXTEND_RIGHTL = 2,
	EXTEND_BOTHL  = 3
};

//template<typename TSeedL, typename int, typename int>
inline void
updateExtendedSeedL(SeedL& seed,
					ExtensionDirectionL direction, //as there are only 4 directions we may consider even smaller data types
					unsigned short cols,
					unsigned short rows,
					unsigned short lowerDiag,
					unsigned short upperDiag)
{
	//TODO 
	//functions that return diagonal from seed
	//functions set diagonal for seed

	if (direction == EXTEND_LEFTL)
	{
		unsigned short beginDiag = seed.beginDiagonal;
		// Set lower and upper diagonals.
		
		if (getLowerDiagonal(seed) > beginDiag + lowerDiag)
			setLowerDiagonal(seed, beginDiag + lowerDiag);
		if (getUpperDiagonal(seed) < beginDiag + upperDiag)
			setUpperDiagonal(seed, beginDiag + upperDiag);

		// Set new start position of seed.
		setBeginPositionH(seed, getBeginPositionH(seed) - rows);
		setBeginPositionV(seed, getBeginPositionV(seed) - cols);
	} else {  // direction == EXTEND_RIGHTL
		// Set new lower and upper diagonals.
		unsigned short endDiag = seed.endDiagonal;
		if (getUpperDiagonal(seed) < endDiag - lowerDiag)
			setUpperDiagonal(seed, (endDiag - lowerDiag));
		if (getLowerDiagonal(seed) > (endDiag - upperDiag))
			setLowerDiagonal(seed, endDiag - upperDiag);

		// Set new end position of seed.
		setEndPositionH(seed, getEndPositionH(seed) + rows);
		setEndPositionV(seed, getEndPositionV(seed) + cols);
		
	}
	// assert(seed.upperDiagonal >= seed.lowerDiagonal);
	// assert(seed.upperDiagonal >= seed.beginDiagonal);
	// assert(seed.upperDiagonal >= seed.endDiagonal);
	// assert(seed.beginDiagonal >= seed.lowerDiagonal);
	// assert(seed.endDiagonal >= seed.lowerDiagonal);
	
}

inline void
calcExtendedLowerDiag(unsigned short& lowerDiag,
					   unsigned short const & minCol,
					   unsigned short const & antiDiagNo)
{
	unsigned short minRow = antiDiagNo - minCol;
	if (minCol - minRow < lowerDiag)
		lowerDiag = minCol - minRow;
}

inline void
calcExtendedUpperDiag(unsigned short & upperDiag,
					   unsigned short const &maxCol,
					   unsigned short const &antiDiagNo)
{
	unsigned short maxRow = antiDiagNo + 1 - maxCol;
	if (maxCol - 1 - maxRow > upperDiag)
		upperDiag = maxCol - 1 - maxRow;
}

//inline void
//swapAntiDiags(std::vector<int> & antiDiag1,
//			   std::vector<int> & antiDiag2,
//			   std::vector<int> & antiDiag3)
//{
//	std::vector<int> temp = antiDiag1;
//	antiDiag1 = antiDiag2;
//	antiDiag2 = antiDiag3;
//	antiDiag3 = temp;
//}

//inline int
//initAntiDiag3(std::vector<int>& antiDiag3,
//			   unsigned short const & offset,
//			   unsigned short const & maxCol,
//			   unsigned short const & antiDiagNo,
//			   int const & minScore,
//			   short const & gapCost,
//			   int const & undefined)
//{
//	antiDiag3.resize(maxCol + 1 - offset);
//
//	antiDiag3[0] = undefined;
//	antiDiag3[maxCol - offset] = undefined;
//
//	if (antiDiagNo * gapCost > minScore)
//	{
//		if (offset == 0) // init first column
//			antiDiag3[0] = antiDiagNo * gapCost;
//		if (antiDiagNo - maxCol == 0) // init first row
//			antiDiag3[maxCol - offset] = antiDiagNo * gapCost;
//	}
//	return offset;
//}

//inline void
//initAntiDiags(std::vector<int> & antiDiag2,
//			   std::vector<int> & antiDiag3,
//			   short const& dropOff,
//			   short const& gapCost,
//			   int const& undefined)
//{
//	// antiDiagonals will be swaped in while loop BEFORE computation of antiDiag3 entries
//	//  -> no initialization of antiDiag1 necessary
//
//	//antiDiag2.resize(1);
//	antiDiag2[0] = 0;
//
//	//antiDiag3.resize(2);
//	if (-gapCost > dropOff)
//	{
//		antiDiag3[0] = undefined;
//		antiDiag3[1] = undefined;
//	}
//	else
//	{
//		antiDiag3[0] = gapCost;
//		antiDiag3[1] = gapCost;
//	}
//}

//int
//extendSeedLGappedXDropLeftAVX2(
//		SeedL & seed,
//		std::string const & querySeg,
//		std::string const & databaseSeg,
//		ExtensionDirectionL const & direction,
//		ScoringSchemeL &scoringScheme,
//		short const &scoreDropOff)
//{

typedef union {
	__m256i simd;
	int16_t elem[16] = {-1}; // 
} _m256i_16_t;

void print_m256i_16(__m256i a) {

	_m256i_16_t t;
	t.simd = a;

	printf("{%d,%d,%d,%d,%d,%d,%d,%d,"
			"%d,%d,%d,%d,%d,%d,%d,%d}\n",
			t.elem[ 0], t.elem[ 1], t.elem[ 2], t.elem[ 3],
			t.elem[ 4], t.elem[ 5], t.elem[ 6], t.elem[ 7],
			t.elem[ 8], t.elem[ 9], t.elem[10], t.elem[11],
			t.elem[12], t.elem[13], t.elem[14], t.elem[15]
			);
}

inline _m256i_16_t 
shiftRight (_m256i_16_t& data, const short& count) {

	_m256i_16_t m; 

	for(short i = 0; i < 15; i++)         // data are saved in reversed order
		m.elem[i] = data.elem[i + count]; // count is the offset difference

	// replicating last element
	m.elem[15] = data.elem[15];

	return m;
}

//inline void 
//maskOp (_m256i_16_t& dat, _m256i_16_t& antiDiag3, short& minCol, short& maxCol, short& offset3, short& offset2) {
//
//	_m256i_16_t tmp;
//
//	tmp.simd = _mm256_load_si256 (&dat.simd); // 0,1 in 1,2 di antiDiag3
//	dat.simd = _mm256_load_si256 (&antiDiag3.simd);
//
//	for(short i = minCol; i < maxCol; i++) // data are saved in reversed order
//	{
//		dat.elem[i - offset3] = tmp.elem[i - 1 - offset2];
//	}
//}

//inline _m256i_16_t maskOp (_m256i_16_t& tmp, _m256i_16_t& antiDiag3) {
//
//	_m256i_16_t m; 
//	for(int16_t i = 0; i < 15; i++) // data are saved in reversed order
//	{
//		m.elem[i+1] = tmp.elem[i];
//	}
//	m.elem[0] = antiDiag3.elem[0];
//	return m;
//}

int
extendSeedLGappedXDropRightAVX2(
		SeedL & seed,
		std::string const & querySeg,
		std::string const & databaseSeg,
		ExtensionDirectionL const & direction,
		ScoringSchemeL &scoringScheme,
		short const &scoreDropOff)
{
// Put the largest number of the 16 numbers in vm into m.
// Function from https://github.com/giuliaguidi/Complete-Striped-Smith-Waterman-Library/blob/master/src/ssw.c
#define max16(m, vm) 	(vm) = _mm256_max_epi16((vm), _mm256_srli_epi16((vm), 8)); \
				  		(vm) = _mm256_max_epi16((vm), _mm256_srli_epi16((vm), 4)); \
				  		(vm) = _mm256_max_epi16((vm), _mm256_srli_epi16((vm), 2)); \
				  		(vm) = _mm256_max_epi16((vm), _mm256_srli_epi16((vm), 1)); \
				  		(m)  = _mm256_extract_epi16((vm), 0)

	//std::chrono::duration<double>  diff;
	unsigned short cols = querySeg.length() + 1;
	unsigned short rows = databaseSeg.length() + 1;

	if (rows <= 1 || cols <= 1)
		return 0;

	// convert from string to __m256i* array
	short* query  = new short[cols]; // this is the entire query 			
	short* target = new short[rows];

	std::copy(querySeg.begin(), querySeg.end(), query); 
	std::copy(databaseSeg.begin(), databaseSeg.end(), target); 

	unsigned short len = 2 * std::max (cols, rows); // number of antidiagonals (does not change in any implementation)
	const short minErrScore = std::numeric_limits<short>::min() / len; // minimal allowed error penalty
	setScoreGap(scoringScheme, std::max (scoreGap(scoringScheme), minErrScore));

	setScoreMismatch(scoringScheme, std::max(scoreMismatch(scoringScheme), minErrScore));

	short gapCost   = scoreGap(scoringScheme);
	short undefined = std::numeric_limits<short>::min() - gapCost; // out minus infinity 		

	_m256i_16_t antiDiag1; 	// 16 (vector width) 16-bit integers
	_m256i_16_t antiDiag2; 	// 16 (vector width) 16-bit integers
	_m256i_16_t antiDiag3; 	// 16 (vector width) 16-bit integers

	// 
	//
	
	short minCol = 1;
	short maxCol = 2;

	short offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
	short offset2 = 0; //                                                       in antiDiag2
	short offset3 = 0; //                                                       in antiDiag3

	antiDiag2.simd = _mm256_setzero_si256 (); 			// initialize vector with zeros

	if (-gapCost > scoreDropOff) // initialize vector with -inf
	{
		antiDiag3.simd = _mm256_set1_epi16 (undefined); 	// broadcast 16-bit integer a to all elements of dst
	}
	else // initialize vector with gapCost
	{
		antiDiag3.simd = _mm256_set1_epi16 (gapCost); 	// broadcast 16-bit integer a to all elements of dst
	}

	short antiDiagNo = 1; 	 // the currently calculated anti-diagonal
	short best       = 0; 	 // maximal score value in the DP matrix (for drop-off calculation)

	unsigned short lowerDiag  = 0; 
	unsigned short upperDiag  = 0;

	short antiDiag1size = 0;	 // init
	short antiDiag2size = 1;	 // init
	short antiDiag3size = 2;	 // init

	_m256i_16_t tmp;

	while (minCol < maxCol) // this diff cannot be greater than 16
	{
		// data must be aligned when loading to and storing to avoid severe performance penalties
		++antiDiagNo;
		// swap antiDiags
		tmp.simd       = _mm256_load_si256 (&antiDiag1.simd);
		antiDiag1.simd = _mm256_load_si256 (&antiDiag2.simd);
		antiDiag2.simd = _mm256_load_si256 (&antiDiag3.simd);
		antiDiag3.simd = _mm256_load_si256 (&tmp.simd); 	// init to -inf at each iteration

		offset1 = offset2;
		offset2 = offset3;
		offset3 = minCol - 1;

		antiDiag1size = antiDiag2size;
		antiDiag2size = antiDiag3size;
		antiDiag3size = maxCol + 1 - offset3; // double check this in original seqan

		int16_t antiDiagBest  = antiDiagNo * gapCost; // init

		antiDiag3.elem[0] = undefined;
		antiDiag3.elem[maxCol - offset3] = undefined;

		if (antiDiagNo * gapCost > best - scoreDropOff)
		{
			if (offset3 == 0) // init first column
				antiDiag3.elem[0] = antiDiagNo * gapCost;
			if (antiDiagNo - maxCol == 0) // init first row
				antiDiag3.elem[maxCol - offset3] = antiDiagNo * gapCost;
		}

		printf("antidiag3 init ");
		print_m256i_16(antiDiag3.simd);
		//for (int16_t col = minCol; col < maxCol; col += 16)
		//{
		short nbases = maxCol - minCol;
		assert (nbases < 17);
		printf("minCol %d maxCol %d offset3 %d offset2 %d offset1 %d\n", minCol, maxCol, offset3, offset2, offset1);
		// im considering only right side
		short queryPos = minCol - 1;// - offset3 preserve right alignment when computing tmpscore
		short dbPos = antiDiagNo - maxCol;// + offset3);

		// calculate matrix entry (-> antiDiag3[col])
		// TODO : double check after compilation and put into a separate function
		printf("antidiag1 ");
		print_m256i_16(antiDiag1.simd);
		printf("antidiag2 ");
		print_m256i_16(antiDiag2.simd);

		tmp = shiftRight (antiDiag2, 1);
		printf("antidiag2 - 1 ");
		print_m256i_16(tmp.simd);
		tmp.simd = _mm256_max_epi16 (antiDiag2.simd, tmp.simd);
		tmp.simd = _mm256_add_epi16 (tmp.simd, _mm256_set1_epi16 (gapCost));
		printf("max antidiag2 ");
		print_m256i_16(tmp.simd);

		_m256i_16_t _m_query, _m_target;

		_m_query.simd  = _mm256_loadu_si256 ((__m256i*)(query  + queryPos)); // load sixteen bases from querySeg
		_m_target.simd = _mm256_loadu_si256 ((__m256i*)(target + dbPos)); 	// load sixteen bases from targetSeg

		// bases on target need to be reversed
		std::reverse(_m_target.elem, _m_target.elem + nbases);

		//print_m256i_16 (_m_query.simd );
		//print_m256i_16 (_m_target.simd);
		// here : score(scoringScheme, querySeg[queryPos], databaseSeg[dbPos])
		_m256i_16_t tmpscore;
		tmpscore.simd = _mm256_cmpeq_epi16 (_m_query.simd, _m_target.simd); // 0xFFFF where equal, 0 where different
		tmpscore.simd = _mm256_blendv_epi8 (_mm256_set1_epi16 (scoreMismatch(scoringScheme)), _mm256_set1_epi16 (scoreMatch(scoringScheme)), tmpscore.simd);
		// here : add tmpscore to antiDiag1 	
		tmpscore.simd = _mm256_add_epi16 (tmpscore.simd, antiDiag1.simd);
		printf("max antidiag1 ");
		print_m256i_16(tmpscore.simd);
		_m256i_16_t tmpscore2;
		for(short i = 0; i < 15; i++)         // data are saved in reversed order
			tmpscore2.elem[i+1] = tmpscore.elem[i]; // count is the offset difference

		// replicating last element
		tmpscore2.elem[0] = undefined;


		//unsigned short count = offset3-offset1;
		//tmpscore = shiftRight (tmpscore, count);
		printf("max antidiag1 shifted ");
		print_m256i_16(tmpscore2.simd);
		//count = offset3-offset2;
		//tmp = shiftRight (tmpscore, count);

		printf("max antidiag2 ");
		print_m256i_16(tmp.simd);

		tmp.simd = _mm256_max_epi16 (tmp.simd, tmpscore2.simd);

		printf("max ");
		print_m256i_16(tmp.simd);

		for(unsigned short i = minCol-offset3; i < maxCol; i++) // data are saved in reversed order
			antiDiag3.elem[i] = tmp.elem[i];

		printf("antidiag3 wanna be ");
		print_m256i_16(antiDiag3.simd);
		__m256i mask = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (best - scoreDropOff), antiDiag3.simd);  // 0xFFFF true (-1), 0 false
		antiDiag3.simd = _mm256_blendv_epi8 (antiDiag3.simd, _mm256_set1_epi16 (undefined), mask);

		printf("antiDiag3 updated ");
		print_m256i_16(antiDiag3.simd);

		antiDiagBest = *std::max_element(antiDiag3.elem + offset3, antiDiag3.elem + antiDiag3size);
		best = (best > antiDiagBest) ? best : antiDiagBest;
		printf("best %d antiDiagBest %d\n\n", best, antiDiagBest);

		//int16_t bestCol   = 0;
		//int16_t bestRow   = 0;
		//int16_t bestScore = 0;

		// seed extension wrt best score
		// TODO : not in seqan -- do this later
		//__m256i mask10 = _mm256_cmpgt_epi16(antiDiagBest + 1, best)
		//if (antiDiagBest >= best)
		//{
		//	bestCol	= length(antiDiag3) + offset3 - 2;
		//	bestRow	= antiDiagNo - bestExtensionCol;
		//	bestScore	= best;
		//}

		// calculate new minCol and minCol
		while (minCol - offset3 < antiDiag3size && antiDiag3.elem[minCol - offset3] == undefined &&
			   minCol - offset2 - 1 < antiDiag2size && antiDiag2.elem[minCol - offset2 - 1] == undefined)
		{
			++minCol;
		}

		// calculate new maxCol
		while (maxCol - offset3 > 0 && (antiDiag3.elem[maxCol - offset3 - 1] == undefined) &&
									(antiDiag2.elem[maxCol - offset2 - 1] == undefined))
		{
			--maxCol;
		}
		maxCol++;

		// this do not need to be vectorized just fix types
		// calculate new lowerDiag and upperDiag of extended seed
		calcExtendedLowerDiag(lowerDiag, minCol, antiDiagNo);
		calcExtendedUpperDiag(upperDiag, maxCol - 1, antiDiagNo);

		// end of databaseSeg reached?
		minCol = (minCol > (antiDiagNo + 2 - rows)) ? minCol : (antiDiagNo + 2 - rows);
		// end of querySeg reached?
		maxCol = (maxCol < cols) ? maxCol : cols;
	}

	// find positions of longest extension
	// reached ends of both segments
	int16_t longestExtensionCol = antiDiag3size + offset3 - 2;
	int16_t longestExtensionRow = antiDiagNo - longestExtensionCol;
	int16_t longestExtensionScore = antiDiag3.elem[longestExtensionCol - offset3];

	if (longestExtensionScore == undefined)
	{
		if (antiDiag2.elem[antiDiag2size-2] != undefined)
		{
			// reached end of query segment
			longestExtensionCol = antiDiag2size + offset2 - 2;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2.elem[longestExtensionCol - offset2];
		}
		else if (antiDiag2size > 2 && antiDiag2.elem[antiDiag2size-3] != undefined)
		{
			// reached end of database segment
			longestExtensionCol = antiDiag2size + offset2 - 3;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2.elem[longestExtensionCol - offset2];
		}
	}

	if (longestExtensionScore == undefined)
	{
		// general case
		for (int i = 0; i < antiDiag1size; ++i)
		{
			if (antiDiag1.elem[i] > longestExtensionScore)
			{
				longestExtensionScore = antiDiag1.elem[i];
				longestExtensionCol = i + offset1;
				longestExtensionRow = antiDiagNo - 2 - longestExtensionCol;
			}
		}
	}

	delete [] query;
	delete [] target;
 
 	//std::cout << "longestExtensionScore : " << longestExtensionScore << std::endl; 

	// update seed
	if (longestExtensionScore != undefined)
		updateExtendedSeedL(seed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);

	return longestExtensionScore; 
}

inline int
extendSeedL(SeedL& seed,
			ExtensionDirectionL direction,
			std::string const& target,
			std::string const& query,
			ScoringSchemeL & penalties,
			int const& XDrop,
			int const& kmer_length)
{
	assert(scoreGapExtend(penalties) < 0); 
	assert(scoreGapOpen(penalties) < 0); 	// this is the same ad GapExtend for linear scoring scheme
	//assert(scoreMismatch(penalties) < 0);
	//assert(scoreMatch(penalties) > 0); 
	assert(scoreGapOpen(penalties) == scoreGapExtend(extend));

	int scoreLeft=0;
	int scoreRight=0;
	Result scoreFinal;

	//if (direction == EXTEND_LEFTL || direction == EXTEND_BOTHL)
	//{
	//	// string substr (size_t pos = 0, size_t len = npos) const;
	//	// returns a newly constructed string object with its value initialized to a copy of a substring of this object
	//	std::string targetPrefix = target.substr(0, getBeginPositionH(seed));	// from read start til start seed (seed not included)
	//	std::string queryPrefix = query.substr(0, getBeginPositionV(seed));	// from read start til start seed (seed not included)
	//	scoreLeft = extendSeedLGappedXDropOneDirectionAVX2(seed, queryPrefix, targetPrefix, EXTEND_LEFTL, penalties, XDrop);
	//}

	if (direction == EXTEND_RIGHTL || direction == EXTEND_BOTHL)
	{
		// Do not extend to the right if we are already at the beginning of an
		// infix or the sequence itself.
		std::string targetSuffix = target.substr(getEndPositionH(seed), target.length()); 	// from end seed until the end (seed not included)
		std::string querySuffix = query.substr(getEndPositionV(seed), query.length());		// from end seed until the end (seed not included)

		scoreRight = extendSeedLGappedXDropRightAVX2(seed, querySuffix, targetSuffix, EXTEND_RIGHTL, penalties, XDrop);
	}

	//Result myalignment(kmer_length); // do not add KMER_LENGTH later
	//std::cout<<"scoreLeft logan: "<<scoreLeft<<" scoreRight logan: "<<scoreRight<<std::endl;
	//myalignment.score = scoreLeft + scoreRight + kmer_length; // we have already accounted for seed match score
	int res = scoreLeft + scoreRight + kmer_length;
	//myalignment.myseed = seed;	// extended begin and end of the seed

	return res;
}

#ifdef DEBUG

#endif

