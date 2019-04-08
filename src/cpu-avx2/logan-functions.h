
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
	int16_t elem[16] = {-1};
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

inline _m256i_16_t shiftLeft (_m256i_16_t& data) {

	_m256i_16_t m; 
	for(int16_t i = 0; i < 15; i++) // data are saved in reversed order
	{
		m.elem[i] = data.elem[i+1];
	}
	m.elem[15] = data.elem[15];
	return m;
}

inline void 
maskOp (_m256i_16_t& dat, _m256i_16_t& antiDiag3, short& minCol, short& maxCol, short& offset3) {

	_m256i_16_t tmp;

	tmp.simd = _mm256_load_si256 (&dat.simd); // 0,1 in 1,2 di antiDiag3
	dat.simd = _mm256_load_si256 (&antiDiag3.simd);

	for(short i = minCol; i < maxCol; i++) // data are saved in reversed order
	{
		dat.elem[i] = tmp.elem[i-1-offset3];
	}
}

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

	//std::chrono::duration<double>  diff;
	unsigned short cols = querySeg.length()+1;
	unsigned short rows = databaseSeg.length()+1;

	if (rows == 1 || cols == 1)
		return 0;

	// convert from string to __m256i* array
	int16_t* query  = new int16_t[cols];
	int16_t* target = new int16_t[rows];

	std::copy(querySeg.begin(), querySeg.end(), query); 
	std::copy(databaseSeg.begin(), databaseSeg.end(), target); 

	unsigned short len = 2 * std::max (cols, rows); // number of antidiagonals (does not change in any implementation)
	const short minErrScore = std::numeric_limits<short>::min() / len; // minimal allowed error penalty
	setScoreGap(scoringScheme, std::max (scoreGap(scoringScheme), minErrScore));

	setScoreMismatch(scoringScheme, std::max(scoreMismatch(scoringScheme), minErrScore));

	int16_t gapCost   = scoreGap(scoringScheme);
	int16_t undefined = std::numeric_limits<short>::min() - gapCost;

	_m256i_16_t antiDiag1; 	// 16 (vector width) 16-bit integers
	_m256i_16_t antiDiag2; 	// 16 (vector width) 16-bit integers
	_m256i_16_t antiDiag3; 	// 16 (vector width) 16-bit integers

	int16_t minCol = 1;
	int16_t maxCol = 2;

	int16_t offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
	int16_t offset2 = 0; //                                                       in antiDiag2
	int16_t offset3 = 0; //                                                       in antiDiag3

	antiDiag2.simd = _mm256_setzero_si256 (); 			// initialize vector with zeros

	if (-gapCost > scoreDropOff) // initialize vector with -inf
	{
		antiDiag3.simd = _mm256_set1_epi16 (undefined); 	// broadcast 16-bit integer a to all elements of dst
	}
	else // initialize vector with gapCost
	{
		antiDiag3.simd = _mm256_set1_epi16 (gapCost); 	// broadcast 16-bit integer a to all elements of dst
	}

	int16_t antiDiagNo = 1; 	 // the currently calculated anti-diagonal
	int16_t best       = 0; 	 // maximal score value in the DP matrix (for drop-off calculation)

	unsigned short lowerDiag  = 0; 
	unsigned short upperDiag  = 0;

	int16_t antiDiag1size = 0;	 // init
	int16_t antiDiag2size = 1;	 // init
	int16_t antiDiag3size = 2;	 // init

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

		//for (int16_t col = minCol; col < maxCol; col += 16)
		//{
		short nbases = maxCol - minCol;
		assert (nbases < 17);

		// im considering only right side
		short queryPos = minCol - 1; 
		short dbPos = antiDiagNo - maxCol;

		// calculate matrix entry (-> antiDiag3[col])
		// TODO : double check after compilation and put into a separate function
		tmp = shiftLeft (antiDiag2);

		tmp.simd = _mm256_max_epi16 (antiDiag2.simd, tmp.simd);
		tmp.simd = _mm256_add_epi16 (tmp.simd, _mm256_set1_epi16 (gapCost));
		// need to consider only numBases bases from the sequences
		_m256i_16_t _m_query, _m_target;
		_m_query.simd = _mm256_loadu_si256 ((__m256i*)(query  + queryPos)); // load sixteen bases from querySeg
		_m_target.simd = _mm256_loadu_si256 ((__m256i*)(target + dbPos)); 	// load sixteen bases from targetSeg

		// bases on target need to be reversed
		std::reverse(_m_target.elem, _m_target.elem + nbases);

		// here : score(scoringScheme, querySeg[queryPos], databaseSeg[dbPos])
		__m256i tmpscore = _mm256_cmpeq_epi16 (_m_query.simd, _m_target.simd); // 0xFFFF where equal, 0 where different
		tmpscore = _mm256_blendv_epi8 (_mm256_set1_epi16 (scoreMismatch(scoringScheme)), _mm256_set1_epi16 (scoreMatch(scoringScheme)), tmpscore);
		// here : add tmpscore to antiDiag1 	
		tmpscore = _mm256_add_epi16 (tmpscore, antiDiag1.simd);
		tmp.simd = _mm256_max_epi16 (tmp.simd, tmpscore);
		//printf("tmp antiDiag1 ");
		//print_m256i_16(tmp.simd);
		//printf("antiDiag3 ");
		//print_m256i_16(antiDiag3.simd);
		//printf("minCol %d maxCol %d offset3 %d\n", minCol, maxCol, offset3);
		maskOp (tmp, antiDiag3, minCol, maxCol, offset3);//, minCol, maxCol);

		// issue here I'm updating wrong cell
		//printf("tmp ");
		//print_m256i_16(tmp.simd);

		__m256i mask = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (best - scoreDropOff), tmp.simd);  // 0xFFFF true (-1), 0 false
		//printf("mask ");
		//print_m256i_16(mask);
		antiDiag3.simd = _mm256_blendv_epi8 (tmp.simd, _mm256_set1_epi16 (undefined), mask);
		//tappabuco = _mm256_load_si256 (&antiDiag3.simd);
		printf("antiDiag3 updated ");
		print_m256i_16(antiDiag3.simd);
		// TODO: skipping control here double check 	
		// if true, this should contain antiDiagBest it shouldn't harm
		// max should be in the first position double check
		//}
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
		//printf("minCol - offset3 %d antiDiag3size %d antiDiag3.elem[minCol - offset3] %d\n", minCol - offset3, antiDiag3size, antiDiag3.elem[minCol - offset3]);
		//printf("minCol - offset2 %d antiDiag2size %d antiDiag2.elem[minCol - offset2 - 1] %d\n", minCol - offset2, antiDiag2size, antiDiag2.elem[minCol - offset2 - 1]);
		//print_m256i_16(antiDiag3.simd);
		while (minCol - offset3 < antiDiag3size && antiDiag3.elem[minCol - offset3] == undefined &&
			   minCol - offset2 - 1 < antiDiag2size && antiDiag2.elem[minCol - offset2 - 1] == undefined)
		{
			++minCol;
		}

		// these are ones if verified
		//__m256i condition1 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (antiDiag3size), _mm256_set1_epi16 (minCol - offset3));
		//__m256i condition3 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (antiDiag2size), _mm256_set1_epi16 (minCol - offset2 - 1));
		//__m256i condition2 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag3.elem[minCol - offset3]), _mm256_set1_epi16 (undefined));
		//__m256i condition4 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag2.elem[minCol - offset2 - 1]), _mm256_set1_epi16 (undefined));
		//__m256i condition5 = _mm256_and_si256 (_mm256_and_si256 (condition1, condition2), _mm256_and_si256 (condition3, condition4));
		//while(condition5)
		//{
		//	// incremented by one if true, otherwise no increment
		//	minCol += _mm256_extract_epi16 (condition5, 0);
		//	// update conditions
		//	condition1 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (antiDiag3size), _mm256_set1_epi16 (minCol - offset3));
		//	condition3 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (antiDiag2size), _mm256_set1_epi16 (minCol - offset2 - 1));
		//	condition2 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag3.elem[minCol - offset3]), _mm256_set1_epi16 (undefined));
		//	condition4 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag2.elem[minCol - offset2 - 1]), _mm256_set1_epi16 (undefined));
		//	condition5 = _mm256_and_si256 (_mm256_and_si256 (condition1, condition2), _mm256_and_si256 (condition3, condition4));
		//}

		// calculate new maxCol
		while (maxCol - offset3 > 0 && (antiDiag3.elem[maxCol - offset3 - 1] == undefined) &&
									   (antiDiag2.elem[maxCol - offset2 - 1] == undefined))
		{
			--maxCol;
		}
		maxCol++;
		//__m256i condition6 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (maxCol - offset3), _mm256_setzero_si256 ());
		//__m256i condition7 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag3.elem[maxCol - offset3 - 1]), _mm256_set1_epi16 (undefined));
		//__m256i condition8 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag2.elem[maxCol - offset2 - 1]), _mm256_set1_epi16 (undefined));
		//__m256i condition9 = _mm256_and_si256 (_mm256_and_si256 (condition6, condition7), condition8);
		//while(condition9)
		//{
		//	// decremented by one if true, otherwise no decrement
		//	maxCol -= _mm256_extract_epi16 (condition9, 0);
		//	// update conditions
		//	condition6 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (maxCol - offset3), _mm256_setzero_si256 ());
		//	condition7 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag3.elem[maxCol - offset3 - 1]), _mm256_set1_epi16 (undefined));
		//	condition8 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag2.elem[maxCol - offset2 - 1]), _mm256_set1_epi16 (undefined));
		//	condition9 = _mm256_and_si256 (_mm256_and_si256 (condition6, condition7), condition8);
		//}
		//maxCol++;
		// this do not need to be vectorized just fix types
		// calculate new lowerDiag and upperDiag of extended seed
		calcExtendedLowerDiag(lowerDiag, minCol, antiDiagNo);
		calcExtendedUpperDiag(upperDiag, maxCol - 1, antiDiagNo);

		// end of databaseSeg reached?
		minCol = (minCol > (antiDiagNo + 2 - rows)) ? minCol : (antiDiagNo + 2 - rows);
		//std::cout << "minCol : " << minCol << std::endl;
		// end of querySeg reached?
		maxCol = (maxCol < cols) ? maxCol : cols;
		//std::cout << "maxCol : " << maxCol << std::endl;
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

