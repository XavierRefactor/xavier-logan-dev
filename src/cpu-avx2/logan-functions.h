
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

union {
	__m256i simdDiag;
	__int16 antiDiag[16];

} myDiag;

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

	unsigned short len = 2 * std::max(cols, rows); // number of antidiagonals (does not change in any implementation)
	short const minErrScore = std::numeric_limits<short>::min() / len; // minimal allowed error penalty
	setScoreGap(scoringScheme, std::max(scoreGap(scoringScheme), minErrScore));

	setScoreMismatch(scoringScheme, std::max(scoreMismatch(scoringScheme), minErrScore));

	__m256i gapCost   = _mm256_set1_epi16 ( scoreGap(scoringScheme) );
	__m256i undefined = _mm256_set1_epi16 ( _mm256_sub_epi16 ( _mm256_set1_epi16 ( std::numeric_limits<short>::min() ), gapCost );

	myDiag antiDiag1; 	// 16 (vector width) 16-bit integers
	myDiag antiDiag2; 	// 16 (vector width) 16-bit integers
	myDiag antiDiag3; 	// 16 (vector width) 16-bit integers

	antiDiag1.elem[16] = {0}; // init
	antiDiag2.elem[16] = {0}; // init
	antiDiag3.elem[16] = {0}; // init

	__int16 minCol = 1;
	__int16 maxCol = 2;

	__int16 offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
	__int16 offset2 = 0; //                                                       in antiDiag2
	__int16 offset3 = 0; //                                                       in antiDiag3

	antiDiag2.simd = _mm256_setzero_si256 (); 			// initialize vector with zeros

	//antiDiag3.resize(2);
	if (-gapCost > dropOff) // initialize vector with -inf
	{
		antiDiag3.simd = _mm256_set1_epi16 (undefined); 	// broadcast 16-bit integer a to all elements of dst
	}
	else // initialize vector with gapCost
	{
		antiDiag3.simd = _mm256_set1_epi16 (gapCost); 	// broadcast 16-bit integer a to all elements of dst
	}

	__int16 antiDiagNo = 1; 	 // the currently calculated anti-diagonal
	__int16 best       = 0; 	 // maximal score value in the DP matrix (for drop-off calculation)

	__int16 lowerDiag  = 0; 
	__int16 upperDiag  = 0;

	__int16 antiDiag1size = 0;	 // init
	__int16 antiDiag2size = 1;	 // init
	__int16 antiDiag3size = 2;	 // init

	while (minCol < maxCol) // this diff cannot be greater than 16
	{
		// data must be aligned when loading to and storing to avoid severe performance penalties
		++antiDiagNo;
		// swap antiDiags
		antiDiag1.simd = _mm256_load_epi16 (&antiDiag2);
		antiDiag2.simd = _mm256_load_epi16 (&antiDiag3);
		antiDiag3.simd = _mm256_set1_epi16 (undefined); 	// init to -inf at each iteration

		// antiDiag3.size() = maxCol+1-offset (resize in original initDiag3) : double check 
		antiDiag1size = antiDiag2size;
		antiDiag2size = antiDiag3size;
		antiDiag3size = maxCol + 1 - offset3; // double check this in original seqan

		offset1 = offset2;
		offset2 = offset3;
		offset3 = minCol - 1;

		__int16 bestCol   = 0;
		__int16 bestRow   = 0;
		__int16 bestScore = 0;

		// multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst
		__m256i antiDiagBest = _mm256_mullo_epi16 (_mm256_set1_epi16 (antiDiagNo), _mm256_set1_epi16 (gapCost));
		__m256i bestDrop     = _mm256_sub_epi16   (_mm256_set1_epi16 (best) , _mm256_set1_epi16 (scoreDropOff));
		__m256i antiMax      = _mm256_sub_epi16   (_mm256_set1_epi16 (antiDiagNo), _mm256_set1_epi16 (maxCol));

		__m256i mask1 = _mm256_cmpgt_epi16 (antiDiagBest, bestDrop); 			// if (antiDiagNo * gapCost > best - scoreDropOff) mask1 set to 1, otherwise 0
		__m256i mask2 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (offset3), _mm256_setzero_si256()); 	// if (offset3 == 0) mask2 set to 1, otherwise 0
		__m256i mask3 = _mm256_and_si256   (mask1, mask2); 						// if (antiDiagNo * gapCost > best - scoreDropOff) AND if (offset3 == 0) mask3 set to 1, otherwise 0
		__m256i mask4 = _mm256_cmpeq_epi16 (antiMax, _mm256_setzero_si256()); 	// if (antiDiagNo - maxCol == 0) mask4 set to 1, otherwise 0
		__m256i mask5 = _mm256_and_si256   (mask1, mask4); 						// if (antiDiagNo * gapCost > best - scoreDropOff) AND if (antiDiagNo - maxCol == 0) mask5 set to 1, otherwise 0

		__int16 address3[16] = {0};
		address3[0] = 1;

		__int16 address5[16] = {0};
		address5[antiDiagNo - maxCol] = 1;

		// check feasibility of this operation
		__m256i mask6 = address3;
		__m256i mask7 = address5;

		mask6      = _mm256_mullo_epi16 (mask6, mask3); // if mask3 == 0, mask6 == 0, otheriwise remain the same as declared
		antiDiag3.simd  = _mm256_blend_epi16 (antiDiag3.simd, antiDiagBest, mask6); // if mask6 == 0, antiDiag3 remain the same as before

		// antiDiag3[maxCol - offset3] = antiDiagNo * gapCost;
		mask7      = _mm256_mullo_epi16 (mask7, mask5); // if mask5 == 0, mask7 == 0, otheriwise remain the same as declared
		antiDiag3.simd  = _mm256_blend_epi16 (antiDiag3.simd, antiDiagBest, mask7); // if mask7 == 0, antiDiag3 remain the same as before

		for (__int16 col = minCol; col < maxCol; col += 16)
		{
			__int16 i3 = col - offset3;
			__int16 i2 = col - offset2;
			__int16 i1 = col - offset1;

			__int16 queryPos = col - 1; 
			__int16 dbPos = antiDiagNo - col - 1;

			// TODO: modified from: https://stackoverflow.com/questions/19494114/parallel-prefix-cumulative-sum-with-sse
			//inline __m256i scan(__m256i x)
			//{
			//	__m256 t0, t1;
			//	//shift1_AVX + add
			//	t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
			//	t1 = _mm256_permute2f128_ps(t0, t0, 41);
			//	x  = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x11));
			//	//shift2_AVX + add
			//	t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
			//	t1 = _mm256_permute2f128_ps(t0, t0, 41);
			//	x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x33));
			//	//shift3_AVX + add
			//	x = _mm256_add_ps(x,_mm256_permute2f128_ps(x, x, 41));
			//	return x;
			//}
			//__m256i offset = _mm256_setzero_si256 ();
			//__m256i x      = _mm256_loadu_epi16 (&antiDiag2[i]);
			//__m256i out = scan (x);
			//out = _mm256_add_epi16 (out, offset);
			//_mm256_storeu_epi16 (&tmp[i], out);
			////broadcast last element
			//__m256i t0 = _mm256_permute2f128_si256 (out, out, 0x11);
			//offset = _mm256_permutexvar_epi16 (t0, 0x7FFF);

			// TODO: scan operation to obtain tmp
			// calculate matrix entry (-> antiDiag3[col])
			//__m256i tmp = std::max(antiDiag2[i2-1], antiDiag2[i2]) + gapCost;

			__m256i query  = _mm256_load_epi16 (querySeg  + queryPos);  // load sixteen bases from querySeg
			__m256i target = _mm256_load_epi16 (targetSeg + targetPos); // load sixteen bases from targetSeg
			// need to add a control on target it might contain not valid data
			// if base of query == base of target
			// if mask position == 1 put score match, otherwise score mismatch
			__m256i mask8  = _mm256_cmpeq_epi16 (query, target);
			// tmp1 = // result from masking --> score(scoringScheme, querySeg[queryPos], databaseSeg[dbPos])
			tmp1 = tmp1 + antiDiag1[i1 - 1]; // scan operation again
			tmp = max(tmp, tmp1); // tmp = std::max(tmp, antiDiag1[i1 - 1] + score(scoringScheme, querySeg[queryPos], databaseSeg[dbPos]));

			__m256i mask9 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (best - scoreDropOff), tmp)
			//if (tmp < best - scoreDropOff)
			//{
			// zeros if false, ones if true : -inf if ones, tmp values if zeros in mask9
			antiDiag3 = _mm256_blend_epi16 (_mm256_set1_epi16 (undefined), tmp, mask9);
			//antiDiag3[i3] = undefined;
			//}
			// TODO: skipping control here double check 	
			// if true, this should contain antiDiagBest it shouldn't harm
			// max should be in the first position double check
		}
		antiDiagBest = _mm256_max_epi16 (antiDiagBest, tmp)
		
		//else
		//{
		//	antiDiag3[i3] = tmp;
		//	antiDiagBest = std::max(antiDiagBest, tmp);
		//}

		// seed extension wrt best score
		// TODO : not in seqan -- do this later
		//__m256i mask10 = _mm256_cmpgt_epi16(antiDiagBest + 1, best)
		//if (antiDiagBest >= best)
		//{
		//	bestCol	= length(antiDiag3) + offset3 - 2;
		//	bestRow	= antiDiagNo - bestExtensionCol;
		//	bestScore	= best;
		//}

		// antiDiagBest = *max_element(antiDiag3.begin(), antiDiag3.end());
		// best = (best > antiDiagBest) ? best : antiDiagBest;
		// ones where best is greater, otherwise zeros
		__m256i mask10 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (best), antiDiagBest);
		best = _mm256_extract_epi16 (_mm256_blend_epi16 (_mm256_set1_epi16 (best), antiDiagBest, mask10), 0);

		// calculate new minCol and minCol
		//while (minCol - offset3 < antiDiag3.size() && antiDiag3[minCol - offset3] == undefined &&
		//	   minCol - offset2 - 1 < antiDiag2.size() && antiDiag2[minCol - offset2 - 1] == undefined)
		//{
		//	++minCol;
		//}
		// these are ones if verified
		__m256i condition1 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (antiDiag3size), _mm256_set1_epi16 (minCol - offset3));
		__m256i condition3 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (antiDiag2size), _mm256_set1_epi16 (minCol - offset2 - 1));
		__m256i condition2 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag3.elem[minCol - offset3]), _mm256_set1_epi16 (undefined));
		__m256i condition4 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag2.elem[minCol - offset2 - 1]), _mm256_set1_epi16 (undefined));
		__m256i condition5 = _mm256_and_si256 (_mm256_and_si256 (condition1, condition2), _mm256_and_si256 (condition3, condition4));

		while(condition5)
		{
			// incremented by one if true, otherwise no increment
			minCol += _mm256_extract_epi16 (condition5, 0);

			// update conditions
			condition1 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (antiDiag3size), _mm256_set1_epi16 (minCol - offset3));
			condition3 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (antiDiag2size), _mm256_set1_epi16 (minCol - offset2 - 1));
			condition2 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag3.elem[minCol - offset3]), _mm256_set1_epi16 (undefined));
			condition4 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag2.elem[minCol - offset2 - 1]), _mm256_set1_epi16 (undefined));
			condition5 = _mm256_and_si256 (_mm256_and_si256 (condition1, condition2), _mm256_and_si256 (condition3, condition4));
		}

		// calculate new maxCol
		//while (maxCol - offset3 > 0 && (antiDiag3[maxCol - offset3 - 1] == undefined) &&
		//							   (antiDiag2[maxCol - offset2 - 1] == undefined))
		//{
		//	--maxCol;
		//}
		//++maxCol;
		__m256i condition6 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (maxCol - offset3), _mm256_setzero_si256 ());
		__m256i condition7 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag3.elem[maxCol - offset3 - 1]), _mm256_set1_epi16 (undefined));
		__m256i condition8 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag2.elem[maxCol - offset2 - 1]), _mm256_set1_epi16 (undefined));
		__m256i condition9 = _mm256_and_si256 (_mm256_and_si256 (condition6, condition7), condition8);

		while(condition9)
		{
			// decremented by one if true, otherwise no decrement
			maxCol -= _mm256_extract_epi16 (condition9, 0);

			// update conditions
			condition6 = _mm256_cmpgt_epi16 (_mm256_set1_epi16 (maxCol - offset3), _mm256_setzero_si256 ());
			condition7 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag3.elem[maxCol - offset3 - 1]), _mm256_set1_epi16 (undefined));
			condition8 = _mm256_cmpeq_epi16 (_mm256_set1_epi16 (antiDiag2.elem[maxCol - offset2 - 1]), _mm256_set1_epi16 (undefined));
			condition9 = _mm256_and_si256 (_mm256_and_si256 (condition6, condition7), condition8);
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
	__int16 longestExtensionCol = antiDiag3size + offset3 - 2;
	__int16 longestExtensionRow = antiDiagNo - longestExtensionCol;
	__int16 longestExtensionScore = antiDiag3.elem[longestExtensionCol - offset3];

	if (longestExtensionScore == undefined)
	{
		if (antiDiag2[antiDiag2size-2] != undefined)
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

	if (direction == EXTEND_LEFTL || direction == EXTEND_BOTHL)
	{
		// string substr (size_t pos = 0, size_t len = npos) const;
		// returns a newly constructed string object with its value initialized to a copy of a substring of this object
		std::string targetPrefix = target.substr(0, getBeginPositionH(seed));	// from read start til start seed (seed not included)
		std::string queryPrefix = query.substr(0, getBeginPositionV(seed));	// from read start til start seed (seed not included)

		scoreLeft = extendSeedLGappedXDropOneDirectionAVX2(seed, queryPrefix, targetPrefix, EXTEND_LEFTL, penalties, XDrop);
	}

	if (direction == EXTEND_RIGHTL || direction == EXTEND_BOTHL)
	{
		// Do not extend to the right if we are already at the beginning of an
		// infix or the sequence itself.
		std::string targetSuffix = target.substr(getEndPositionH(seed), target.length()); 	// from end seed until the end (seed not included)
		std::string querySuffix = query.substr(getEndPositionV(seed), query.length());		// from end seed until the end (seed not included)

		scoreRight = extendSeedLGappedXDropOneDirectionAVX2(seed, querySuffix, targetSuffix, EXTEND_RIGHTL, penalties, XDrop);
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

