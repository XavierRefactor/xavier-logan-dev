
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

int
extendSeedLGappedXDropOneDirectionAVX2(
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

	register __m256i antiDiag1; 	// 16 (vector width) 16-bit integers
	register __m256i antiDiag2; 	// 16 (vector width) 16-bit integers
	register __m256i antiDiag3; 	// 16 (vector width) 16-bit integers

	__m256i minCol = _mm256_set1_epi16 (1);
	__m256i maxCol = _mm256_set1_epi16 (2);

	__m256i offset1 = _mm256_setzero_si256(); // number of leading columns that need not be calculated in antiDiag1
	__m256i offset2 = _mm256_setzero_si256(); //                                                       in antiDiag2
	__m256i offset3 = _mm256_setzero_si256(); //                                                       in antiDiag3

	antiDiag2 = _mm256_setzero_si256 (); 			// initialize vector with zeros

	//antiDiag3.resize(2);
	if (-gapCost > dropOff) // initialize vector with -inf
	{
		antiDiag3 = _mm256_set1_epi16 (undefined); 	// broadcast 16-bit integer a to all elements of dst
	}
	else // initialize vector with gapCost
	{
		antiDiag3 = _mm256_set1_epi16 (gapCost); 	// broadcast 16-bit integer a to all elements of dst
	}

	__m256i antiDiagNo = _mm256_set1_epi16 (1); 	// the currently calculated anti-diagonal
	__m256i best = _mm256_setzero_si256(); 			// maximal score value in the DP matrix (for drop-off calculation)

	__m256i lowerDiag = _mm256_setzero_si256();
	__m256i upperDiag = _mm256_setzero_si256();

	while (minCol < maxCol) // this diff cannot be greater than 16
	{
		// data must be aligned when loading to and storing to avoid severe performance penalties
		++antiDiagNo;
		// swap antiDiags
		antiDiag1 = _mm256_load_epi16 (&antiDiag2);
		antiDiag2 = _mm256_load_epi16 (&antiDiag3);
		antiDiag3 = _mm256_set1_epi16 (undefined); 	// init to -inf at each iteration

		offset1 = offset2;
		offset2 = offset3;
		offset3 = minCol-1;

		int bestExtensionCol = 0;
		int bestExtensionRow = 0;
		int bestExtensionScore = 0;

		// multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst
		__m256i antiDiagBest = _mm256_mullo_epi16 (antiDiagNo, gapCost);
		__m256i bestDrop   = _mm256_sub_epi16 (best, scoreDropOff);
		__m256i antiMax    = _mm256_sub_epi16 (antiDiagNo, maxCol);

		// if (antiDiagNo * gapCost > best - scoreDropOff) mask1 set to 1, otherwise 0
		__m256i mask1 	 = _mm256_cmpgt_epi16 (antiDiagBest, bestDrop);
		// if (offset3 == 0) mask2 set to 1, otherwise 0
		__m256i mask2 	 = _mm256_cmpeq_epi16 (offset3, _mm256_setzero_si256());
		// if (antiDiagNo * gapCost > best - scoreDropOff) AND if (offset3 == 0) mask3 set to 1, otherwise 0
		__m256i mask3 	 = _mm256_and_si256 (mask1, mask2);
		// if (antiDiagNo - maxCol == 0) mask4 set to 1, otherwise 0
		__m256i mask4 	 = _mm256_cmpeq_epi16 (antiMax, _mm256_setzero_si256());
		// if (antiDiagNo * gapCost > best - scoreDropOff) AND if (antiDiagNo - maxCol == 0) mask5 set to 1, otherwise 0
		__m256i mask5 	 = _mm256_and_si256 (mask1, mask4);

		__m256i mask6  = _mm256_setr_epi16 (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		mask6      = _mm256_mullo_epi16 (mask6, mask3); // if mask3 == 0, 1st lane == 0, otheriwise remain the same as declared
		antiDiag3  = _mm256_blend_epi16 (antiDiag3, antiDiagBest, mask6); // if 1stlane == 0, antiDiag3 remain the same as before 	

		// how I compute this at runtime?
		// antiDiag3[maxCol - offset3] = antiDiagNo * gapCost;
		__m256i mask7 	= _mm256_cmpeq_epi16 (antiMax, antiMax);
		mask7      = _mm256_mullo_epi16 (mask7, mask5); // if mask5 == 0, mask7 == 0, otheriwise remain the same as declared
		antiDiag3  = _mm256_blend_epi16 (antiDiag3, antiDiagBest, mask7); // if mask7s == 0, antiDiag3 remain the same as before 	

		//  unroll here
		//  for (short col = minCol; col < maxCol; col += 16) {
		//	// indices on anti-diagonals

		__m256i i3 = col - offset3;
		__m256i i2 = col - offset2;
		__m256i i1 = col - offset1;

		// indices in query and database segments
		int queryPos, dbPos;
		if (direction == EXTEND_RIGHTL)
		{
			queryPos = col - 1;
			dbPos = antiDiagNo - col - 1;
		}
		else // direction == EXTEND_LEFTL
		{
			queryPos = cols - 1 - col;
			dbPos = rows - 1 + col - antiDiagNo;
		}

		// Calculate matrix entry (-> antiDiag3[col])
		int tmp = std::max(antiDiag2[i2-1], antiDiag2[i2]) + gapCost;
		tmp = std::max(tmp, antiDiag1[i1 - 1] + score(scoringScheme, querySeg[queryPos], databaseSeg[dbPos]));
		
		
		if (tmp < best - scoreDropOff)
		{
			antiDiag3[i3] = undefined;
		}
		else
		{
			antiDiag3[i3] = tmp;
			antiDiagBest = std::max(antiDiagBest, tmp);
		}
		//}

		// seed extension wrt best score
		if (antiDiagBest >= best)
		{
			bestExtensionCol	= length(antiDiag3) + offset3 - 2;
			bestExtensionRow	= antiDiagNo - bestExtensionCol;
			bestExtensionScore	= best;
		}

		//antiDiagBest = *max_element(antiDiag3.begin(), antiDiag3.end());
		best = (best > antiDiagBest) ? best : antiDiagBest;

		// Calculate new minCol and minCol
		while (minCol - offset3 < antiDiag3.size() && antiDiag3[minCol - offset3] == undefined &&
			   minCol - offset2 - 1 < antiDiag2.size() && antiDiag2[minCol - offset2 - 1] == undefined)
		{
			++minCol;
		}

		// Calculate new maxCol
		while (maxCol - offset3 > 0 && (antiDiag3[maxCol - offset3 - 1] == undefined) &&
									   (antiDiag2[maxCol - offset2 - 1] == undefined))
		{
			--maxCol;
		}
		++maxCol;

		// Calculate new lowerDiag and upperDiag of extended seed
		calcExtendedLowerDiag(lowerDiag, minCol, antiDiagNo);
		calcExtendedUpperDiag(upperDiag, maxCol - 1, antiDiagNo);

		// end of databaseSeg reached?
		minCol = (minCol > (antiDiagNo + 2 - rows)) ? minCol : (antiDiagNo + 2 - rows);
		// end of querySeg reached?
		maxCol = (maxCol < cols) ? maxCol : cols;
	}

	// find positions of longest extension
	// reached ends of both segments
	int longestExtensionCol = antiDiag3.size() + offset3 - 2;
	int longestExtensionRow = antiDiagNo - longestExtensionCol;
	int longestExtensionScore = antiDiag3[longestExtensionCol - offset3];
	if (longestExtensionScore == undefined)
	{
		if (antiDiag2[antiDiag2.size()-2] != undefined)
		{
			// reached end of query segment
			longestExtensionCol = antiDiag2.size() + offset2 - 2;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
		}
		else if (antiDiag2.size() > 2 && antiDiag2[antiDiag2.size()-3] != undefined)
		{
			// reached end of database segment
			longestExtensionCol = antiDiag2.size() + offset2 - 3;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
		}
	}
	if (longestExtensionScore == undefined)
	{
		// general case
		for (int i = 0; i < antiDiag1.size(); ++i)
		{
			if (antiDiag1[i] > longestExtensionScore)
			{
				longestExtensionScore = antiDiag1[i];
				longestExtensionCol = i + offset1;
				longestExtensionRow = antiDiagNo - 2 - longestExtensionCol;
			}
		}
	}

	// update seed
	if (bestExtensionScore != undefined)
		updateExtendedSeedL(seed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);

	return bestExtensionScore;

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

