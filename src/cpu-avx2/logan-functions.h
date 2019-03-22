
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
	//typedef typename Size<TQuerySegment>::Type int;
	//typedef typename SeedL<Simple,TConfig>::int int;
	
	//std::chrono::duration<double>  diff;
	unsigned short cols = querySeg.length()+1;
	unsigned short rows = databaseSeg.length()+1;
	if (rows == 1 || cols == 1)
		return 0;

	unsigned short len = 2 * std::max(cols, rows); // number of antidiagonals (does not change in any implementation)
	short const minErrScore = std::numeric_limits<short>::min() / len; // minimal allowed error penalty
	setScoreGap(scoringScheme, std::max(scoreGap(scoringScheme), minErrScore));
	//std::string * tag = 0;
	//(void)tag;
	setScoreMismatch(scoringScheme, std::max(scoreMismatch(scoringScheme), minErrScore));

	short gapCost = scoreGap(scoringScheme);
	short undefined = std::numeric_limits<short>::min() - gapCost;

	// DP matrix is calculated by anti-diagonals
	register __m256i antiDiag1; 	// 16 (vector width) 16-bit integers
	register __m256i antiDiag2; 	// 16 (vector width) 16-bit integers
	register __m256i antiDiag3; 	// 16 (vector width) 16-bit integers

	// Indices on anti-diagonals include gap column/gap row:
	//   - decrease indices by 1 for position in query/database segment
	//   - first calculated entry is on anti-diagonal n\B0 2

	unsigned short minCol = 1;
	unsigned short maxCol = 2;

	unsigned short offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
	unsigned short offset2 = 0; //                                                       in antiDiag2
	unsigned short offset3 = 0; //                                                       in antiDiag3

	//initAntiDiags(antiDiag2, antiDiag3, scoreDropOff, gapCost, undefined);
	// antiDiagonals will be swaped in while loop BEFORE computation of antiDiag3 entries
	//  -> no initialization of antiDiag1 necessary

	//antiDiag2[0] = 0;
	antiDiag2 = _mm256_setzero_si256 (); 	

	//antiDiag3.resize(2);
	if (-gapCost > dropOff)
	{
		// antiDiag3[0] = undefined;
		// antiDiag3[1] = undefined;
		antiDiag3 = _mm256_set1_epi16 (undefined); 	// broadcast 16-bit integer a to all elements of dst
	}
	else
	{
		// antiDiag3[0] = gapCost;
		// antiDiag3[1] = gapCost;
		antiDiag3 = _mm256_set1_epi16 (gapCost); 	// broadcast 16-bit integer a to all elements of dst
	}

	unsigned short antiDiagNo = 1; 	// the currently calculated anti-diagonal
	unsigned short best = 0; 		// maximal score value in the DP matrix (for drop-off calculation)

	unsigned short lowerDiag = 0;
	unsigned short upperDiag = 0;

	// This is fixed so no need to load within the loop
	__m256i undef  = _mm256_set1_epi16(undefined);
	// mask that set to zero the first element	
	short neg = -1;
	short pos =  1;
	// I need to set the first value to undefined not zero 	
	__m256i fpmask = _mm256_setr_epi16 (pos, neg, neg, neg, neg, neg, neg, neg, neg, neg, neg, neg, neg, neg, neg, neg);

	while (minCol < maxCol) // this diff cannot be greater than 16
	{

		++antiDiagNo;
		// temp = antiDiag1;
		// swap antiDiags
		// check memory alignment
		register __m256i temp;
		// __m256i _mm256_setr_epi16 eet packed 16-bit integers in dst with the supplied values in reverse order
		temp 	  = _mm256_load_epi16 (&antiDiag1);
		antiDiag1 = _mm256_load_epi16 (&antiDiag2);
		antiDiag2 = _mm256_load_epi16 (&antiDiag3);
		antiDiag3 = _mm256_load_epi16 (&temp);

		offset1 = offset2;
		offset2 = offset3;
		offset3 = minCol-1;

		// antiDiag3 = _mm256_maskload_epi16 (antiDiag3, fpmask); // antiDiag3 has the first position set to 0 (check consistency positions)
		// antiDiag3[maxCol - offset] = undefined; // which position is this?

		if (antiDiagNo * gapCost > minScore)
		{
			if (offset == 0) // init first column
			{
				__m256i val = _mm256_set1_epi16(antiDiagNo * gapCost);
				antiDiag3 = _mm256_maskload_epi16 (antiDiag3, fpmask); // antiDiag3 has the first position set to 0 (check consistency positions)
				// first element set to val (check consistency positions)
				//antiDiag3[0] = antiDiagNo * gapCost;
				// till here
			}
			if (antiDiagNo - maxCol == 0) // init first row
				antiDiag3[maxCol - offset] = antiDiagNo * gapCost;
		}

		int antiDiagBest = antiDiagNo * gapCost;

		// GGGG: loop to be unrolled by factor vector width
		for (short col = minCol; col < maxCol; ++col) {
			// indices on anti-diagonals
			
			int i3 = col - offset3;
			int i2 = col - offset2;
			int i1 = col - offset1;

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
			
			
		}

		//antiDiagBest = *max_element(antiDiag3.begin(), antiDiag3.end());
		best = (best > antiDiagBest) ? best : antiDiagBest;

		// Calculate new minCol and minCol
		// GGGG: some fancy operation with masks
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
			
		
		//index++;
	}
	//std::cout << "logan time: " <<  diff.count() <<std::endl;
	

	
	//std::cout << "cycles logan" << index << std::endl;
	
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
	if (longestExtensionScore != undefined)//AAAA it was !=
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

