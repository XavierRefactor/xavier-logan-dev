//
////==================================================================
//// Title:  C++ x-drop seed-and-extend alignment algorithm
//// Author: G. Guidi, A. Zeni
//// Date:   6 March 2019
////==================================================================
//
//// -----------------------------------------------------------------
//// Function extendSeed                         [GappedXDrop, noSIMD]
//// -----------------------------------------------------------------
//
//
//
#include<vector>
#include<iostream>
#include"logan.h"
#include"score.h"

// #include <bits/stdc++.h> 
enum ExtensionDirection
{
    EXTEND_NONE  = 0,
    EXTEND_LEFT  = 1,
    EXTEND_RIGHT = 2,
    EXTEND_BOTH  = 3
};

//template<typename TSeed, typename int, typename int>
inline void
updateExtendedSeed(Seed& seed,
					ExtensionDirection direction, //as there are only 4 directions we may consider even smaller data types
					int cols,
					int rows,
					int lowerDiag,
					int upperDiag)
{
	//TODO 
	//functions that return diagonal from seed
	//functions set diagonal for seed
	//which direction is which? h is q and v is t or viceversa?
	if (direction == EXTEND_LEFT)
	{
		// Set lower and upper diagonals.
		int beginDiag = seed.beginDiagonal;
		if (seed.lowerDiagonal > (beginDiag + lowerDiag))
			seed.lowerDiagonal = (beginDiag + lowerDiag);
		if (seed.upperDiagonal < (beginDiag + upperDiag))
			seed.upperDiagonal = (beginDiag + upperDiag);

		// Set new start position of seed.
		seed.beginPositionH -= rows;
		seed.beginPositionV -= cols;
	} else {  // direction == EXTEND_RIGHT
		// Set new lower and upper diagonals.
		int endDiag = seed.endDiagonal;
		if (seed.upperDiagonal < (endDiag - lowerDiag))
			seed.upperDiagonal = (endDiag - lowerDiag);
		if (seed.lowerDiagonal > (endDiag - upperDiag))
			seed.lowerDiagonal = (endDiag - upperDiag);

		// Set new end position of seed.
		seed.endPositionH += rows;
		seed.endPositionV += cols;
	}
	assert(seed.upperDiagonal >= seed.lowerDiagonal);
	assert(seed.upperDiagonal >= seed.beginDiagonal);
	assert(seed.upperDiagonal >= seed.endDiagonal);
	assert(seed.beginDiagonal >= seed.lowerDiagonal);
	assert(seed.endDiagonal >= seed.lowerDiagonal);
}

inline void
calcExtendedLowerDiag(int& lowerDiag,
					   int minCol,
					   int antiDiagNo)
{
	int minRow = antiDiagNo - minCol;
	if (minCol - minRow < lowerDiag)
		lowerDiag = minCol - minRow;
}

inline void
calcExtendedUpperDiag(int & upperDiag,
					   int maxCol,
					   int antiDiagNo)
{
	int maxRow = antiDiagNo + 1 - maxCol;
	if ((int)maxCol - 1 - (int)maxRow > upperDiag)
		upperDiag = maxCol - 1 - maxRow;
}

void
extendSeedGappedXDropOneDirectionLimitScoreMismatch(ScoringScheme & scoringScheme,
													 int minErrScore)
{
	setScoreMismatch(scoringScheme, std::max(scoreMismatch(scoringScheme), minErrScore));
}

inline void
swapAntiDiags(std::vector<int> & antiDiag1,
			   std::vector<int> & antiDiag2,
			   std::vector<int> & antiDiag3)
{
	std::vector<int> temp;
	temp = antiDiag1;
	antiDiag1 = antiDiag2;
	antiDiag2 = antiDiag3;
	antiDiag3 = temp;
}

inline int
initAntiDiag3(std::vector<int> & antiDiag3,
			   int offset,
			   int maxCol,
			   int antiDiagNo,
			   int minScore,
			   int gapCost,
			   int undefined)
{
	antiDiag3.resize(maxCol + 1 - offset);

	antiDiag3[0] = undefined;
	antiDiag3[maxCol - offset] = undefined;

	if (antiDiagNo * gapCost > minScore)
	{
		if (offset == 0) // init first column
			antiDiag3[0] = antiDiagNo * gapCost;
		if (antiDiagNo - maxCol == 0) // init first row
			antiDiag3[maxCol - offset] = antiDiagNo * gapCost;
	}
	return offset;
}

inline void
initAntiDiags(std::vector<int> & ,
			   std::vector<int> & antiDiag2,
			   std::vector<int> & antiDiag3,
			   int dropOff,
			   int gapCost,
			   int undefined)
{
	// antiDiagonals will be swaped in while loop BEFORE computation of antiDiag3 entries
	//  -> no initialization of antiDiag1 necessary

	antiDiag2.resize(1);
	antiDiag2[0] = 0;

	antiDiag3.resize(2);
	if (-gapCost > dropOff)
	{
		antiDiag3[0] = undefined;
		antiDiag3[1] = undefined;
	}
	else
	{
		antiDiag3[0] = gapCost;
		antiDiag3[1] = gapCost;
	}
}

int
extendSeedGappedXDropOneDirection(
		Seed & seed,
		std::string const & querySeg,
		std::string const & databaseSeg,
		ExtensionDirection direction,
		ScoringScheme scoringScheme,
		int scoreDropOff)
{
	//typedef typename Size<TQuerySegment>::Type int;
	//typedef typename Seed<Simple,TConfig>::int int;

	int cols = querySeg.size()+1;
	int rows = databaseSeg.size()+1;
	if (rows == 1 || cols == 1)
		return 0;

	int len = 2 * std::max(cols, rows); // number of antidiagonals
	int const minErrScore = std::numeric_limits<int>::min() / len; // minimal allowed error penalty
	setScoreGap(scoringScheme, std::max(scoreGap(scoringScheme), minErrScore));
	//std::string * tag = 0;
	//(void)tag;
	extendSeedGappedXDropOneDirectionLimitScoreMismatch(scoringScheme, minErrScore);

	int gapCost = scoreGap(scoringScheme);
	int undefined = std::numeric_limits<int>::min() - gapCost;

	// DP matrix is calculated by anti-diagonals
	std::vector<int> antiDiag1;    //smallest anti-diagonal
	std::vector<int> antiDiag2;
	std::vector<int> antiDiag3;    //current anti-diagonal

	// Indices on anti-diagonals include gap column/gap row:
	//   - decrease indices by 1 for position in query/database segment
	//   - first calculated entry is on anti-diagonal n\B0 2

	int minCol = 1;
	int maxCol = 2;

	int offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
	int offset2 = 0; //                                                       in antiDiag2
	int offset3 = 0; //                                                       in antiDiag3

	initAntiDiags(antiDiag1, antiDiag2, antiDiag3, scoreDropOff, gapCost, undefined);
	int antiDiagNo = 1; // the currently calculated anti-diagonal

	int best = 0; // maximal score value in the DP matrix (for drop-off calculation)

	int lowerDiag = 0;
	int upperDiag = 0;
	//AAAA part to parallelize???
	while (minCol < maxCol)
	{
		++antiDiagNo;
		swapAntiDiags(antiDiag1, antiDiag2, antiDiag3);
		offset1 = offset2;
		offset2 = offset3;
		offset3 = minCol-1;
		initAntiDiag3(antiDiag3, offset3, maxCol, antiDiagNo, best - scoreDropOff, gapCost, undefined);

		int antiDiagBest = antiDiagNo * gapCost;
		for (int col = minCol; col < maxCol; ++col) {
			// indices on anti-diagonals
			int i3 = col - offset3;
			int i2 = col - offset2;
			int i1 = col - offset1;

			// indices in query and database segments
			int queryPos, dbPos;
			if (direction == EXTEND_RIGHT)
			{
				queryPos = col - 1;
				dbPos = antiDiagNo - col - 1;
			}
			else // direction == EXTEND_LEFT
			{
				queryPos = cols - 1 - col;
				dbPos = rows - 1 + col - antiDiagNo;
			}

			// Calculate matrix entry (-> antiDiag3[col])
			int tmp = std::max(antiDiag2[i2-1], antiDiag2[i2]) + gapCost;
			tmp = std::max(tmp, antiDiag1[i1 - 1] + 
				score(scoringScheme, sequenceEntryForScore(scoringScheme, querySeg, queryPos),
													  sequenceEntryForScore(scoringScheme, databaseSeg, dbPos)));
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
		best = std::max(best, antiDiagBest);

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
		minCol = std::max((int)minCol, (int)antiDiagNo + 2 - (int)rows);
		// end of querySeg reached?
		maxCol = std::min(maxCol, cols);
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
	if (longestExtensionScore != undefined)
		updateExtendedSeed(seed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);
	return longestExtensionScore;
}

inline Result
extendSeed(Seed& seed,
			ExtensionDirection direction,
			std::string const& target,
			std::string const& query,
			ScoringScheme const& penalties,
			int& XDrop,
			int kmer_length)
{
	assert(scoreGapExtend(penalties) < 0); 
	assert(scoreGapOpen(penalties) < 0); 	// this is the same ad GapExtend for linear scoring scheme
	assert(scoreMismatch(penalties) < 0);
	assert(scoreMatch(penalties) > 0); 

	int scoreLeft;
	int scoreRight;
	Result scoreFinal;

	if (direction == EXTEND_LEFT || direction == EXTEND_BOTH)
	{
		// string substr (size_t pos = 0, size_t len = npos) const;
		// returns a newly constructed string object with its value initialized to a copy of a substring of this object
		std::string targetPrefix = target.substr(0, getBeginPositionH(seed));	// from read start til start seed (seed not included)
		std::string queryPrefix = query.substr(0, getBeginPositionV(seed));	// from read start til start seed (seed not included)

		scoreLeft = extendSeedGappedXDropOneDirection(seed, queryPrefix, targetPrefix, EXTEND_LEFT, penalties, XDrop);
	}

	if (direction == EXTEND_RIGHT || direction == EXTEND_BOTH)
	{
		// Do not extend to the right if we are already at the beginning of an
		// infix or the sequence itself.
		std::string targetSuffix = target.substr(getEndPositionH(seed)); 	// from end seed until the end (seed not included)
		std::string querySuffix = query.substr(getEndPositionV(seed));		// from end seed until the end (seed not included)

		scoreRight = extendSeedGappedXDropOneDirection(seed, querySuffix, targetSuffix, EXTEND_RIGHT, penalties, XDrop);
	}

	Result myalignment(kmer_length); // do not add KMER_LENGTH later

	myalignment.score = scoreLeft + scoreRight; // we have already accounted for seed match score
	myalignment.myseed = seed;	// extended begin and end of the seed

	return myalignment;
}

//AAAA TODO??? might need some attention since TAlphabet is a graph
// void
// extendSeedGappedXDropOneDirectionLimitScoreMismatch(Score & scoringScheme,
// 													 int minErrScore,
// 													 TAlphabet * /*tag*/)
// {
// 	// We cannot set a lower limit for the mismatch score since the score might be a scoring matrix such as Blosum62.
// 	// Instead, we perform a check on the matrix scores.
// #if SEQAN_ENABLE_DEBUG
// 	{
// 		for (unsigned i = 0; i < valueSize<TAlphabet>(); ++i)
// 			for (unsigned j = 0; j <= i; ++j)
// 				if(score(scoringScheme, TAlphabet(i), TAlphabet(j)) < minErrScore)
// 					printf("Mismatch score too small!, i = %u, j = %u\n");
// 	}
// #else
// 	(void)scoringScheme;
// 	(void)minErrScore;
// #endif  // #if SEQAN_ENABLE_DEBUG
// }

int main(int argc, char const *argv[])
{
	//DEBUG ONLY
	Seed myseed;
	ScoringScheme myscore;
	ExtensionDirection dir = static_cast<ExtensionDirection>(atoi(argv[1]));
	std::string target = argv[2];
	std::string query = argv[3];
	int xdrop = atoi(argv[4]);
	int kmer_length = atoi(argv[5]);
	Result r = extendSeed(myseed, dir, target, query, myscore, xdrop, kmer_length);
	std::cout << r.score << std::endl;
	return 0;
}


