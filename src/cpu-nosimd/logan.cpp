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
//// Limit score;  In the general case we cannot do this so we simply perform a check on the score mismatch values.
//template <typename TScoreValue, typename TScoreSpec, typename TAlphabet>
//void
//_extendSeedGappedXDropOneDirectionLimitScoreMismatch(Score<TScoreValue, TScoreSpec> & scoringScheme,
//													 TScoreValue minErrScore,
//													 TAlphabet * /*tag*/)
//{
//	// We cannot set a lower limit for the mismatch score since the score might be a scoring matrix such as Blosum62.
//	// Instead, we perform a check on the matrix scores.
//#if SEQAN_ENABLE_DEBUG
//	{
//		for (unsigned i = 0; i < valueSize<TAlphabet>(); ++i)
//			for (unsigned j = 0; j <= i; ++j)
//				SEQAN_ASSERT_GEQ_MSG(score(scoringScheme, TAlphabet(i), TAlphabet(j)), minErrScore,
//									 "Mismatch score too small!, i = %u, j = %u");
//	}
//#else
//	(void)scoringScheme;
//	(void)minErrScore;
//#endif  // #if SEQAN_ENABLE_DEBUG
//}
//
//// In the case of a SimpleScore, however, we can set this.
//template <typename TScoreValue, typename TAlphabet>
//void
//_extendSeedGappedXDropOneDirectionLimitScoreMismatch(Score<TScoreValue, Simple> & scoringScheme,
//													 TScoreValue minErrScore,
//													 TAlphabet * /*tag*/)
//{
//	setScoreMismatch(scoringScheme, std::max(scoreMismatch(scoringScheme), minErrScore));
//}
//
//template<typename TConfig, typename TQuerySegment, typename TDatabaseSegment, typename TScoreValue, typename TScoreSpec>
//TScoreValue
//_extendSeedGappedXDropOneDirection(
//		Seed<Simple, TConfig> & seed,
//		TQuerySegment const & querySeg,
//		TDatabaseSegment const & databaseSeg,
//		ExtensionDirection direction,
//		Score<TScoreValue, TScoreSpec> scoringScheme,
//		TScoreValue scoreDropOff)
//{
//	typedef typename Size<TQuerySegment>::Type int;
//	typedef typename Seed<Simple,TConfig>::int int;
//
//	int cols = length(querySeg)+1;
//	int rows = length(databaseSeg)+1;
//	if (rows == 1 || cols == 1)
//		return 0;
//
//	TScoreValue len = 2 * _max(cols, rows); // number of antidiagonals
//	TScoreValue const minErrScore = std::numeric_limits<TScoreValue>::min() / len; // minimal allowed error penalty
//	setScoreGap(scoringScheme, _max(scoreGap(scoringScheme), minErrScore));
//	typename Value<TQuerySegment>::Type * tag = 0;
//	(void)tag;
//	_extendSeedGappedXDropOneDirectionLimitScoreMismatch(scoringScheme, minErrScore, tag);
//
//	TScoreValue gapCost = scoreGap(scoringScheme);
//	TScoreValue undefined = std::numeric_limits<TScoreValue>::min() - gapCost;
//
//	// DP matrix is calculated by anti-diagonals
//	String<TScoreValue> antiDiag1;    //smallest anti-diagonal
//	String<TScoreValue> antiDiag2;
//	String<TScoreValue> antiDiag3;    //current anti-diagonal
//
//	// Indices on anti-diagonals include gap column/gap row:
//	//   - decrease indices by 1 for position in query/database segment
//	//   - first calculated entry is on anti-diagonal n\B0 2
//
//	int minCol = 1;
//	int maxCol = 2;
//
//	int offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
//	int offset2 = 0; //                                                       in antiDiag2
//	int offset3 = 0; //                                                       in antiDiag3
//
//	_initAntiDiags(antiDiag1, antiDiag2, antiDiag3, scoreDropOff, gapCost, undefined);
//	int antiDiagNo = 1; // the currently calculated anti-diagonal
//
//	TScoreValue best = 0; // maximal score value in the DP matrix (for drop-off calculation)
//
//	int lowerDiag = 0;
//	int upperDiag = 0;
//
//	while (minCol < maxCol)
//	{
//		++antiDiagNo;
//		_swapAntiDiags(antiDiag1, antiDiag2, antiDiag3);
//		offset1 = offset2;
//		offset2 = offset3;
//		offset3 = minCol-1;
//		_initAntiDiag3(antiDiag3, offset3, maxCol, antiDiagNo, best - scoreDropOff, gapCost, undefined);
//
//		TScoreValue antiDiagBest = antiDiagNo * gapCost;
//		for (int col = minCol; col < maxCol; ++col) {
//			// indices on anti-diagonals
//			int i3 = col - offset3;
//			int i2 = col - offset2;
//			int i1 = col - offset1;
//
//			// indices in query and database segments
//			int queryPos, dbPos;
//			if (direction == EXTEND_RIGHT)
//			{
//				queryPos = col - 1;
//				dbPos = antiDiagNo - col - 1;
//			}
//			else // direction == EXTEND_LEFT
//			{
//				queryPos = cols - 1 - col;
//				dbPos = rows - 1 + col - antiDiagNo;
//			}
//
//			// Calculate matrix entry (-> antiDiag3[col])
//			TScoreValue tmp = _max(antiDiag2[i2-1], antiDiag2[i2]) + gapCost;
//			tmp = _max(tmp, antiDiag1[i1 - 1] + score(scoringScheme, sequenceEntryForScore(scoringScheme, querySeg, queryPos),
//													  sequenceEntryForScore(scoringScheme, databaseSeg, dbPos)));
//			if (tmp < best - scoreDropOff)
//			{
//				antiDiag3[i3] = undefined;
//			}
//			else
//			{
//				antiDiag3[i3] = tmp;
//				antiDiagBest = _max(antiDiagBest, tmp);
//			}
//		}
//		best = _max(best, antiDiagBest);
//
//		// Calculate new minCol and minCol
//		while (minCol - offset3 < length(antiDiag3) && antiDiag3[minCol - offset3] == undefined &&
//			   minCol - offset2 - 1 < length(antiDiag2) && antiDiag2[minCol - offset2 - 1] == undefined)
//		{
//			++minCol;
//		}
//
//		// Calculate new maxCol
//		while (maxCol - offset3 > 0 && (antiDiag3[maxCol - offset3 - 1] == undefined) &&
//									   (antiDiag2[maxCol - offset2 - 1] == undefined))
//		{
//			--maxCol;
//		}
//		++maxCol;
//
//		// Calculate new lowerDiag and upperDiag of extended seed
//		_calcExtendedLowerDiag(lowerDiag, minCol, antiDiagNo);
//		_calcExtendedUpperDiag(upperDiag, maxCol - 1, antiDiagNo);
//
//		// end of databaseSeg reached?
//		minCol = _max((int)minCol, (int)antiDiagNo + 2 - (int)rows);
//		// end of querySeg reached?
//		maxCol = _min(maxCol, cols);
//	}
//
//	// find positions of longest extension
//
//	// reached ends of both segments
//	int longestExtensionCol = length(antiDiag3) + offset3 - 2;
//	int longestExtensionRow = antiDiagNo - longestExtensionCol;
//	TScoreValue longestExtensionScore = antiDiag3[longestExtensionCol - offset3];
//
//	if (longestExtensionScore == undefined)
//	{
//		if (antiDiag2[length(antiDiag2)-2] != undefined)
//		{
//			// reached end of query segment
//			longestExtensionCol = length(antiDiag2) + offset2 - 2;
//			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
//			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
//		}
//		else if (length(antiDiag2) > 2 && antiDiag2[length(antiDiag2)-3] != undefined)
//		{
//			// reached end of database segment
//			longestExtensionCol = length(antiDiag2) + offset2 - 3;
//			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
//			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
//		}
//	}
//
//	if (longestExtensionScore == undefined)
//	{
//		// general case
//		for (int i = 0; i < length(antiDiag1); ++i)
//		{
//			if (antiDiag1[i] > longestExtensionScore)
//			{
//				longestExtensionScore = antiDiag1[i];
//				longestExtensionCol = i + offset1;
//				longestExtensionRow = antiDiagNo - 2 - longestExtensionCol;
//			}
//		}
//	}
//
//	// update seed
//	if (longestExtensionScore != undefined)
//		_updateExtendedSeed(seed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);
//	return longestExtensionScore;
//}

#include"logan.h"
#include"score.h"

enum ExtensionDirection
{
    EXTEND_NONE  = 0,
    EXTEND_LEFT  = 1,
    EXTEND_RIGHT = 2,
    EXTEND_BOTH  = 3
};

inline Result
extendSeed(Seed& seed,
			short direction,
			std::string const& target,
			std::string const& query,
			ScoringScheme const& penalties,
			int& XDrop)
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
		std::string targetPrefix = target.substr(0, beginPositionH(seed));	// from read start til start seed (seed not included)
		std::string queryPrefix = query.substr(0, beginPositionV(seed));	// from read start til start seed (seed not included)

		scoreLeft = _extendSeedGappedXDropOneDirection(seed, queryPrefix, databasePrefix, EXTEND_LEFT, penalties, XDrop);
	}

	if (direction == EXTEND_RIGHT || direction == EXTEND_BOTH)
	{
		// Do not extend to the right if we are already at the beginning of an
		// infix or the sequence itself.
		std::string targetSuffix = target.substr(endPositionH(seed)); 	// from end seed until the end (seed not included)
		std::string querySuffix = query.substr(endPositionV(seed));		// from end seed until the end (seed not included)

		scoreRight = _extendSeedGappedXDropOneDirection(seed, querySuffix, databaseSuffix, EXTEND_RIGHT, penalties, XDrop);
	}

	Result myalignment(KMER_LENGTH); // do not add KMER_LENGTH later

	myalignment.score = scoreLeft + scoreRight; // we have already accounted for seed match score
	myalignment.myseed = seed;	// extended begin and end of the seed

	return myalignment;
}

inline void
initAntiDiags(std::string & ,
			   std::string & antiDiag2,
			   std::string & antiDiag3,
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

// template<typename TAntiDiag>
inline void
swapAntiDiags(std::string & antiDiag1,
			   std::string & antiDiag2,
			   std::string & antiDiag3)
{
	std::string temp;
	strcpy(temp, antiDiag1);
	strcpy(antiDiag1, antiDiag2);
	strcpy(antiDiag2, antiDiag3);
	strcpy(antiDiag3, temp);
}

inline int
initAntiDiag3(std::string & antiDiag3,
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

//template<typename TSeed, typename int, typename int>
inline void
updateExtendedSeed(Seed& seed,
					short direction, //as there are only 4 directions we may consider even smaller data types
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
		std::string beginDiag = seed.beginDiagonal;
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