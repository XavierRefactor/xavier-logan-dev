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
//	typedef typename Size<TQuerySegment>::Type TSize;
//	typedef typename Seed<Simple,TConfig>::TDiagonal TDiagonal;
//
//	TSize cols = length(querySeg)+1;
//	TSize rows = length(databaseSeg)+1;
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
//	TSize minCol = 1;
//	TSize maxCol = 2;
//
//	TSize offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
//	TSize offset2 = 0; //                                                       in antiDiag2
//	TSize offset3 = 0; //                                                       in antiDiag3
//
//	_initAntiDiags(antiDiag1, antiDiag2, antiDiag3, scoreDropOff, gapCost, undefined);
//	TSize antiDiagNo = 1; // the currently calculated anti-diagonal
//
//	TScoreValue best = 0; // maximal score value in the DP matrix (for drop-off calculation)
//
//	TDiagonal lowerDiag = 0;
//	TDiagonal upperDiag = 0;
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
//		for (TSize col = minCol; col < maxCol; ++col) {
//			// indices on anti-diagonals
//			TSize i3 = col - offset3;
//			TSize i2 = col - offset2;
//			TSize i1 = col - offset1;
//
//			// indices in query and database segments
//			TSize queryPos, dbPos;
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
//	TSize longestExtensionCol = length(antiDiag3) + offset3 - 2;
//	TSize longestExtensionRow = antiDiagNo - longestExtensionCol;
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
//		for (TSize i = 0; i < length(antiDiag1); ++i)
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
enum ExtensionDirection
{
    EXTEND_NONE  = 0,
    EXTEND_LEFT  = 1,
    EXTEND_RIGHT = 2,
    EXTEND_BOTH  = 3
};

inline int
extendSeed(LOGAN::Seed& seed,
			std::string const& target,
			std::string const& query,
			Score<TScoreValue, TScoreSpec> const & scoringScheme, // TODO
			int& XDrop)
{
	// Let's start using only linear gap penalty, will introduce affine gap penalty later
	assert(penatlyI(scoringScheme) < 0); // I = indels (gaps)
	assert(penatlyX(scoringScheme) < 0); // X = mismatches
	assert(penaltyM(scoringScheme) > 0); // M = matches

	int scoreLeft;
	int scoreRight;
	int scoreFinal;

	// string substr (size_t pos = 0, size_t len = npos) const;
	// returns a newly constructed string object with its value initialized to a copy of a substring of this object
	std::string targetPrefix = target.substr(0, endPositionT(seed));
	std::string queryPrefix = query.substr(0, endPositionQ(seed));

	scoreLeft = _extendSeedGappedXDropOneDirection(seed, queryPrefix, databasePrefix, EXTEND_LEFT, scoringScheme, scoreDropOff);

	if (direction == EXTEND_RIGHT || direction == EXTEND_BOTH)
	{
		// Do not extend to the right if we are already at the beginning of an
		// infix or the sequence itself.

		typedef typename Suffix<TDatabase const>::Type TDatabaseSuffix;
		typedef typename Suffix<TQuery const>::Type TQuerySuffix;

		TDatabaseSuffix databaseSuffix = suffix(database, endPositionH(seed));
		TQuerySuffix querySuffix = suffix(query, endPositionV(seed));
		// std::cout << "database = " << database << std::endl;
		// std::cout << "database Suffix = " << databaseSuffix << std::endl;
		// std::cout << "query = " << query << std::endl;
		// std::cout << "query Suffix = " << querySuffix << std::endl;
		// TODO(holtgrew): Update _extendSeedGappedXDropOneDirection and switch query/database order.
		longestExtensionScoreRight =  _extendSeedGappedXDropOneDirection(seed, querySuffix, databaseSuffix, EXTEND_RIGHT, scoringScheme, scoreDropOff);
	}
	
	longestExtensionScore = longestExtensionScoreRight + longestExtensionScoreLeft;
	return (int)longestExtensionScore+KMER_LENGTH;
	// TODO(holtgrew): Update seed's score?!
}

inline void
_initAntiDiags(std::string & ,
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
_swapAntiDiags(std::string & antiDiag1,
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
_initAntiDiag3(std::string & antiDiag3,
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
_calcExtendedLowerDiag(std::string & lowerDiag,
					   int minCol,
					   int antiDiagNo)
{
	int minRow = antiDiagNo - minCol;
	if (minCol - minRow < lowerDiag)
		lowerDiag = minCol - minRow;
}

inline void
_calcExtendedUpperDiag(TDiagonal & upperDiag,
					   TSize maxCol,
					   TSize antiDiagNo)
{
	TSize maxRow = antiDiagNo + 1 - maxCol;
	if ((TDiagonal)maxCol - 1 - (TDiagonal)maxRow > upperDiag)
		upperDiag = maxCol - 1 - maxRow;
}

//template<typename TSeed, typename TSize, typename TDiagonal>
inline void
_updateExtendedSeed(LOGAN::Seed& seed,
					short direction, //as there are only 4 directions we may consider even smaller data types
					int cols,
					int rows,
					std::string lowerDiag,
					std::string upperDiag)
{
	//TODO 
	//functions that return diagonal from seed
	//functions set diagonal for seed
	//which direction is which? h is q and v is t or viceversa?
	if (direction == EXTEND_LEFT)
	{
		// Set lower and upper diagonals.
		std::string beginDiag = beginDiagonal(seed);
		if (lowerDiagonal(seed) > beginDiag + lowerDiag)
			setLowerDiagonal(seed, beginDiag + lowerDiag);
		if (upperDiagonal(seed) < beginDiag + upperDiag)
			setUpperDiagonal(seed, beginDiag + upperDiag);

		// Set new start position of seed.
		setBeginPositionH(seed, beginPositionH(seed) - rows);
		setBeginPositionV(seed, beginPositionV(seed) - cols);
	} else {  // direction == EXTEND_RIGHT
		// Set new lower and upper diagonals.
		TDiagonal endDiag = endDiagonal(seed);
		if (upperDiagonal(seed) < endDiag - lowerDiag)
			setUpperDiagonal(seed, endDiag - lowerDiag);
		if (lowerDiagonal(seed) > endDiag - upperDiag)
			setLowerDiagonal(seed, endDiag - upperDiag);

		// Set new end position of seed.
		setEndPositionH(seed, endPositionH(seed) + rows);
		setEndPositionV(seed, endPositionV(seed) + cols);
	}
	assert(upperDiagonal(seed) >= lowerDiagonal(seed));
	assert(upperDiagonal(seed) >= beginDiagonal(seed));
	assert(upperDiagonal(seed) >= endDiagonal(seed));
	assert(beginDiagonal(seed) >= lowerDiagonal(seed));
	assert(endDiagonal(seed) >= lowerDiagonal(seed));
}