// ---------------------------------------------------------------------------
// Function extendSeed                                           [GappedXDrop]
// ---------------------------------------------------------------------------

template<typename TAntiDiag, typename TDropOff, typename TScoreValue>
inline void
_initAntiDiags(TAntiDiag & ,
               TAntiDiag & antiDiag2,
               TAntiDiag & antiDiag3,
               TDropOff dropOff,
               TScoreValue gapCost,
               TScoreValue undefined)
{
    // antiDiagonals will be swaped in while loop BEFORE computation of antiDiag3 entries
    //  -> no initialization of antiDiag1 necessary

    resize(antiDiag2, 1);
    antiDiag2[0] = 0;

    resize(antiDiag3, 2);
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

template<typename TAntiDiag>
inline void
_swapAntiDiags(TAntiDiag & antiDiag1,
               TAntiDiag & antiDiag2,
               TAntiDiag & antiDiag3)
{
    TAntiDiag temp;
    move(temp, antiDiag1);
    move(antiDiag1, antiDiag2);
    move(antiDiag2, antiDiag3);
    move(antiDiag3, temp);
}

template<typename TAntiDiag, typename TSize, typename TScoreValue>
inline TSize
_initAntiDiag3(TAntiDiag & antiDiag3,
               TSize offset,
               TSize maxCol,
               TSize antiDiagNo,
               TScoreValue minScore,
               TScoreValue gapCost,
               TScoreValue undefined)
{
    resize(antiDiag3, maxCol + 1 - offset);

    antiDiag3[0] = undefined;
    antiDiag3[maxCol - offset] = undefined;

    if ((int)antiDiagNo * gapCost > minScore)
    {
        if (offset == 0) // init first column
            antiDiag3[0] = antiDiagNo * gapCost;
        if (antiDiagNo - maxCol == 0) // init first row
            antiDiag3[maxCol - offset] = antiDiagNo * gapCost;
    }
    return offset;
}

template<typename TDiagonal, typename TSize>
inline void
_calcExtendedLowerDiag(TDiagonal & lowerDiag,
                       TSize minCol,
                       TSize antiDiagNo)
{
    TSize minRow = antiDiagNo - minCol;
    if ((TDiagonal)minCol - (TDiagonal)minRow < lowerDiag)
        lowerDiag = (TDiagonal)minCol - (TDiagonal)minRow;
}

template<typename TDiagonal, typename TSize>
inline void
_calcExtendedUpperDiag(TDiagonal & upperDiag,
                       TSize maxCol,
                       TSize antiDiagNo)
{
    TSize maxRow = antiDiagNo + 1 - maxCol;
    if ((TDiagonal)maxCol - 1 - (TDiagonal)maxRow > upperDiag)
        upperDiag = maxCol - 1 - maxRow;
}

template<typename TSeed, typename TSize, typename TDiagonal>
inline void
_updateExtendedSeed(TSeed & seed,
                    ExtensionDirection direction,
                    TSize cols,
                    TSize rows,
                    TDiagonal lowerDiag,
                    TDiagonal upperDiag)
{
    if (direction == EXTEND_LEFT)
    {
        // Set lower and upper diagonals.
        TDiagonal beginDiag = beginDiagonal(seed);
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
    SEQAN_ASSERT_GEQ(upperDiagonal(seed), lowerDiagonal(seed));
    SEQAN_ASSERT_GEQ(upperDiagonal(seed), beginDiagonal(seed));
    SEQAN_ASSERT_GEQ(upperDiagonal(seed), endDiagonal(seed));
    SEQAN_ASSERT_GEQ(beginDiagonal(seed), lowerDiagonal(seed));
    SEQAN_ASSERT_GEQ(endDiagonal(seed), lowerDiagonal(seed));
}

// Limit score;  In the general case we cannot do this so we simply perform a check on the score mismatch values.
template <typename TScoreValue, typename TScoreSpec, typename TAlphabet>
void
_extendSeedGappedXDropOneDirectionLimitScoreMismatch(Score<TScoreValue, TScoreSpec> & scoringScheme,
                                                     TScoreValue minErrScore,
                                                     TAlphabet * /*tag*/)
{
    // We cannot set a lower limit for the mismatch score since the score might be a scoring matrix such as Blosum62.
    // Instead, we perform a check on the matrix scores.
#if SEQAN_ENABLE_DEBUG
    {
        for (unsigned i = 0; i < valueSize<TAlphabet>(); ++i)
            for (unsigned j = 0; j <= i; ++j)
                SEQAN_ASSERT_GEQ_MSG(score(scoringScheme, TAlphabet(i), TAlphabet(j)), minErrScore,
                                     "Mismatch score too small!, i = %u, j = %u");
    }
#else
    (void)scoringScheme;
    (void)minErrScore;
#endif  // #if SEQAN_ENABLE_DEBUG
}

// In the case of a SimpleScore, however, we can set this.
template <typename TScoreValue, typename TAlphabet>
void
_extendSeedGappedXDropOneDirectionLimitScoreMismatch(Score<TScoreValue, Simple> & scoringScheme,
                                                     TScoreValue minErrScore,
                                                     TAlphabet * /*tag*/)
{
    setScoreMismatch(scoringScheme, std::max(scoreMismatch(scoringScheme), minErrScore));
}

template<typename TConfig, typename TQuerySegment, typename TDatabaseSegment, typename TScoreValue, typename TScoreSpec>
TScoreValue
_extendSeedGappedXDropOneDirection(
        Seed<Simple, TConfig> & seed,
        TQuerySegment const & querySeg,
        TDatabaseSegment const & databaseSeg,
        ExtensionDirection direction,
        Score<TScoreValue, TScoreSpec> scoringScheme,
        TScoreValue scoreDropOff)
{
    typedef typename Size<TQuerySegment>::Type TSize;
    typedef typename Seed<Simple,TConfig>::TDiagonal TDiagonal;

    TSize cols = length(querySeg)+1;
    TSize rows = length(databaseSeg)+1;
    if (rows == 1 || cols == 1)
        return 0;

    TScoreValue len = 2 * _max(cols, rows); // number of antidiagonals
    TScoreValue const minErrScore = std::numeric_limits<TScoreValue>::min() / len; // minimal allowed error penalty
    setScoreGap(scoringScheme, _max(scoreGap(scoringScheme), minErrScore));
    typename Value<TQuerySegment>::Type * tag = 0;
    (void)tag;
    _extendSeedGappedXDropOneDirectionLimitScoreMismatch(scoringScheme, minErrScore, tag);

    TScoreValue gapCost = scoreGap(scoringScheme);
    TScoreValue undefined = std::numeric_limits<TScoreValue>::min() - gapCost;

    // DP matrix is calculated by anti-diagonals
    String<TScoreValue> antiDiag1;    //smallest anti-diagonal
    String<TScoreValue> antiDiag2;
    String<TScoreValue> antiDiag3;    //current anti-diagonal

    // Indices on anti-diagonals include gap column/gap row:
    //   - decrease indices by 1 for position in query/database segment
    //   - first calculated entry is on anti-diagonal n\B0 2

    TSize minCol = 1;
    TSize maxCol = 2;

    TSize offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
    TSize offset2 = 0; //                                                       in antiDiag2
    TSize offset3 = 0; //                                                       in antiDiag3

    _initAntiDiags(antiDiag1, antiDiag2, antiDiag3, scoreDropOff, gapCost, undefined);
    TSize antiDiagNo = 1; // the currently calculated anti-diagonal

    TScoreValue best = 0; // maximal score value in the DP matrix (for drop-off calculation)

    TDiagonal lowerDiag = 0;
    TDiagonal upperDiag = 0;

    while (minCol < maxCol)
    {
        ++antiDiagNo;
        _swapAntiDiags(antiDiag1, antiDiag2, antiDiag3);
        offset1 = offset2;
        offset2 = offset3;
        offset3 = minCol-1;
        _initAntiDiag3(antiDiag3, offset3, maxCol, antiDiagNo, best - scoreDropOff, gapCost, undefined);

        TScoreValue antiDiagBest = antiDiagNo * gapCost;
        for (TSize col = minCol; col < maxCol; ++col) {
            // indices on anti-diagonals
            TSize i3 = col - offset3;
            TSize i2 = col - offset2;
            TSize i1 = col - offset1;

            // indices in query and database segments
            TSize queryPos, dbPos;
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
            TScoreValue tmp = _max(antiDiag2[i2-1], antiDiag2[i2]) + gapCost;
            tmp = _max(tmp, antiDiag1[i1 - 1] + score(scoringScheme, sequenceEntryForScore(scoringScheme, querySeg, queryPos),
                                                      sequenceEntryForScore(scoringScheme, databaseSeg, dbPos)));
            if (tmp < best - scoreDropOff)
            {
                antiDiag3[i3] = undefined;
            }
            else
            {
                antiDiag3[i3] = tmp;
                antiDiagBest = _max(antiDiagBest, tmp);
            }
        }
        best = _max(best, antiDiagBest);

        // Calculate new minCol and minCol
        while (minCol - offset3 < length(antiDiag3) && antiDiag3[minCol - offset3] == undefined &&
               minCol - offset2 - 1 < length(antiDiag2) && antiDiag2[minCol - offset2 - 1] == undefined)
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
        _calcExtendedLowerDiag(lowerDiag, minCol, antiDiagNo);
        _calcExtendedUpperDiag(upperDiag, maxCol - 1, antiDiagNo);

        // end of databaseSeg reached?
        minCol = _max((int)minCol, (int)antiDiagNo + 2 - (int)rows);
        // end of querySeg reached?
        maxCol = _min(maxCol, cols);
    }

    // find positions of longest extension

    // reached ends of both segments
    TSize longestExtensionCol = length(antiDiag3) + offset3 - 2;
    TSize longestExtensionRow = antiDiagNo - longestExtensionCol;
    TScoreValue longestExtensionScore = antiDiag3[longestExtensionCol - offset3];

    if (longestExtensionScore == undefined)
    {
        if (antiDiag2[length(antiDiag2)-2] != undefined)
        {
            // reached end of query segment
            longestExtensionCol = length(antiDiag2) + offset2 - 2;
            longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
            longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
        }
        else if (length(antiDiag2) > 2 && antiDiag2[length(antiDiag2)-3] != undefined)
        {
            // reached end of database segment
            longestExtensionCol = length(antiDiag2) + offset2 - 3;
            longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
            longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
        }
    }

    if (longestExtensionScore == undefined)
    {
        // general case
        for (TSize i = 0; i < length(antiDiag1); ++i)
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
        _updateExtendedSeed(seed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);
    return longestExtensionScore;
}

template <typename TConfig, typename TDatabase, typename TQuery, typename TScoreValue, typename TScoreSpec>
inline int
extendSeed(Seed<Simple, TConfig> & seed,
           TDatabase const & database,
           TQuery const & query,
           ExtensionDirection direction,
           Score<TScoreValue, TScoreSpec> const & scoringScheme,
           TScoreValue scoreDropOff,
           GappedXDrop const &)
{
    // For gapped X-drop extension of Simple Seeds, we can simply
    // update the begin and end values in each dimension as well as the diagonals.

    // The algorithm only works for linear gap scores < 0, mismatch scores < 0
    // and match scores > 0.
    // TODO(holtgrew): We could introduce such check functions for score matrices.
    // TODO(holtgrew): Originally, this function only worked for simple scoring schemes, does the algorithm also work correctly for BLOSUM62? This matrix contains zeroes. Also see [10729].
    // SEQAN_ASSERT_GT(scoreMatch(scoringScheme), 0);
    // SEQAN_ASSERT_LT(scoreMismatch(scoringScheme), 0);
    SEQAN_ASSERT_LT(scoreGapOpen(scoringScheme), 0);
    SEQAN_ASSERT_LT(scoreGapExtend(scoringScheme), 0);
    SEQAN_ASSERT_EQ(scoreGapExtend(scoringScheme), scoreGapOpen(scoringScheme));
    TScoreValue longestExtensionScoreLeft;
    TScoreValue longestExtensionScoreRight;
    TScoreValue longestExtensionScore;

    if (direction == EXTEND_LEFT || direction == EXTEND_BOTH)
    {
        // Do not extend to the left if we are already at the beginning of an
        // infix or the sequence itself.

        typedef typename Prefix<TDatabase const>::Type TDatabasePrefix;
        typedef typename Prefix<TQuery const>::Type TQueryPrefix;

        TDatabasePrefix databasePrefix = prefix(database, beginPositionH(seed));
        TQueryPrefix queryPrefix = prefix(query, beginPositionV(seed));
        // TODO(holtgrew): Update _extendSeedGappedXDropOneDirection and switch query/database order.
        longestExtensionScoreLeft = _extendSeedGappedXDropOneDirection(seed, queryPrefix, databasePrefix, EXTEND_LEFT, scoringScheme, scoreDropOff);
    }

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