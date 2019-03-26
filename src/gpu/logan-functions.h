
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

//#include<vector>
#include<iostream>
// #include<boost/array.hpp>
#include"logan.h"
#include"score.h"
//using namespace seqan;
// #include <bits/stdc++.h> 
enum ExtensionDirectionL
{
    EXTEND_NONEL  = 0,
    EXTEND_LEFTL  = 1,
    EXTEND_RIGHTL = 2,
    EXTEND_BOTHL  = 3
};

//template<typename TSeedL, typename int, typename int>
void
__device__ updateExtendedSeedL(SeedL& seed,
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

void
__device__ computeAntidiag(gpuVector<int> &antiDiag1,
				gpuVector<int> &antiDiag2,
				gpuVector<int> &antiDiag3,
				unsigned short const &offset1,
				unsigned short const &offset2,
				unsigned short const &offset3,
				ExtensionDirectionL const &direction,
				unsigned short const &antiDiagNo,
				short const &gapCost,
				ScoringSchemeL &scoringScheme,
				char *querySeg,
				char *databaseSeg,
				int const &undefined,
				int const &best,
				short const &scoreDropOff,
				unsigned short const &cols,
				unsigned short const &rows,
				unsigned short  const &maxCol,
				unsigned short  const &minCol)
{
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
		int tmp = max(antiDiag2[i2-1], antiDiag2[i2]) + gapCost;
		tmp = max(tmp, antiDiag1[i1 - 1] + score(scoringScheme, querySeg[queryPos], databaseSeg[dbPos]));
		
		
		if (tmp < best - scoreDropOff)
		{
			antiDiag3[i3] = undefined;
		}
		else
		{
			antiDiag3[i3] = tmp;
			//antiDiagBest = max(antiDiagBest, tmp);
		}
	
	
	}
}
void
__device__ calcExtendedLowerDiag(unsigned short& lowerDiag,
					   unsigned short const & minCol,
					   unsigned short const & antiDiagNo)
{
	unsigned short minRow = antiDiagNo - minCol;
	if (minCol - minRow < lowerDiag)
		lowerDiag = minCol - minRow;
}

void
__device__ calcExtendedUpperDiag(unsigned short & upperDiag,
					   unsigned short const &maxCol,
					   unsigned short const &antiDiagNo)
{
	unsigned short maxRow = antiDiagNo + 1 - maxCol;
	if (maxCol - 1 - maxRow > upperDiag)
		upperDiag = maxCol - 1 - maxRow;
}

void
__device__ swapAntiDiags(gpuVector<int> &antiDiag1,
			   gpuVector<int> &antiDiag2,
			   gpuVector<int> &antiDiag3)
{
	gpuVector<int> temp = antiDiag1;
	antiDiag1 = antiDiag2;
	antiDiag2 = antiDiag3;
	antiDiag3 = temp;
}

int
__device__ initAntiDiag3(gpuVector<int> &antiDiag3,
			   unsigned short const & offset,
			   unsigned short const & maxCol,
			   unsigned short const & antiDiagNo,
			   int const & minScore,
			   short const & gapCost,
			   int const & undefined)
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

void
__device__ initAntiDiags(gpuVector<int> &antiDiag2,
			   gpuVector<int> &antiDiag3,
			   short const& dropOff,
			   short const& gapCost,
			   int const& undefined)
{
	// antiDiagonals will be swaped in while loop BEFORE computation of antiDiag3 entries
	//  -> no initialization of antiDiag1 necessary

	antiDiag2.resize(1);

	//resize(antiDiag2, 1);
	antiDiag2[0] = 0;

	antiDiag3.resize(2);
	//resize(antiDiag3, 2);
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
//AAAA to be optmized
int
__device__ maxElem(gpuVector<int> &antiDiag,
			int const& undefined
	){
	int max=undefined;
	for(int i=0; i<antiDiag.size(); i++){
		if(antiDiag[i]>max)
			max=antiDiag[i];
	}
	return max;
}

int
__device__ extendSeedLGappedXDropOneDirection(
		SeedL & seed,
		char *querySeg,
		char *databaseSeg,
		ExtensionDirectionL const & direction,
		ScoringSchemeL &scoringScheme,
		short const &scoreDropOff,
		unsigned short querySegLength,
		unsigned short databaseSegLength
		)
{
	//typedef typename Size<TQuerySegment>::Type int;
	//typedef typename SeedL<Simple,TConfig>::int int;
	
	//std::chrono::duration<double>  diff;
	unsigned short cols = querySegLength+1;
	unsigned short rows = databaseSegLength+1;
	if (rows == 1 || cols == 1)
		return 0;
	int minimumVal = -2147483648;//as the minimum value will always be calculated starting from an integer, we can fix it without the need to call the function
	unsigned short len = 2 * max(cols, rows); // number of antidiagonals (does not change in any implementation)
	int const minErrScore = minimumVal / len; // minimal allowed error penalty
	setScoreGap(scoringScheme, max(scoreGap(scoringScheme), minErrScore));
	//std::string * tag = 0;
	//(void)tag;
	setScoreMismatch(scoringScheme, max(scoreMismatch(scoringScheme), minErrScore));

	short gapCost = scoreGap(scoringScheme);
	//std::cout<<gapCost<<std::endl;
	int undefined = minimumVal - gapCost;

	// DP matrix is calculated by anti-diagonals
	gpuVector<int> antiDiag1;    //smallest anti-diagonal
    gpuVector<int> antiDiag2;
    gpuVector<int> antiDiag3;   //current anti-diagonal

	// Indices on anti-diagonals include gap column/gap row:
	//   - decrease indices by 1 for position in query/database segment
	//   - first calculated entry is on anti-diagonal n\B0 2

	unsigned short minCol = 1;
	unsigned short maxCol = 2;

	unsigned short offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
	unsigned short offset2 = 0; //                                                       in antiDiag2
	unsigned short offset3 = 0; //                                                       in antiDiag3

	initAntiDiags(antiDiag2, antiDiag3, scoreDropOff, gapCost, undefined);
	unsigned short antiDiagNo = 1; // the currently calculated anti-diagonal

	unsigned short best = 0; // maximal score value in the DP matrix (for drop-off calculation)

	unsigned short lowerDiag = 0;
	unsigned short upperDiag = 0;
	//AAAA part to parallelize???
	//int index = 0;
	// std::chrono::duration<double>  diff;
	// auto start = std::chrono::high_resolution_clock::now();
	while (minCol < maxCol)
	{	

		
		++antiDiagNo;
		swapAntiDiags(antiDiag1, antiDiag2, antiDiag3);
		//antiDiag2 -> antiDiag1
		//antiDiag3 -> antiDiag2
		//antiDiag1 -> antiDiag3
		offset1 = offset2;
		offset2 = offset3;
		offset3 = minCol-1;
		initAntiDiag3(antiDiag3, offset3, maxCol, antiDiagNo, best - scoreDropOff, gapCost, undefined);

		int antiDiagBest = antiDiagNo * gapCost;
		//AAAA this must be parallelized
		//#pragma omp parallel for
		//auto start = std::chrono::high_resolution_clock::now();
		computeAntidiag(antiDiag1,antiDiag2,antiDiag3,offset1,offset2,offset3,direction,antiDiagNo,gapCost,scoringScheme,querySeg,databaseSeg,undefined,best,scoreDropOff,cols,rows,maxCol,minCol);
		//auto end = std::chrono::high_resolution_clock::now();
		//diff += end-start;
		
		antiDiagBest = maxElem(antiDiag3, undefined);
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
		minCol = max(minCol,(antiDiagNo + 2 - rows));
		// end of querySeg reached?
		maxCol = min(maxCol, cols);
			
		
		//index++;
	}
	//std::cout << "logan time: " <<  diff.count() <<std::endl;
	// auto end = std::chrono::high_resolution_clock::now();
	// diff += end-start;
	// std::cout << "logan: "<<diff.count() <<std::endl;


	
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
//optimize this to run on GPU
int
__device__ extendSeedL(SeedL& seed,
			ExtensionDirectionL direction,
			char* target,
			char* query,
			ScoringSchemeL & penalties,
			int const& XDrop,
			int const& kmer_length,
			int const& query_l,
			int const& target_l)
{
	// if(scoreGapExtend(penalties) >= 0)
	// {
	// //	std::cout<<"Error: Logan does not support gap extension penalty >= 0\n";
	// 	exit(1);
	// }
	// if(scoreGapOpen(penalties) >= 0)
	// {
	// //	std::cout<<"Error: Logan does not support gap opening penalty >= 0\n";
	// 	exit(1);
	// }
	//assert(scoreMismatch(penalties) < 0);
	//assert(scoreMatch(penalties) > 0); 
	assert(scoreGapOpen(penalties) == scoreGapExtend(penalties));

	int scoreLeft=0;
	int scoreRight=0;
	Result scoreFinal;
	char *queryPrefix, *querySuffix, *targetPrefix, *targetSuffix;

	queryPrefix = (char *) malloc( sizeof(char) * query_l);
	querySuffix = (char *) malloc( sizeof(char) * query_l);
	targetPrefix = (char *) malloc( sizeof(char) * target_l);
	targetSuffix = (char *) malloc( sizeof(char) * target_l);
	if (direction == EXTEND_LEFTL || direction == EXTEND_BOTHL)
	{
		// string substr (size_t pos = 0, size_t len = npos) const;
		// returns a newly constructed string object with its value initialized to a copy of a substring of this object
		memcpy(targetPrefix, target,getBeginPositionH(seed));	// from read start til start seed (seed not included)
		memcpy(queryPrefix, query, getBeginPositionV(seed));  // from read start til start seed (seed not included)

		scoreLeft = extendSeedLGappedXDropOneDirection(seed, queryPrefix, targetPrefix, EXTEND_LEFTL, penalties, XDrop,getBeginPositionV(seed),getBeginPositionH(seed));
	}

	if (direction == EXTEND_RIGHTL || direction == EXTEND_BOTHL)
	{
		// Do not extend to the right if we are already at the beginning of an
		// infix or the sequence itself.
		memcpy(targetSuffix, target+getEndPositionH(seed), target_l);  // from end seed until the end (seed not included)
		memcpy(querySuffix, query+getEndPositionV(seed), query_l);    // from end seed until the end (seed not included)

		scoreRight = extendSeedLGappedXDropOneDirection(seed, querySuffix, targetSuffix, EXTEND_RIGHTL, penalties, XDrop,query_l-getEndPositionV(seed),target_l-getEndPositionH(seed));
	}

	//Result myalignment(kmer_length); // do not add KMER_LENGTH later
	//std::cout<<"scoreLeft logan: "<<scoreLeft<<" scoreRight logan: "<<scoreRight<<std::endl;
	//myalignment.score = scoreLeft + scoreRight + kmer_length; // we have already accounted for seed match score
	int res = scoreLeft + scoreRight + kmer_length;
	//myalignment.myseed = seed;	// extended begin and end of the seed
	free(queryPrefix);
	free(querySuffix);
	free(targetPrefix);
	free(targetSuffix);

	return res;
}

// #ifdef DEBUG

// //AAAA TODO??? might need some attention since TAlphabet is a graph
// // void
// // extendSeedLGappedXDropOneDirectionLimitScoreMismatch(Score & scoringScheme,
// // 													 int minErrScore,
// // 													 TAlphabet * /*tag*/)
// // {
// // 	// We cannot set a lower limit for the mismatch score since the score might be a scoring matrix such as Blosum62.
// // 	// Instead, we perform a check on the matrix scores.
// // #if SEQAN_ENABLE_DEBUG
// // 	{
// // 		for (unsigned i = 0; i < valueSize<TAlphabet>(); ++i)
// // 			for (unsigned j = 0; j <= i; ++j)
// // 				if(score(scoringScheme, TAlphabet(i), TAlphabet(j)) < minErrScore)
// // 					printf("Mismatch score too small!, i = %u, j = %u\n");
// // 	}
// // #else
// // 	(void)scoringScheme;
// // 	(void)minErrScore;
// // #endif  // #if SEQAN_ENABLE_DEBUG
// // }

// // int main(int argc, char const *argv[])
// // {
// // 	//DEBUG ONLY
// // 	SeedL myseed;
// // 	ScoringSchemeL myscore;
// // 	ExtensionDirectionL dir = static_cast<ExtensionDirectionL>(atoi(argv[1]));
// // 	std::string target = argv[2];
// // 	std::string query = argv[3];
// // 	int xdrop = atoi(argv[4]);
// // 	int kmer_length = atoi(argv[5]);
// // 	Result r = extendSeedL(myseed, dir, target, query, myscore, xdrop, kmer_length);
// // 	std::cout << r.score << std::endl;
// // 	return 0;
// // }

// #endif

