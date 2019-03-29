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
int max_element(int *antiDiag,
		int dim
		){
	int max = -2147483648;
	for(int i = 0; i < dim; i++){
		if(antiDiag[i]>max)
			max = antiDiag[i];
	}	

	return max;
}


__global__ void
computeAntidiag(int *antiDiag1,
		int *antiDiag2,
		int *antiDiag3,
		unsigned short const offset1,
		unsigned short const offset2,
		unsigned short const offset3,
		ExtensionDirectionL const direction,
		unsigned short const antiDiagNo,
		short const gapCost,
		ScoringSchemeL scoringScheme,
		char *querySeg,
		char *databaseSeg,
		int const undefined,
		int const best,
		short const scoreDropOff,
		unsigned short const cols,
		unsigned short const rows,
		unsigned short const maxCol,
		unsigned short const minCol)
{
	//for (short col = minCol; col < maxCol; ++col) {
	// indices on anti-diagonals
	int col = threadIdx.x;
	if(col < maxCol){
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
calcExtendedLowerDiag(unsigned short& lowerDiag,
					   unsigned short const & minCol,
					   unsigned short const & antiDiagNo)
{
	unsigned short minRow = antiDiagNo - minCol;
	if (minCol - minRow < lowerDiag)
		lowerDiag = minCol - minRow;
}

void
calcExtendedUpperDiag(unsigned short & upperDiag,
					   unsigned short const &maxCol,
					   unsigned short const &antiDiagNo)
{
	unsigned short maxRow = antiDiagNo + 1 - maxCol;
	if (maxCol - 1 - maxRow > upperDiag)
		upperDiag = maxCol - 1 - maxRow;
}

void
swapAntiDiags(int *antiDiag1,
			   int *antiDiag2,
			   int *antiDiag3,
			   int &dim1,
			   int &dim2,
			   int &dim3)
{
	//std::vector<int> temp = antiDiag1;
	int temp = dim1;
	dim1 = dim2;
	dim2 = dim3;
	dim3 = temp;
	int *tempArr = antiDiag1;
	antiDiag1 = antiDiag2;
	antiDiag2 = antiDiag3;
	antiDiag3 = tempArr;
}

int
initAntiDiag3(int *antiDiag3,
			   unsigned short const & offset,
			   unsigned short const & maxCol,
			   unsigned short const & antiDiagNo,
			   int const & minScore,
			   short const & gapCost,
			   int const & undefined,
			   int &dim3)
{
	dim3 = maxCol + 1 - offset;

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
initAntiDiags(int *antiDiag2,
			   int *antiDiag3,
			   short const& dropOff,
			   short const& gapCost,
			   int const& undefined,		
			   int &dim1,
			   int &dim2,
			   int &dim3)
{
	// antiDiagonals will be swaped in while loop BEFORE computation of antiDiag3 entries
	//  -> no initialization of antiDiag1 necessary

	dim2 = 1;

	//resize(antiDiag2, 1);
	antiDiag2[0] = 0;

	dim3 = 2;
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

int
extendSeedLGappedXDropOneDirection(
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


	//TODO create class
	// DP matrix is calculated by anti-diagonals
	int *antiDiag1;    //smallest anti-diagonal
	int *antiDiag2;
	int *antiDiag3;   //current anti-diagonal
	int antiDiag1size=0, antiDiag2size=0, antiDiag3size=0;

	antiDiag1 = (int *)malloc(sizeof(32)*max(querySeg.length(), databaseSeg.length()));
	antiDiag2 = (int *)malloc(sizeof(32)*max(querySeg.length(), databaseSeg.length()));
	antiDiag3 = (int *)malloc(sizeof(32)*max(querySeg.length(), databaseSeg.length()));

	// Indices on anti-diagonals include gap column/gap row:
	//   - decrease indices by 1 for position in query/database segment
	//   - first calculated entry is on anti-diagonal n\B0 2

	unsigned short minCol = 1;
	unsigned short maxCol = 2;

	unsigned short offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
	unsigned short offset2 = 0; //                                                       in antiDiag2
	unsigned short offset3 = 0; //                                                       in antiDiag3

	initAntiDiags(antiDiag2, antiDiag3, scoreDropOff, gapCost, undefined, antiDiag1size, antiDiag2size, antiDiag3size);
	unsigned short antiDiagNo = 1; // the currently calculated anti-diagonal

	unsigned short best = 0; // maximal score value in the DP matrix (for drop-off calculation)

	unsigned short lowerDiag = 0;
	unsigned short upperDiag = 0;
	//AAAA part to parallelize???
	//int index = 0;
	// std::chrono::duration<double>  diff;
	// auto start = std::chrono::high_resolution_clock::now();
	char *query, *qseg, *tseg, *target;
	cudaMalloc(&query, querySeg.length()*sizeof(char));
        cudaMalloc(&target, databaseSeg.length()*sizeof(char));
	qseg = (char *)malloc(querySeg.length()*sizeof(char));
	tseg = (char *)malloc(databaseSeg.length()*sizeof(char));  
	cudaMemcpy(query, qseg,querySeg.length(),cudaMemcpyHostToDevice);
        cudaMemcpy(target, tseg, databaseSeg.length(),cudaMemcpyHostToDevice);
	
	while (minCol < maxCol)
	{	

		
		++antiDiagNo;
		swapAntiDiags(antiDiag1, antiDiag2, antiDiag3, antiDiag1size, antiDiag2size, antiDiag3size);
		//antiDiag2 -> antiDiag1
		//antiDiag3 -> antiDiag2
		//antiDiag1 -> antiDiag3
		offset1 = offset2;
		offset2 = offset3;
		offset3 = minCol-1;
		initAntiDiag3(antiDiag3, offset3, maxCol, antiDiagNo, best - scoreDropOff, gapCost, undefined, antiDiag3size);

		int antiDiagBest = antiDiagNo * gapCost;
		//AAAA this must be parallelized

		int *a1, *a2, *a3;
		//char *query, *target;

		cudaMalloc(&a1, antiDiag1size*sizeof(int));
		cudaMalloc(&a2, antiDiag2size*sizeof(int));
		cudaMalloc(&a3, antiDiag3size*sizeof(int));
			
		//cudaMalloc(&query, querySeg.length());
		//cudaMalloc(&target, databaseSeg.length());	
		
		cudaMemcpy(a1, antiDiag1, antiDiag1size,cudaMemcpyHostToDevice);
		cudaMemcpy(a2, antiDiag2, antiDiag2size,cudaMemcpyHostToDevice);
		cudaMemcpy(a3, antiDiag3, antiDiag3size,cudaMemcpyHostToDevice);		
		//cudaMemcpy(query, querySeg.c_str(),querySeg.length(),cudaMemcpyHostToDevice);
		//cudaMemcpy(target, databaseSeg.c_str(), databaseSeg.length(),cudaMemcpyHostToDevice);
		computeAntidiag <<<1,1024>>> (a1,a2,a3,offset1,offset2,offset3,direction,antiDiagNo,gapCost,scoringScheme,query,target,undefined,best,scoreDropOff,cols,rows,maxCol,minCol);
	 	cudaDeviceSynchronize();	
		cudaMemcpy(antiDiag1, a1, antiDiag1size,cudaMemcpyDeviceToHost);
		cudaMemcpy(antiDiag2, a2, antiDiag2size,cudaMemcpyDeviceToHost);		
		cudaMemcpy(antiDiag3, a3, antiDiag3size,cudaMemcpyDeviceToHost);

		antiDiagBest = max_element(antiDiag3, antiDiag3size);
		best = (best > antiDiagBest) ? best : antiDiagBest;
	
		cudaFree(a1);
		cudaFree(a2);
		cudaFree(a3);
		cudaFree(query);
		cudaFree(target);
		// Calculate new minCol and minCol
		while (minCol - offset3 < antiDiag3size && antiDiag3[minCol - offset3] == undefined &&
			   minCol - offset2 - 1 < antiDiag2size && antiDiag2[minCol - offset2 - 1] == undefined)
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
	int longestExtensionCol = antiDiag3size + offset3 - 2;
	int longestExtensionRow = antiDiagNo - longestExtensionCol;
	int longestExtensionScore = antiDiag3[longestExtensionCol - offset3];

	if (longestExtensionScore == undefined)
	{
		if (antiDiag2[antiDiag2size-2] != undefined)
		{
			// reached end of query segment
			longestExtensionCol = antiDiag2size + offset2 - 2;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
		}
		else if (antiDiag2size > 2 && antiDiag2[antiDiag2size - 3] != undefined)
		{
			// reached end of database segment
			longestExtensionCol = antiDiag2size + offset2 - 3;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
		}
	}

	if (longestExtensionScore == undefined)
	{
		// general case
		for (int i = 0; i < antiDiag1size; ++i)
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

int
extendSeedL(SeedL& seed,
			ExtensionDirectionL direction,
			std::string const& target,
			std::string const& query,
			ScoringSchemeL & penalties,
			int const& XDrop,
			int const& kmer_length)
{
	if(scoreGapExtend(penalties) >= 0)
	{
		std::cout<<"Error: Logan does not support gap extension penalty >= 0\n";
		exit(1);
	}
	if(scoreGapOpen(penalties) >= 0)
	{
		std::cout<<"Error: Logan does not support gap opening penalty >= 0\n";
		exit(1);
	}
	//assert(scoreMismatch(penalties) < 0);
	//assert(scoreMatch(penalties) > 0); 
	//assert(scoreGapOpen(penalties) == scoreGapExtend(penalties));

	int scoreLeft=0;
	int scoreRight=0;
	Result scoreFinal;

	if (direction == EXTEND_LEFTL || direction == EXTEND_BOTHL)
	{
		// string substr (size_t pos = 0, size_t len = npos) const;
		// returns a newly constructed string object with its value initialized to a copy of a substring of this object
		std::string targetPrefix = target.substr(0, getBeginPositionH(seed));	// from read start til start seed (seed not included)
		std::string queryPrefix = query.substr(0, getBeginPositionV(seed));	// from read start til start seed (seed not included)

		scoreLeft = extendSeedLGappedXDropOneDirection(seed, queryPrefix, targetPrefix, EXTEND_LEFTL, penalties, XDrop);
	}

	if (direction == EXTEND_RIGHTL || direction == EXTEND_BOTHL)
	{
		// Do not extend to the right if we are already at the beginning of an
		// infix or the sequence itself.
		std::string targetSuffix = target.substr(getEndPositionH(seed), target.length()); 	// from end seed until the end (seed not included)
		std::string querySuffix = query.substr(getEndPositionV(seed), query.length());		// from end seed until the end (seed not included)

		scoreRight = extendSeedLGappedXDropOneDirection(seed, querySuffix, targetSuffix, EXTEND_RIGHTL, penalties, XDrop);
	}

	//Result myalignment(kmer_length); // do not add KMER_LENGTH later
	//std::cout<<"scoreLeft logan: "<<scoreLeft<<" scoreRight logan: "<<scoreRight<<std::endl;
	//myalignment.score = scoreLeft + scoreRight + kmer_length; // we have already accounted for seed match score
	int res = scoreLeft + scoreRight + kmer_length;
	//myalignment.myseed = seed;	// extended begin and end of the seed

	return res;
}

