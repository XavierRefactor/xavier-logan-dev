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

#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true){

	if(code != cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if(abort) exit(code);
	}
}

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



void
__global__ computeAntidiag(int *antiDiag1,
				int *antiDiag2,
				int *antiDiag3,
				int offset1,
				int offset2,
				int offset3,
				ExtensionDirectionL direction,
				int antiDiagNo,
				int gapCost,
				ScoringSchemeL scoringScheme,
				char* querySeg,
				char* databaseSeg,
				int undefined,
				int best,
				short scoreDropOff,
				int cols,
				int rows,
				int maxCol,
				int minCol)
{
	//printf(" GPU: %d\n", antiDiag1[minCol - offset1]);
	//for (int col = minCol; col < maxCol; ++col) {
	// indices on anti-diagonals
	int threadId = threadIdx.x;
	//if(threadId == 0)
		//printf(" GPU: %c\n", querySeg[0]);
	int col = threadId + minCol;
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
		int tmp = (antiDiag2[i2-1] > antiDiag2[i2]) ? antiDiag2[i2-1] : antiDiag2[i2];
		tmp+=gapCost;
		int sc = (querySeg[queryPos] ==  databaseSeg[dbPos]) ? 1 : -1;
		//printf("%d", sc);
		tmp = (tmp> antiDiag1[i1 - 1] + sc) ? tmp : antiDiag1[i1 - 1] ;//ore(scoringScheme, querySeg[queryPos], databaseSeg[dbPos]));
		//if(col<minCol+1)
			//std::cout << querySeg[queryPos]<< databaseSeg[dbPos] << "\n";
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
calcExtendedLowerDiag(int &lowerDiag,
		      int &minCol,
		      int antiDiagNo)
{
	unsigned short minRow = antiDiagNo - minCol;
	if (minCol - minRow < lowerDiag)
		lowerDiag = minCol - minRow;
}

void
calcExtendedUpperDiag(int & upperDiag,
			  int maxCol,
			  int antiDiagNo)
{
	unsigned short maxRow = antiDiagNo + 1 - maxCol;
	if (maxCol - 1 - maxRow > upperDiag)
		upperDiag = maxCol - 1 - maxRow;
}

void
swapAntiDiags(std::vector<int> &antiDiag1,
			   std::vector<int> &antiDiag2,
			   std::vector<int> &antiDiag3)
{
	//std::vector<int> temp = antiDiag1;
	swap(antiDiag1,antiDiag2);
	swap(antiDiag2,antiDiag3);
}

int
initAntiDiag3(std::vector<int> &antiDiag3,
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

void
initAntiDiags(std::vector<int> &antiDiag2,
			   std::vector<int> &antiDiag3,
			   int dropOff,
			   int gapCost,
			   int undefined)
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

int
extendSeedLGappedXDropOneDirection(
		SeedL seed,
		std::string const querySeg,
		std::string const databaseSeg,
		ExtensionDirectionL const direction,
		ScoringSchemeL &scoringScheme,
		int const scoreDropOff)
{
	//typedef typename Size<TQuerySegment>::Type int;
	//typedef typename SeedL<Simple,TConfig>::int int;
	
	//std::chrono::duration<double>  diff;
	int cols = querySeg.length()+1;
	int rows = databaseSeg.length()+1;
	if (rows == 1 || cols == 1)
		return 0;

	int minimumVal = std::numeric_limits<int>::min();//as the minimum value will always be calculated starting from an integer, we can fix it without the need to call the function
	int len = 2 * max(cols, rows); // number of antidiagonals (does not change in any implementation)
	int minErrScore = minimumVal / len; // minimal allowed error penalty
	setScoreGap(scoringScheme, max(scoreGap(scoringScheme), minErrScore));
	//std::string * tag = 0;
	//(void)tag;
	setScoreMismatch(scoringScheme, max(scoreMismatch(scoringScheme), minErrScore));

	int gapCost = scoreGap(scoringScheme);
	//std::cout<<gapCost<<std::endl;
	int undefined = minimumVal - gapCost;


	//TODO create class
	// DP matrix is calculated by anti-diagonals
	//int *antiDiag1;    //smallest anti-diagonal
	//int *antiDiag2;
	//int *antiDiag3;   //current anti-diagonal
	std::vector<int> antiDiag1;    //smallest anti-diagonal
    std::vector<int> antiDiag2;
    std::vector<int> antiDiag3;   //current anti-diagonal

	//antiDiag1 = (int *)malloc(sizeof(int)*max(querySeg.length(), databaseSeg.length()));
	//antiDiag2 = (int *)malloc(sizeof(int)*max(querySeg.length(), databaseSeg.length()));
	//antiDiag3 = (int *)malloc(sizeof(int)*max(querySeg.length(), databaseSeg.length()));

	// Indices on anti-diagonals include gap column/gap row:
	//   - decrease indices by 1 for position in query/database segment
	//   - first calculated entry is on anti-diagonal n\B0 2
	int minCol = 1;
	int maxCol = 2;

	int offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
	int offset2 = 0; //                                                       in antiDiag2
	int offset3 = 0; //                                                       in antiDiag3

	initAntiDiags(antiDiag2, antiDiag3, scoreDropOff, gapCost, undefined);
	int antiDiagNo = 1; // the currently calculated anti-diagonal

	int best = 0; // maximal score value in the DP matrix (for drop-off calculation)

	int lowerDiag = 0;
	int upperDiag = 0;
	
	while (minCol < maxCol)
	{	

		
		antiDiagNo++;
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
		//char *query, *target;
		int *a1_h =(int *)malloc(sizeof(int)*antiDiag1.size());
		int *a2_h =(int *)malloc(sizeof(int)*antiDiag2.size());
		int *a3_h =(int *)malloc(sizeof(int)*antiDiag3.size());
		char *q_h =(char *)malloc(sizeof(char)*querySeg.length());
		char *db_h=(char *)malloc(sizeof(char)*databaseSeg.length()); 
		for(int i = 0; i<antiDiag1.size(); i++){        
            a1_h[i]=antiDiag1[i];
        }
        for(int i = 0; i<antiDiag2.size(); i++){
            a2_h[i]=antiDiag2[i];
        }
        for(int i = 0; i<antiDiag3.size(); i++){
            a3_h[i]=antiDiag3[i];
        }
        for(int i = 0; i<querySeg.length(); i++){
        	q_h[i] = querySeg[i]; 
        }
        for(int i = 0; i<databaseSeg.length(); i++){
        	db_h[i] = databaseSeg[i]; 
        }
		//copy or eliminate vectors for antidiags
		int *a1_d, *a2_d, *a3_d;
		char *q_d, *db_d;
		cudaErrchk(cudaMalloc(&q_d, querySeg.length() *sizeof(char)));
		cudaErrchk(cudaMalloc(&db_d, databaseSeg.length()*sizeof(char)));
		cudaErrchk(cudaMalloc(&a1_d, antiDiag1.size()*sizeof(int)));
		cudaErrchk(cudaMalloc(&a2_d, antiDiag2.size()*sizeof(int)));
		cudaErrchk(cudaMalloc(&a3_d, antiDiag3.size()*sizeof(int)));
		cudaErrchk(cudaMemcpy(a1_d, a1_h, antiDiag1.size()*sizeof(int),cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(a2_d, a2_h, antiDiag2.size()*sizeof(int),cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(a3_d, a3_h, antiDiag3.size()*sizeof(int),cudaMemcpyHostToDevice));		
		cudaErrchk(cudaMemcpy(q_d, q_h, querySeg.length()*sizeof(char), cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(db_d, db_h, databaseSeg.length()*sizeof(char),cudaMemcpyHostToDevice));
		
		if(antiDiagNo == 2){	
		std::cout << " Before : ";
		for (int i = 0; i < antiDiag3.size(); i++) {
        		std::cout << antiDiag3.at(i) << ' ';
        	}
		}
		
		computeAntidiag <<<1,antiDiag3.size()>>> (a1_d,a2_d,a3_d,offset1,offset2,offset3,direction,antiDiagNo,gapCost,scoringScheme,q_d,db_d,undefined,best,scoreDropOff,cols,rows,maxCol,minCol);
	 	cudaDeviceSynchronize();
		cudaErrchk(cudaMemcpy(a3_h, a3_d, antiDiag3.size()*sizeof(int), cudaMemcpyDeviceToHost));
		std::copy(a3_h, a3_h + antiDiag3.size(), antiDiag3.begin());
		
		cudaErrchk(cudaFree(a1_d));
		cudaErrchk(cudaFree(a2_d));
		cudaErrchk(cudaFree(a3_d));
		cudaErrchk(cudaFree(q_d));
		cudaErrchk(cudaFree(db_d));
		free(a1_h);
		free(a2_h);
		free(a3_h);
		free(q_h);
		free(db_h);

		antiDiagBest = *max_element(antiDiag3.begin(), antiDiag3.end());
		
		if(antiDiagNo == 2){
		std::cout << " After : ";
		for (int i = 0; i < antiDiag3.size(); i++) {
            		std::cout << antiDiag3.at(i) << ' ' << a3_h[i] << ' ';
		}
		std::cout << '\n';
		}
        	//std::cout << '\n';
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

	// free(antiDiag1);
	// free(antiDiag2);
	// free(antiDiag3);
	//free(tseg);
	//free(qseg);
	// update seed
	if (longestExtensionScore != undefined)//AAAA it was !=
		updateExtendedSeedL(seed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);

	return longestExtensionScore;

}

int
extendSeedL(SeedL& seed,
			ExtensionDirectionL direction,
			std::string target,
			std::string query,
			ScoringSchemeL penalties,
			int XDrop,
			int kmer_length)
{
	//printf("extending");
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
		//printf("align left");
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

