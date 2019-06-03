//==================================================================
// Title:  C++ x-drop seed-and-extend alignment algorithm
// Author: G. Guidi, A. Zeni
// Date:   6 March 2019
//==================================================================

// -----------------------------------------------------------------
// Function extendSeedL                         [GappedXDrop, Cuda]
// -----------------------------------------------------------------

//remove asserts to speedup 

//#define DEBUG

#include<vector>
#include<iostream>
#include<chrono>
//#include <cub/block/block_load.cuh>
//#include <cub/block/block_store.cuh>
//#include <cub/block/block_reduce.cuh>
//#include <cub/cub.cuh>
//#include<boost/array.hpp>
#include"logan.h"
#include"score.h"

// using namespace cub;

#define N_THREADS 1024
#define N_BLOCKS 29000
#define MIN -32768
#define BYTES_INT 4
#define XDROP 21
// #define N_STREAMS 60
#define MAX_SIZE_ANTIDIAG 1024

//trying to see if the scoring scheme is a bottleneck in some way
#define MATCH     1
#define MISMATCH -1
#define GAP_EXT  -1
#define GAP_OPEN -1
#define UNDEF -32767
#define NOW std::chrono::high_resolution_clock::now()

using namespace std;
using namespace chrono;

#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true){

	if(code != cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if(abort) exit(code);
	}
}

using namespace std;

enum ExtensionDirectionL
{
    EXTEND_NONEL  = 0,
    EXTEND_LEFTL  = 1,
    EXTEND_RIGHTL = 2,
    EXTEND_BOTHL  = 3
};

__device__ inline int array_max(short *array,
				int &dim,
				int &minCol,
				int &ant_offset)
{
	//printf("%d\n", dim1);
	__shared__ short localArray[N_THREADS/2];
	unsigned int tid = threadIdx.x;
	int half = dim>>1;
	if(tid < half){
		localArray[tid] = max_logan(array[tid+minCol-ant_offset],array[tid+minCol+half-ant_offset]);
	}
	//__syncthreads();		
	for(int offset = dim/4; offset > 0; offset>>=1){
		if(tid < offset) localArray[tid] = max_logan(localArray[tid],localArray[tid+offset]);
	//	__syncthreads();
	}
	//__syncthreads();
	return localArray[0];
	
}


__device__ int simple_max(short *antidiag,
			  int &dim,
			  int &offset){
	int max = antidiag[0];
	for(int i = 1; i < dim; i++){
		if(antidiag[i]>max)
			max=antidiag[i];
	}
	return max;

}

//template<typename TSeedL, typename int, typename int>
__device__ inline void updateExtendedSeedL(SeedL& seed,
					ExtensionDirectionL direction, //as there are only 4 directions we may consider even smaller data types
					int &cols,
					int &rows,
					int &lowerDiag,
					int &upperDiag)
{
	if (direction == EXTEND_LEFTL)
	{
		int beginDiag = seed.beginDiagonal;
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
		int endDiag = seed.endDiagonal;
		if (getUpperDiagonal(seed) < endDiag - lowerDiag)
			setUpperDiagonal(seed, (endDiag - lowerDiag));
		if (getLowerDiagonal(seed) > (endDiag - upperDiag))
			setLowerDiagonal(seed, endDiag - upperDiag);

		// Set new end position of seed.
		setEndPositionH(seed, getEndPositionH(seed) + rows);
		setEndPositionV(seed, getEndPositionV(seed) + cols);
		
	}
}

__device__ inline void computeAntidiag(short *antiDiag1,
									short *antiDiag2,
									short *antiDiag3,
									char* querySeg,
									char* databaseSeg,
									int &best,
									int &scoreDropOff,
									int &cols,
									int &rows,
									int &minCol,
									int &maxCol,
									int &antiDiagNo,
									int &offset1,
									int &offset2,
									ExtensionDirectionL direction
									){
	int tid = threadIdx.x;
	int col = tid + minCol;
	int queryPos, dbPos;
	
	if(direction == EXTEND_LEFTL){
		queryPos = cols - 1 - col;
		dbPos = rows - 1 + col - antiDiagNo;
	}else{//EXTEND RIGHT
		queryPos = col - 1;
		dbPos = antiDiagNo - col - 1;
	}
	
	if(col < maxCol){
		//printf("%d\n",antiDiagNo);
		int tmp = max_logan(antiDiag2[col-offset2],antiDiag2[col-offset2-1]) + GAP_EXT;
		//printf("%d\n",tid);
		int score = (querySeg[queryPos] == databaseSeg[dbPos]) ? MATCH : MISMATCH;
		tmp = max_logan(antiDiag1[col-offset1-1]+score,tmp);
		antiDiag3[tid+1] = (tmp < best - scoreDropOff) ? UNDEF : tmp;
	}
}

__device__ inline void calcExtendedLowerDiag(int *lowerDiag,
		      int const &minCol,
		      int const &antiDiagNo)
{
	int minRow = antiDiagNo - minCol;
	if (minCol - minRow < *lowerDiag)
		*lowerDiag = minCol - minRow;
}

__device__ inline void calcExtendedUpperDiag(int *upperDiag,
			  int const &maxCol,
			  int const &antiDiagNo)
{
	int maxRow = antiDiagNo + 1 - maxCol;
	if (maxCol - 1 - maxRow > *upperDiag)
		*upperDiag = maxCol - 1 - maxRow;
}

__device__ inline void swapAntiDiags(short *antiDiag1,
	   				short *antiDiag2,
	   				short *antiDiag3,
	   				int *a1size,
	   				int *a2size,
	   				int *a3size,
	   				int *offset1,
	   				int *offset2,
	   				int *offset3,
	   				int *minCol)
{
	
	
	short *t = antiDiag1;
	antiDiag1 = antiDiag2;
	antiDiag2 = antiDiag3;
	antiDiag3 = t;
	int t_l = *a1size;
	*a1size = *a2size;
	*a2size = *a3size;
	*a3size = t_l;
	*offset1 = *offset2;
	*offset2 = *offset3;
	*offset3 = *minCol-1;

}

__device__ inline void initAntiDiag3(short *antiDiag3,
							int *a3size,
			   				int const &offset,
			   				int const &maxCol,
			   				int const &antiDiagNo,
			   				int const &minScore,
			   				int const &gapCost,
			   				int const &undefined)
{
	
	*a3size = maxCol + 1 - offset;
	antiDiag3[0] = undefined;
	antiDiag3[maxCol - offset] = undefined;

	if (antiDiagNo * gapCost > minScore)
	{
		if (offset == 0) // init first column
			antiDiag3[0] = antiDiagNo * gapCost;
		if (antiDiagNo - maxCol == 0) // init first row
			antiDiag3[maxCol - offset] = antiDiagNo * gapCost;
	}
	//return offset;
}

__device__ inline void initAntiDiags(
			   short *antiDiag1,
			   short *antiDiag2,
			   short *antiDiag3,
			   int *a2size,
			   int *a3size,
			   int const &dropOff,
			   int const &gapCost,
			   int const &undefined)
{
	// antiDiagonals will be swaped in while loop BEFORE computation of antiDiag3 entries
	//  -> no initialization of antiDiag1 necessary
	//int tid = threadIdx.x;
	//if(tid<N_THREADS){
	
	//	antiDiag1[tid]=UNDEF;			
	//	antiDiag2[tid]=UNDEF;
	//	antiDiag3[tid]=UNDEF;
		

	//}
	//__syncthreads();	
	
	//antiDiag2.resize(1);
	*a2size = 1;

	//resize(antiDiag2, 1);
	antiDiag2[0] = 0;

	//antiDiag3.resize(2);
	*a3size = 2;

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

__global__ void extendSeedLGappedXDropOneDirection(
		SeedL *seed,
		char *querySegArray,
		char *databaseSegArray,
		ExtensionDirectionL direction,
		ScoringSchemeL *scoringScheme,
		int scoreDropOff,
		int *res,
		// int *antiDiag1,
		// int *antiDiag2,
		// int *antiDiag3,
		int *qL,
		int *dbL,
		int *offsetQuery,
		int *offsetTarget)
{
	//typedef typename Size<TQuerySegment>::Type int;
	//typedef typename SeedL<Simple,TConfig>::int int;
	int myId = blockIdx.x;
	int myTId = threadIdx.x;
	char *querySeg;
	char *databaseSeg;

	if(myId==0){
		querySeg = querySegArray;
		databaseSeg = databaseSegArray;
	}
	else{
		querySeg = querySegArray + offsetQuery[myId-1];
		databaseSeg = databaseSegArray + offsetTarget[myId-1];
	}

	__shared__ short antiDiag1p[N_THREADS];
	__shared__ short antiDiag2p[N_THREADS];
	__shared__ short antiDiag3p[N_THREADS];
	short* antiDiag1 = (short*) antiDiag1p;
	short* antiDiag2 = (short*) antiDiag2p;
	short* antiDiag3 = (short*) antiDiag3p;
	
	//dimension of the antidiagonals
	int a1size = 0, a2size = 0, a3size = 0;
	
	int cols = qL[myId]+1;//querySeg.length()+1;//should be enough even if we allocate just one time the string
	int rows = dbL[myId]+1;//databaseSeg.length()+1;//
	if (rows == 1 || cols == 1)
		return;

	//int gapCost = scoreGap(scoringScheme);
	//printf("%d\n", gapCost);
	int undefined = UNDEF;
	int minCol = 1;
	int maxCol = 2;

	int offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
	int offset2 = 0; //                                                       in antiDiag2
	int offset3 = 0; //                                                       in antiDiag3

	initAntiDiags(antiDiag1,antiDiag2, antiDiag3, &a2size, &a3size, scoreDropOff, GAP_EXT, undefined);
	int antiDiagNo = 1; // the currently calculated anti-diagonal

	int best = 0; // maximal score value in the DP matrix (for drop-off calculation)

	int lowerDiag = 0;
	int upperDiag = 0;

	while (minCol < maxCol)
	{	

		
		++antiDiagNo;

 		//antidiagswap
 		//antiDiag2 -> antiDiag1
		//antiDiag3 -> antiDiag2
		//antiDiag1 -> antiDiag3
		short *t = antiDiag1;
		antiDiag1 = antiDiag2;
		antiDiag2 = antiDiag3;
		antiDiag3 = t;
		int t_l = a1size;
		a1size = a2size;
		a2size = a3size;
		a3size = t_l;
		offset1 = offset2;
		offset2 = offset3;
		offset3 = minCol-1;
		//swapAntiDiags(antiDiag1, antiDiag2, antiDiag3, &a1size, &a2size, &a3size, &offset1, &offset2, &offset3, &min);

		initAntiDiag3(antiDiag3, &a3size, offset3, maxCol, antiDiagNo, best - scoreDropOff, GAP_EXT, undefined);
		
		//computeAntidiagLeft(antiDiag1,antiDiag2,antiDiag3,offset1,offset2,offset3,antiDiagNo,GAP_EXT,scoringScheme,querySeg,databaseSeg,undefined,best,scoreDropOff,cols,rows,maxCol,minCol);
		computeAntidiag(antiDiag1, antiDiag2, antiDiag3, querySeg, databaseSeg, best, scoreDropOff, cols, rows, minCol, maxCol, antiDiagNo, offset1, offset2, direction);	 	
		__syncthreads();	
		//int antiDiagBest = simple_max(antiDiag3, a3size, offset3);
	
		int antiDiagBest = array_max(antiDiag3, a3size, minCol, offset3);
	
		//original min and max col update
		best = (best > antiDiagBest) ? best : antiDiagBest;
		
		while (minCol - offset3 < a3size && antiDiag3[minCol - offset3] == undefined &&
			   minCol - offset2 - 1 < a2size && antiDiag2[minCol - offset2 - 1] == undefined)
		{
			++minCol;
		}
		
		while (maxCol - offset3 > 0 && (antiDiag3[maxCol - offset3 - 1] == undefined) &&
									   (antiDiag2[maxCol - offset2 - 1] == undefined))
	 	{
			--maxCol;
		}
		++maxCol;
		
		// Calculate new lowerDiag and upperDiag of extended seed
		calcExtendedLowerDiag(&lowerDiag, minCol, antiDiagNo);
		calcExtendedUpperDiag(&upperDiag, maxCol - 1, antiDiagNo);

		// end of databaseSeg reached?
		minCol = max_logan(minCol,(antiDiagNo + 2 - rows));
		// end of querySeg reached?
		maxCol = min(maxCol, cols);
		
	
	}

	int longestExtensionCol = a3size + offset3 - 2;
	int longestExtensionRow = antiDiagNo - longestExtensionCol;
	int longestExtensionScore = antiDiag3[longestExtensionCol - offset3];
	
	if (longestExtensionScore == undefined)
	{
		if (antiDiag2[a2size -2] != undefined)
		{
			// reached end of query segment
			longestExtensionCol = a2size + offset2 - 2;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
			
		}
		else if (a2size > 2 && antiDiag2[a2size-3] != undefined)
		{
			// reached end of database segment
			longestExtensionCol = a2size + offset2 - 3;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
			
		}
	}


	//could be parallelized in some way
	if (longestExtensionScore == undefined){

		// general case
		for (int i = 0; i < a1size; ++i){

			if (antiDiag1[i] > longestExtensionScore){

				longestExtensionScore = antiDiag1[i];
				longestExtensionCol = i + offset1;
				longestExtensionRow = antiDiagNo - 2 - longestExtensionCol;
				
			}
		}
	}
	// update seed
	if (longestExtensionScore != undefined)
		updateExtendedSeedL(seed[myId], direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);
	//if(threadIdx.x == 0)		
		//printf("%d\n", longestExtensionScore);
	res[myId] = longestExtensionScore;
	
}

inline void extendSeedL(vector<SeedL> &seeds,
			ExtensionDirectionL direction,
			vector<string> &target,
			vector<string> &query,
			vector<ScoringSchemeL> &penalties,
			int const& XDrop,
			int const& kmer_length)
{
	//NB N_BLOCKS should be double or close as possible to target.size()=queryu.size()

	if(scoreGapExtend(penalties[0]) >= 0){

		std::cout<<"Error: Logan does not support gap extension penalty >= 0\n";
		exit(-1);
	}
	if(scoreGapOpen(penalties[0]) >= 0){

		std::cout<<"Error: Logan does not support gap opening penalty >= 0\n";
		exit(-1);
	}
	
	auto start_t1 = NOW;
	//declare streams
	cudaStream_t streams[2];	
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);

	// NB N_BLOCKS should be double or close as possible to target.size()=queryu.size()
	// change here for future implementations
	int nSequences = N_BLOCKS;

	//declare score for left extension
	int *scoreLeft = (int *)malloc(nSequences * sizeof(int));
	
	//declare score for right extension
	int *scoreRight = (int *)malloc(nSequences * sizeof(int));

	//SeedL *seed_ptr = (SeedL *)malloc(nSequences*sizeof(SeedL));

	int *lenLeftQ = (int*)malloc(nSequences*sizeof(int));
	int *lenLeftT = (int*)malloc(nSequences*sizeof(int));
  	int *lenRightQ = (int*)malloc(nSequences*sizeof(int));
  	int *lenRightT = (int*)malloc(nSequences*sizeof(int));
	int *offsetLeftQ = (int*)malloc(nSequences*sizeof(int));
	int *offsetLeftT = (int*)malloc(nSequences*sizeof(int));
  	int *offsetRightQ = (int*)malloc(nSequences*sizeof(int));
  	int *offsetRightT = (int*)malloc(nSequences*sizeof(int));


	vector<string> queryPrefix(nSequences);
	vector<string> targetPrefix(nSequences);
	vector<string> querySuffix(nSequences);
	vector<string> targetSuffix(nSequences);

//divide strings and allocate seed_pointers
	for(int i = 0; i<nSequences;i++){
		
		//penalties[i] = scoringScheme;
		//seed_ptr[i] = seeds[i];	
		queryPrefix[i] = query[i].substr(0, getBeginPositionV(seeds[i]));					// from read start til start seed (seed not included)
		targetPrefix[i] = target[i].substr(0, getBeginPositionH(seeds[i]));					// from read start til start seed (seed not included)
		querySuffix[i] = query[i].substr(getEndPositionV(seeds[i]), query[i].length());		// from end seed until the end (seed not included)
		targetSuffix[i] = target[i].substr(getEndPositionH(seeds[i]), target[i].length()); 	// from end seed until the end (seed not included)
	
	}

	//offset Left Query
	lenLeftQ[0]=offsetLeftQ[0]=queryPrefix[0].size();
  	for(int i = 1; i < nSequences; i++){
  		lenLeftQ[i]=queryPrefix[i].size();
  		offsetLeftQ[i]=offsetLeftQ[i-1]+queryPrefix[i].size();
  	}

  	//offset Left Target
  	lenLeftT[0]=offsetLeftT[0]=targetPrefix[0].size();
  	offsetLeftT[0]=targetPrefix[0].size();
  	for(int i = 1; i < nSequences; i++){
  		lenLeftT[i]=targetPrefix[i].size();
  		offsetLeftT[i]=offsetLeftT[i-1]+targetPrefix[i].size();
  	}

  	//offset Right Query
  	lenRightQ[0]=offsetRightQ[0]=querySuffix[0].size();
  	for(int i = 1; i < nSequences; i++){
  		lenRightQ[i]=querySuffix[i].size();
  		offsetRightQ[i]=offsetRightQ[i-1]+querySuffix[i].size();
  	}

  	//offset Right Target
  	lenRightT[0]=offsetRightT[0]=targetSuffix[0].size();
  	for(int i = 1; i < nSequences; i++){
  		lenRightT[i]=targetSuffix[i].size();
  		offsetRightT[i]=offsetRightT[i-1]+targetSuffix[i].size();
  	}

  	//total length of query/target prefix/suffix
  	int totalLengthQPref = offsetLeftQ[nSequences-1];
  	int totalLengthTPref = offsetLeftT[nSequences-1];
  	int totalLengthQSuff = offsetRightQ[nSequences-1];
  	int totalLengthTSuff = offsetRightT[nSequences-1];
	//cout << totalLengthQPref << " " << totalLengthTPref << " " << totalLengthQSuff << " " << totalLengthTSuff << " " << offsetRightT[28]<< endl;
  	//declare prefixes
	char *prefQ, *prefT;
  	//allocate and copy prefixes strings
  	prefQ = (char*)malloc(sizeof(char)*totalLengthQPref);
  	for(int i = 0; i<nSequences; i++){
  	 	char *seqptr = prefQ + offsetLeftQ[i] - queryPrefix[i].size();
  	 	memcpy(seqptr, queryPrefix[i].c_str(), queryPrefix[i].size());
  	}

  	prefT = (char*)malloc(sizeof(char)*totalLengthTPref);
  	for(int i = 0; i<nSequences; i++){
  	 	char *seqptr = prefT + offsetLeftT[i] - targetPrefix[i].size();;
  	 	memcpy(seqptr, targetPrefix[i].c_str(), targetPrefix[i].size());
  	}

  	//declare suffixes
	char *suffQ, *suffT;
  	//allocate and copy suffixes strings
  	suffQ = (char*)malloc(sizeof(char)*totalLengthQSuff);
  	for(int i = 0; i<nSequences; i++){
  	 	char *seqptr = suffQ + offsetRightQ[i] - querySuffix[i].size();
  	 	memcpy(seqptr, querySuffix[i].c_str(), querySuffix[i].size());
  	}

  	suffT = (char*)malloc(sizeof(char)*totalLengthTSuff);
  	for(int i = 0; i<nSequences; i++){
  	 	char *seqptr = suffT + offsetRightT[i] - targetSuffix[i].size();;
  	 	memcpy(seqptr, targetSuffix[i].c_str(), targetSuffix[i].size());
  	}

  	//declare and allocate GPU strings
  	char *prefQ_d, *prefT_d;
  	char *suffQ_d, *suffT_d;
  	cudaErrchk(cudaMalloc(&prefQ_d, totalLengthQPref*sizeof(char)));
  	cudaErrchk(cudaMalloc(&prefT_d, totalLengthTPref*sizeof(char)));
  	cudaErrchk(cudaMalloc(&suffQ_d, totalLengthQSuff*sizeof(char)));
  	cudaErrchk(cudaMalloc(&suffT_d, totalLengthTSuff*sizeof(char)));

  	//declare and allocate GPU lengths
  	int *lenLeftQ_d, *lenLeftT_d;
  	int *lenRightQ_d, *lenRightT_d;
  	cudaErrchk(cudaMalloc(&lenLeftQ_d, nSequences*sizeof(int)));
  	cudaErrchk(cudaMalloc(&lenLeftT_d, nSequences*sizeof(int)));
  	cudaErrchk(cudaMalloc(&lenRightQ_d, nSequences*sizeof(int)));
  	cudaErrchk(cudaMalloc(&lenRightT_d, nSequences*sizeof(int)));

  	//declare and allocate GPU offsets
  	int *offsetLeftQ_d, *offsetLeftT_d;
  	int *offsetRightQ_d, *offsetRightT_d;
  	cudaErrchk(cudaMalloc(&offsetLeftQ_d, nSequences*sizeof(int)));
  	cudaErrchk(cudaMalloc(&offsetLeftT_d, nSequences*sizeof(int)));
  	cudaErrchk(cudaMalloc(&offsetRightQ_d, nSequences*sizeof(int)));
  	cudaErrchk(cudaMalloc(&offsetRightT_d, nSequences*sizeof(int)));

  	//declare and allocate GPU seeds
  	SeedL *seed_d_l, *seed_d_r;
  	cudaErrchk(cudaMalloc(&seed_d_l, nSequences*sizeof(SeedL)));
  	cudaErrchk(cudaMalloc(&seed_d_r, nSequences*sizeof(SeedL)));

  	//declare and allocate GPU scoring
  	ScoringSchemeL *penalties_r, *penalties_l;
  	cudaErrchk(cudaMalloc(&penalties_r, nSequences*sizeof(ScoringSchemeL)));
  	cudaErrchk(cudaMalloc(&penalties_l, nSequences*sizeof(ScoringSchemeL)));

  	//declare result variables
  	int *scoreLeft_d, *scoreRight_d;
  	cudaErrchk(cudaMalloc(&scoreLeft_d, nSequences*sizeof(int)));
  	cudaErrchk(cudaMalloc(&scoreRight_d, nSequences*sizeof(int)));


  	//copy data to the GPU

  	//sequences
  	cudaErrchk(cudaMemcpyAsync(prefQ_d, prefQ, totalLengthQPref*sizeof(char), cudaMemcpyHostToDevice, streams[0]));	
  	cudaErrchk(cudaMemcpyAsync(prefT_d, prefT, totalLengthTPref*sizeof(char), cudaMemcpyHostToDevice, streams[0]));	
  	cudaErrchk(cudaMemcpyAsync(suffQ_d, suffQ, totalLengthQSuff*sizeof(char), cudaMemcpyHostToDevice, streams[1]));	
  	cudaErrchk(cudaMemcpyAsync(suffT_d, suffT, totalLengthTSuff*sizeof(char), cudaMemcpyHostToDevice, streams[1]));	
  	
  	//lengths
  	cudaErrchk(cudaMemcpyAsync(lenLeftQ_d, lenLeftQ, nSequences*sizeof(int), cudaMemcpyHostToDevice, streams[0]));	
  	cudaErrchk(cudaMemcpyAsync(lenLeftT_d, lenLeftT, nSequences*sizeof(int), cudaMemcpyHostToDevice, streams[0]));	
  	cudaErrchk(cudaMemcpyAsync(lenRightQ_d, lenRightQ, nSequences*sizeof(int), cudaMemcpyHostToDevice, streams[1]));	
  	cudaErrchk(cudaMemcpyAsync(lenRightT_d, lenRightT, nSequences*sizeof(int), cudaMemcpyHostToDevice, streams[1]));	
  	
  	//offsets
  	cudaErrchk(cudaMemcpyAsync(offsetLeftQ_d, offsetLeftQ, nSequences*sizeof(int), cudaMemcpyHostToDevice, streams[0]));	
  	cudaErrchk(cudaMemcpyAsync(offsetLeftT_d, offsetLeftT, nSequences*sizeof(int), cudaMemcpyHostToDevice, streams[0]));	
  	cudaErrchk(cudaMemcpyAsync(offsetRightQ_d, offsetRightQ, nSequences*sizeof(int), cudaMemcpyHostToDevice, streams[1]));	
  	cudaErrchk(cudaMemcpyAsync(offsetRightT_d, offsetRightT, nSequences*sizeof(int), cudaMemcpyHostToDevice, streams[1]));	
  	
  	//seeds
  	cudaErrchk(cudaMemcpyAsync(seed_d_l, &seeds[0], nSequences*sizeof(SeedL), cudaMemcpyHostToDevice, streams[0]));	
	cudaErrchk(cudaMemcpyAsync(seed_d_r, &seeds[0], nSequences*sizeof(SeedL), cudaMemcpyHostToDevice, streams[1]));
  	//scoring scheme
  	cudaErrchk(cudaMemcpyAsync(penalties_l, &penalties[0], nSequences*sizeof(ScoringSchemeL), cudaMemcpyHostToDevice, streams[0]));	
	cudaErrchk(cudaMemcpyAsync(penalties_r, &penalties[0], nSequences*sizeof(ScoringSchemeL), cudaMemcpyHostToDevice, streams[1]));	

	auto end_t1 = NOW;
	auto start_c = NOW;
	
	
	extendSeedLGappedXDropOneDirection <<<N_BLOCKS, N_THREADS, 0, streams[0]>>> (seed_d_l, prefQ_d, prefT_d, EXTEND_LEFTL, penalties_l, XDROP, scoreLeft_d, lenLeftQ_d, lenLeftT_d, offsetLeftQ_d, offsetLeftT_d);
	extendSeedLGappedXDropOneDirection <<<N_BLOCKS, N_THREADS, 0, streams[1]>>> (seed_d_r, suffQ_d, suffT_d, EXTEND_RIGHTL, penalties_r, XDROP, scoreRight_d, lenRightQ_d, lenRightT_d, offsetRightQ_d, offsetRightT_d);
	
	
	auto end_c = NOW;
    	auto start_t2 = NOW;
	
	cudaErrchk(cudaMemcpyAsync(scoreLeft, scoreLeft_d, nSequences*sizeof(int), cudaMemcpyDeviceToHost, streams[0]));
	cudaErrchk(cudaMemcpyAsync(scoreRight, scoreRight_d, nSequences*sizeof(int), cudaMemcpyDeviceToHost, streams[1]));
	cudaDeviceSynchronize();
	//cudaErrchk(cudaMemcpy(&seeds[0], seed_d_l, nSequences*sizeof(ScoringSchemeL), cudaMemcpyDeviceToHost));
	
	auto end_t2 = NOW;
	cudaErrchk(cudaPeekAtLastError());
	duration<double> transfer1, transfer2, compute, tfree;
	transfer1=end_t1-start_t1;
	transfer2=end_t2-start_t2;
	compute=end_c-start_c;
	
	auto start_f = NOW;
	
	cudaErrchk(cudaFree(prefQ_d));
	cudaErrchk(cudaFree(prefT_d));
	cudaErrchk(cudaFree(suffQ_d));
	cudaErrchk(cudaFree(suffT_d));
	cudaErrchk(cudaFree(lenLeftQ_d));
	cudaErrchk(cudaFree(lenLeftT_d));
	cudaErrchk(cudaFree(lenRightQ_d));
	cudaErrchk(cudaFree(lenRightT_d));
	cudaErrchk(cudaFree(offsetLeftQ_d));
	cudaErrchk(cudaFree(offsetLeftT_d));
	cudaErrchk(cudaFree(offsetRightQ_d));
	cudaErrchk(cudaFree(offsetRightT_d));
	cudaErrchk(cudaFree(seed_d_l));
	cudaErrchk(cudaFree(penalties_r));
	cudaErrchk(cudaFree(scoreLeft_d));
	cudaErrchk(cudaFree(scoreRight_d));

	auto end_f = NOW;
	tfree = end_f - start_f;
	//std::cout << "\nTransfer time1: "<<transfer1.count()<<" Transfer time2: "<<transfer2.count() <<" Compute time: "<<compute.count()  <<" Free time: "<< tfree.count() << std::endl;	

	//FIGURE OUT A WAY TO PRINT RESULTS
	//for(int i = 0; i < N_BLOCKS; i++)
	//	cout<< scoreLeft[i]+scoreRight[i]+kmer_length<<endl;	


}




