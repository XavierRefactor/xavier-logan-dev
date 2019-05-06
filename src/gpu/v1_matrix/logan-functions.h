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
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
//#include <cub/cub.cuh>
// #include<boost/array.hpp>
#include"logan.h"
#include"score.h"

using namespace cub;

#define N_THREADS 1024
#define MIN -32768
#define BYTES_INT 4
#define N_STREAMS 60
//trying to see if the scoring scheme is a bottleneck in some way
#define MATCH     1
#define MISMATCH -1
#define GAP_EXT  -1
#define GAP_OPEN -1
#define UNDEF -32767
#define NOW std::chrono::high_resolution_clock::now()

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
				int dim,
				int minCol,
				int ant_offset)
{
	//printf("%d\n", dim1);
	__shared__ short localArray[N_THREADS/2];
	unsigned int tid = threadIdx.x;
	int half = dim>>1;
	if(tid < half){
		localArray[tid] = max_logan(array[tid+minCol-ant_offset],array[tid+minCol-ant_offset+half]);
	}
		
	for(int offset = dim/4; offset > 0; offset>>=1){
		if(tid < offset) localArray[tid] = max_logan(localArray[tid],localArray[tid+offset]);
	}
	__syncthreads();
	return localArray[0];
	
}

__device__ inline int array_max_old(int *array,
				int dim,
				int minCol,
				int ant_offset)
{
	//printf("%d\n", dim1);
	__shared__ int localArray[N_THREADS/2];
	unsigned int tid = threadIdx.x;
	if(tid < dim/2){
		localArray[tid] = max_logan(array[tid],array[tid+dim/2]);
	}
		
	for(int offset = dim/4; offset > 0; offset>>=1){
		if(tid < offset) localArray[tid] = max_logan(localArray[tid],localArray[tid+offset]);
	}
	return localArray[0];
	
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockReduceAlgorithm ALGORITHM>
__device__ inline int cub_max(int *array,
			      int dim,
			      int ant_offset){

	typedef BlockReduce<int, BLOCK_THREADS, ALGORITHM> BlockReduceT;
	__shared__ typename BlockReduceT::TempStorage temp_storage;
	int data[ITEMS_PER_THREAD];
	//if(ant_offset < threadIdx.x < dim+ant_offset) data = array[threadIdx.x];
	LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, array, data);
	int max = BlockReduceT(temp_storage).Reduce(data, cub::Max());
	return max;

}

__device__ int simple_max(int *antidiag,
			  int dim,
			  int offset){
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
					int cols,
					int rows,
					int lowerDiag,
					int upperDiag)
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

__device__ inline void computeAntidiagRight(int *antiDiag1,
				int *antiDiag2,
				int *antiDiag3,
				int offset1,
				int offset2,
				int offset3,
				//ExtensionDirectionL direction,
		//		int direction,
				int antiDiagNo,
				int gapCost,
				ScoringSchemeL scoringScheme,
				char* querySeg,
				char* databaseSeg,
				int undefined,
				int best,
				int scoreDropOff,
				int cols,
				int rows,
				int maxCol,
				int minCol)
{
	//printf(" GPU: %d\n", antiDiag1[minCol - offset1]);
	int threadId = threadIdx.x;
	//for (int col = minCol; col < maxCol; ++col) {
	// indices on anti-diagonals
	//int threadId = threadIdx.x;
	//if(threadId == 0)
		//printf(" GPU: %c\n", querySeg[0]);
	int col = threadId + minCol;
	if(col < maxCol){
		int i3 = col - offset3;
		int i2 = col - offset2;
		int i1 = col - offset1;

		// indices in query and database segments
		int queryPos, dbPos;
		
		queryPos = col - 1;
		dbPos = antiDiagNo - col - 1;
	
		
		// Calculate matrix entry (-> antiDiag3[col])
		int tmp = max(antiDiag2[i2-1], antiDiag2[i2]) +gapCost;
		int score = (querySeg[queryPos] == databaseSeg[dbPos]) ? scoringScheme.match_score  : scoringScheme.mismatch_score;
		tmp = max(tmp, antiDiag1[i1 - 1] + score);
		
		if (tmp < best - scoreDropOff)
		{
			//printf("in\n");
			antiDiag3[i3] = undefined;
		}
		else
		{
			antiDiag3[i3] = tmp;
			//antiDiagBest = max(antiDiagBest, tmp);
		}
		//printf("%d ", antiDiag3[i3]);			
	}
	//__syncthreads();
	
	//if(threadId == 0){
	//	printf("\n");
	//}
}

__device__ inline void computeRight(short *antiDiag1,
									short *antiDiag2,
									short *antiDiag3,
									char* querySeg,
									char* databaseSeg,
									int best,
									int scoreDropOff,
									int cols,
									int rows,
									int minCol,
									int maxCol,
									int antiDiagNo,
									int offset1,
									int offset2//,
									//ScoringSchemeL scoringScheme
									){
	int tid = threadIdx.x;
	int col = tid + minCol;
	int queryPos = col - 1;
	int dbPos = antiDiagNo - col - 1;
	
	if(col < maxCol){
		//printf("%d\n",antiDiagNo);
		int tmp = max_logan(antiDiag2[col-offset2],antiDiag2[col-offset2-1]) + GAP_EXT;
		//printf("%d\n",tid);
		int score = (querySeg[queryPos] == databaseSeg[dbPos]) ? MATCH : MISMATCH;
		tmp = max_logan(antiDiag1[col-offset1-1]+score,tmp);
		if(tmp < best - scoreDropOff)
			antiDiag3[tid+1]=UNDEF;	
		else
			antiDiag3[tid+1]=tmp;
	}
	
}

__device__ inline void computeLeft(short *antiDiag1,
									short *antiDiag2,
									short *antiDiag3,
									char* querySeg,
									char* databaseSeg,
									int best,
									int scoreDropOff,
									int cols,
									int rows,
									int minCol,
									int maxCol,
									int antiDiagNo,
									int offset1,
									int offset2//,
									//ScoringSchemeL scoringScheme
									){
	int tid = threadIdx.x;
	int col = tid + minCol;
	int queryPos = cols - 1 - col;
	int dbPos = rows - 1 + col - antiDiagNo;
	
	if(col < maxCol){
		//printf("%d\n",antiDiagNo);
		int tmp = max_logan(antiDiag2[col-offset2],antiDiag2[col-offset2-1]) + GAP_EXT;
		//printf("%d\n",tid);
		int score = (querySeg[queryPos] == databaseSeg[dbPos]) ? MATCH : MISMATCH;
		tmp = max_logan(antiDiag1[col-offset1-1]+score,tmp);
		antiDiag3[tid+1] = (tmp < best - scoreDropOff) ? UNDEF : tmp;
	}
}




__device__ inline void computeAntidiagLeft(int *antiDiag1,
				int *antiDiag2,
				int *antiDiag3,
				int offset1,
				int offset2,
				int offset3,
				//ExtensionDirectionL direction,
		//		int direction,
				int antiDiagNo,
				int gapCost,
				ScoringSchemeL scoringScheme,
				char* querySeg,
				char* databaseSeg,
				int undefined,
				int best,
				int scoreDropOff,
				int cols,
				int rows,
				int maxCol,
				int minCol)
{
	//printf(" GPU: %d\n", antiDiag1[minCol - offset1]);
	int threadId = threadIdx.x;
	//for (int col = minCol; col < maxCol; ++col) {
	// indices on anti-diagonals
	//int threadId = threadIdx.x;
	//if(threadId == 0)
		//printf(" GPU: %c\n", querySeg[0]);
	int col = threadId + minCol;
	if(col < maxCol){
		int i3 = col - offset3;
		int i2 = col - offset2;
		int i1 = col - offset1;

		// indices in query and database segments
		int queryPos, dbPos;
		
		queryPos = cols - 1 - col;
		dbPos = rows - 1 + col - antiDiagNo;
		
		
		// Calculate matrix entry (-> antiDiag3[col])
		int tmp = max_logan(antiDiag2[i2-1], antiDiag2[i2]) + gapCost;
		int score = (querySeg[queryPos] == databaseSeg[dbPos]) ? scoringScheme.match_score  : scoringScheme.mismatch_score;
		tmp = max_logan(tmp, antiDiag1[i1 - 1] + score);
		
		if (tmp < best - scoreDropOff)
		{
			//printf("in\n");
			antiDiag3[i3] = undefined;
		}
		else
		{
			antiDiag3[i3] = tmp;
			//antiDiagBest = max(antiDiagBest, tmp);
		}
		//printf("%d ", antiDiag3[i3]);				
	}
	//__syncthreads();
	
	//if(threadId == 0){
	//	printf("\n");
	//}
}

__device__ inline void mask_antidiag(int *antidiag,
						 int undefined,
						 int best,
						 int scoreDropOff){
	int tid = threadIdx.x;
	if (antidiag[tid] < best - scoreDropOff)
	{
		antidiag[tid] = undefined;
	}
	//__syncthreads();
}

__device__ inline void calcExtendedLowerDiag(int *lowerDiag,
		      int const minCol,
		      int const antiDiagNo)
{
	int minRow = antiDiagNo - minCol;
	if (minCol - minRow < *lowerDiag)
		*lowerDiag = minCol - minRow;
}

__device__ inline void calcExtendedUpperDiag(int *upperDiag,
			  int const maxCol,
			  int const antiDiagNo)
{
	int maxRow = antiDiagNo + 1 - maxCol;
	if (maxCol - 1 - maxRow > *upperDiag)
		*upperDiag = maxCol - 1 - maxRow;
}

__device__ inline void swapAntiDiags(int *antiDiag1,
	   				int *antiDiag2,
	   				int *antiDiag3)
{
	//std::vector<int> temp = antiDiag1;
	//swap(antiDiag1,antiDiag2);
	//swap(antiDiag2,antiDiag3);
	int *tmp = antiDiag1;
	antiDiag1 = antiDiag2;
	antiDiag2 = antiDiag3;
	//__syncthreads();
	antiDiag3 = tmp;
}

__device__ inline void initAntiDiag3(short *antiDiag3,
							int *a3size,
			   				int const offset,
			   				int const maxCol,
			   				int const antiDiagNo,
			   				int const minScore,
			   				int const gapCost,
			   				int const undefined)
{
	//antiDiag3.resize(maxCol + 1 - offset);
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
			   int const dropOff,
			   int const gapCost,
			   int const undefined)
{
	// antiDiagonals will be swaped in while loop BEFORE computation of antiDiag3 entries
	//  -> no initialization of antiDiag1 necessary
	int tid = threadIdx.x;
	if(tid<N_THREADS){
	
		antiDiag1[tid]=UNDEF;			
		antiDiag2[tid]=UNDEF;
		antiDiag3[tid]=UNDEF;
		

	}
	__syncthreads();
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

__global__ void extendSeedLGappedXDropOneDirectionLeft(
		SeedL* seed,
		char *querySeg,
		char *databaseSeg,
		//ExtensionDirectionL direction,
		ScoringSchemeL scoringScheme,
		int scoreDropOff,
		int *res,
		// int *antiDiag1,
		// int *antiDiag2,
		// int *antiDiag3,
		int qL,
		int dbL)
{
	//typedef typename Size<TQuerySegment>::Type int;
	//typedef typename SeedL<Simple,TConfig>::int int;
	__shared__ short antiDiag1p[N_THREADS];
	__shared__ short antiDiag2p[N_THREADS];
	__shared__ short antiDiag3p[N_THREADS];
	short* antiDiag1 = (short*) antiDiag1p;
	short* antiDiag2 = (short*) antiDiag2p;
	short* antiDiag3 = (short*) antiDiag3p;
	//dimension of the antidiagonals
	int a1size = 0, a2size = 0, a3size = 0;
	
	int cols = qL+1;//querySeg.length()+1;//should be enough even if we allocate just one time the string
	int rows = dbL+1;//databaseSeg.length()+1;//
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
		initAntiDiag3(antiDiag3, &a3size, offset3, maxCol, antiDiagNo, best - scoreDropOff, GAP_EXT, undefined);

		//int antiDiagBest = antiDiagNo * gapCost;	
		//computeAntidiagLeft(antiDiag1,antiDiag2,antiDiag3,offset1,offset2,offset3,antiDiagNo,GAP_EXT,scoringScheme,querySeg,databaseSeg,undefined,best,scoreDropOff,cols,rows,maxCol,minCol);
		computeLeft(antiDiag1, antiDiag2, antiDiag3, querySeg, databaseSeg, best, scoreDropOff, cols, rows, minCol, maxCol, antiDiagNo, offset1, offset2);	 	
		//__syncthreads();
	 	//mask_antidiag(antiDiag3,undefined,best,scoreDropOff);
		//__syncthreads();
		int antiDiagBest = array_max(antiDiag3, a3size, minCol, offset3);//maybe can be implemented as a shared value?
		//int antiDiagBest = cub_max<1024, 1, BLOCK_REDUCE_RAKING>(antiDiag3, a3size, offset3);
		//int antiDiagBest = cub_max<1024, 1, BLOCK_REDUCE_WARP_REDUCTIONS>(antiDiag3);
		//int antiDiagBest = simple_max(antiDiag3, a3size, offset3);
		//__syncthreads();
		//maxCol = maxCol - newMax + 1;
		//original min and max col update
		best = (best > antiDiagBest) ? best : antiDiagBest;
		//if(threadIdx.x == 0){
		while (minCol - offset3 < a3size && antiDiag3[minCol - offset3] == undefined &&
			   minCol - offset2 - 1 < a2size && antiDiag2[minCol - offset2 - 1] == undefined)
		{
			++minCol;
		}
		// if(threadIdx.x == 0) printf(" Mincol after: %d\n", minCol);
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
		//}
	
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
	//if (longestExtensionScore != undefined)
	//	updateExtendedSeedL(*seed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);
	//if(threadIdx.x == 0)		
		//printf("%d\n", longestExtensionScore);
	*res = longestExtensionScore;
	
}

__global__ void extendSeedLGappedXDropOneDirectionRight(
		SeedL* seed,
		char *querySeg,
		char *databaseSeg,
		//ExtensionDirectionL direction,
		ScoringSchemeL scoringScheme,
		int scoreDropOff,
		int *res,
		// int *antiDiag1,
		// int *antiDiag2,
		// int *antiDiag3,
		int qL,
		int dbL)
{
	//typedef typename Size<TQuerySegment>::Type int;
	//typedef typename SeedL<Simple,TConfig>::int int;
	__shared__ short antiDiag1p[N_THREADS];
	__shared__ short antiDiag2p[N_THREADS];
	__shared__ short antiDiag3p[N_THREADS];
	short* antiDiag1 = (short*) antiDiag1p;
	short* antiDiag2 = (short*) antiDiag2p;
	short* antiDiag3 = (short*) antiDiag3p;
	//dimension of the antidiagonals
	int a1size = 0, a2size = 0, a3size = 0;
	
	int cols = qL+1;//querySeg.length()+1;//should be enough even if we allocate just one time the string
	int rows = dbL+1;//databaseSeg.length()+1;//
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
		initAntiDiag3(antiDiag3, &a3size, offset3, maxCol, antiDiagNo, best - scoreDropOff, GAP_EXT, undefined);

		//int antiDiagBest = antiDiagNo * gapCost;	
		//computeAntidiagRight(antiDiag1,antiDiag2,antiDiag3,offset1,offset2,offset3,antiDiagNo,GAP_EXT,scoringScheme,querySeg,databaseSeg,undefined,best,scoreDropOff,cols,rows,maxCol,minCol);
		computeRight(antiDiag1, antiDiag2, antiDiag3, querySeg, databaseSeg, best, scoreDropOff, cols, rows, minCol, maxCol, antiDiagNo, offset1, offset2);
		//__syncthreads();
	 	//mask_antidiag(antiDiag3,undefined,best,scoreDropOff);
		//__syncthreads();
		int antiDiagBest = array_max(antiDiag3, a3size, minCol, offset3);//maybe can be implemented as a shared value?
		//int antiDiagBest = cub_max<1024, 1, BLOCK_REDUCE_RAKING>(antiDiag3, a3size, offset3);
		//int antiDiagBest = cub_max<1024, 1, BLOCK_REDUCE_WARP_REDUCTIONS>(antiDiag3);
		//int antiDiagBest = simple_max(antiDiag3, a3size, offset3);
		//__syncthreads();
		//maxCol = maxCol - newMax + 1;
		//original min and max col update
		best = (best > antiDiagBest) ? best : antiDiagBest;
		//if(threadIdx.x == 0){
		while (minCol - offset3 < a3size && antiDiag3[minCol - offset3] == undefined &&
			   minCol - offset2 - 1 < a2size && antiDiag2[minCol - offset2 - 1] == undefined)
		{
			++minCol;
		}
		// if(threadIdx.x == 0) printf(" Mincol after: %d\n", minCol);
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
		//}
	
	}

	short longestExtensionCol = a3size + offset3 - 2;
	short longestExtensionRow = antiDiagNo - longestExtensionCol;
	short longestExtensionScore = antiDiag3[longestExtensionCol - offset3];
	
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
	//if (longestExtensionScore != undefined)
	//	updateExtendedSeedL(*seed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);
//	if(threadIdx.x == 0)		
//		printf("%d\n", longestExtensionScore);
	*res = longestExtensionScore;
	
}

inline void extendSeedL(vector<SeedL> &seeds,
			ExtensionDirectionL direction,
			vector<string> &target,
			vector<string> &query,
			vector<ScoringSchemeL> &penalties,
			int const& XDrop,
			int const& kmer_length)
{
	//printf("extending");

	if(scoreGapExtend(penalties[0]) >= 0){

		std::cout<<"Error: Logan does not support gap extension penalty >= 0\n";
		exit(1);
	}
	if(scoreGapOpen(penalties[0]) >= 0){

		std::cout<<"Error: Logan does not support gap opening penalty >= 0\n";
		exit(1);
	}
	
	//declare streams
	cudaStream_t streams[N_STREAMS];
	for(int i = 0; i<N_STREAMS; i++)		
		cudaStreamCreate(&streams[i]);


	int **scoreLeft = (int **)malloc(N_STREAMS/2 * sizeof(int*));;
	for(int i = 0; i < N_STREAMS/2; i++) scoreLeft[i]=(int*)malloc(sizeof(int));
	int **scoreRight = (int **)malloc(N_STREAMS/2 * sizeof(int*));;
	for(int i = 0; i < N_STREAMS/2; i++) scoreRight[i]=(int*)malloc(sizeof(int));
	int len[N_STREAMS/2];
	
	
    vector<string> queryPrefix(N_STREAMS/2);
	vector<string> targetPrefix(N_STREAMS/2);
	vector<string> querySuffix(N_STREAMS/2);
	vector<string> targetSuffix(N_STREAMS/2);
	SeedL *seed_ptr[N_STREAMS/2];
	ScoringSchemeL *penalties_local[N_STREAMS/2];

	std::chrono::duration<double>  transfer1, transfer2, compute, tfree;
	auto start_t1 = NOW;

	char *q_l_d[N_STREAMS/2], *db_l_d[N_STREAMS/2];
	char *q_r_d[N_STREAMS/2], *db_r_d[N_STREAMS/2];
	int *scoreLeft_d[N_STREAMS/2];
	int *scoreRight_d[N_STREAMS/2];
	
	SeedL *seed_d_l[N_STREAMS/2];
	SeedL *seed_d_r[N_STREAMS/2];
	ScoringSchemeL *penalties_d_l[N_STREAMS/2];
	ScoringSchemeL *penalties_d_r[N_STREAMS/2];
	
	//divide strings and allocate seed_pointers
	for(int i = 0; i<N_STREAMS/2-1;i++){//PROBLEM HERE si rompe quando i = 29
		seed_ptr[i] = &seeds[i];	
		

		queryPrefix[i] = query[i].substr(0, getBeginPositionV(seeds[i]));					// from read start til start seed (seed not included)
		targetPrefix[i] = target[i].substr(0, getBeginPositionH(seeds[i]));				// from read start til start seed (seed not included)
		querySuffix[i] = query[i].substr(getEndPositionV(seeds[i]), query[i].length());		// from end seed until the end (seed not included)
		targetSuffix[i] = target[i].substr(getEndPositionH(seeds[i]), target[i].length()); 	// from end seed until the end (seed not included)
	}

	char *q_l[N_STREAMS/2];
	for(int i = 0; i < N_STREAMS/2; i++) q_l[i]=(char*)malloc(sizeof(char)*queryPrefix[i].length());
	char *db_l[N_STREAMS/2];
	for(int i = 0; i < N_STREAMS/2; i++) db_l[i]=(char*)malloc(sizeof(char)*targetPrefix[i].length());
	char *q_r[N_STREAMS/2];
	for(int i = 0; i < N_STREAMS/2; i++) q_r[i]=(char*)malloc(sizeof(char)*querySuffix[i].length());
	char *db_r[N_STREAMS/2];
	for(int i = 0; i < N_STREAMS/2; i++) db_r[i]=(char*)malloc(sizeof(char)*targetSuffix[i].length());

	//copy and allocate memory for every stream
	for(int i = 0; i < N_STREAMS/2-1; i++){//idem qua
		
		
		
		len[i] = max_logan(query[i].length(),target[i].length())*2;
		int minErrScore = UNDEF/len[i];
		setScoreGap(penalties[i], max_logan(scoreGap(penalties[i]), minErrScore));
		setScoreMismatch(penalties[i], max_logan(scoreMismatch(penalties[i]), minErrScore));
		//if (direction == EXTEND_LEFTL || direction == EXTEND_BOTHL){

		//allocate memory on host and copy string
		penalties_local[i] = &penalties[i];
		queryPrefix[i].copy(q_l[i], queryPrefix[i].length());
		targetPrefix[i].copy(db_l[i], targetPrefix[i].length());
		
		querySuffix[i].copy(q_r[i], querySuffix[i].length());
		targetSuffix[i].copy(db_r[i], targetSuffix[i].length());
	
		//ScoringSchemeL *info_left_d, *info_right_d;

		//allocate memory for the sequences
		cudaErrchk(cudaMalloc(&q_l_d[i], queryPrefix[i].length()*sizeof(char)));
		cudaErrchk(cudaMalloc(&db_l_d[i], targetPrefix[i].length()*sizeof(char)));
		cudaErrchk(cudaMalloc(&q_r_d[i], querySuffix[i].length()*sizeof(char)));
		cudaErrchk(cudaMalloc(&db_r_d[i], targetSuffix[i].length()*sizeof(char)));
		//allocate memory for seed and score
		cudaErrchk(cudaMalloc(&seed_d_l[i], sizeof(SeedL)));
		cudaErrchk(cudaMalloc(&scoreLeft_d[i], sizeof(int)));
		cudaErrchk(cudaMalloc(&seed_d_r[i], sizeof(SeedL)));
		cudaErrchk(cudaMalloc(&scoreRight_d[i], sizeof(int)));

		//allocate memory for scoring info
		//cudaErrchk(cudaMalloc(&info_left_d, sizeof(ScoringSchemeL)));
		    //cudaErrchk(cudaMalloc(&info_right_d, sizeof(ScoringSchemeL)));	

		//copy sequences, seeds and scoring info on GPU
		cudaErrchk(cudaMemcpyAsync(q_l_d[i], q_l[i], queryPrefix[i].length()*sizeof(char),cudaMemcpyHostToDevice, streams[i*2]));
		cudaErrchk(cudaMemcpyAsync(db_l_d[i], db_l[i], targetPrefix[i].length()*sizeof(char),cudaMemcpyHostToDevice, streams[i*2]));
		cudaErrchk(cudaMemcpyAsync(q_r_d[i], q_r[i], querySuffix[i].length()*sizeof(char),cudaMemcpyHostToDevice, streams[i*2+1]));
		cudaErrchk(cudaMemcpyAsync(db_r_d[i], db_r[i], targetSuffix[i].length()*sizeof(char),cudaMemcpyHostToDevice, streams[i*2+1]));

		cudaErrchk(cudaMemcpyAsync(seed_d_l[i], seed_ptr[i], sizeof(SeedL), cudaMemcpyHostToDevice, streams[i*2]));
		cudaErrchk(cudaMemcpyAsync(seed_d_r[i], seed_ptr[i], sizeof(SeedL), cudaMemcpyHostToDevice, streams[i*2+1]));
		
		//cudaErrchk(cudaMemcpyAsync(penalties_d_l[i], penalties_local[i], sizeof(ScoringSchemeL), cudaMemcpyHostToDevice, streams[i*2]));
		//cudaErrchk(cudaMemcpyAsync(penalties_d_r[i], penalties_local[i], sizeof(ScoringSchemeL), cudaMemcpyHostToDevice, streams[i*2+1]));

	}
	//cudaErrchk(cudaMemcpy(info_left_d, info_ptr, sizeof(ScoringSchemeL), cudaMemcpyHostToDevice));
        //cudaErrchk(cudaMemcpy(info_right_d, info_ptr, sizeof(ScoringSchemeL), cudaMemcpyHostToDevice));
	//call GPU to extend the seed
	auto end_t1 = NOW;
	auto start_c = NOW;
	
	for(int i = 0; i < N_STREAMS/2-1; i++){
		
		extendSeedLGappedXDropOneDirectionLeft <<<1, N_THREADS, 0, streams[i*2]>>> (seed_d_l[i], q_l_d[i], db_l_d[i], penalties[i], XDrop, scoreLeft_d[i], queryPrefix[i].length(), targetPrefix[i].length());//check seed
		extendSeedLGappedXDropOneDirectionRight <<<1, N_THREADS, 0, streams[i*2+1]>>> (seed_d_r[i], q_r_d[i], db_r_d[i], penalties[i], XDrop, scoreRight_d[i], querySuffix[i].length(),targetSuffix[i].length());//check seed
	
		
		//cudaErrchk(cudaStreamSynchronize(i*2));
		//cudaErrchk(cudaStreamSynchronize(i*2+1));
	}
	auto end_c = NOW;
    auto start_t2 = NOW;
	
	for(int i = 0; i < N_STREAMS/2-1; i++){
		cudaErrchk(cudaMemcpyAsync(seed_ptr[i], seed_d_l[i], sizeof(SeedL), cudaMemcpyDeviceToHost, streams[i*2]));//check
		cudaErrchk(cudaMemcpyAsync(scoreLeft[i], scoreLeft_d[i], sizeof(int), cudaMemcpyDeviceToHost, streams[i*2]));
		cudaErrchk(cudaMemcpyAsync(seed_ptr[i], seed_d_r[i], sizeof(SeedL), cudaMemcpyDeviceToHost, streams[i*2+1]));//check
		cudaErrchk(cudaMemcpyAsync(scoreRight[i], scoreRight_d[i], sizeof(int), cudaMemcpyDeviceToHost, streams[i*2+1]));//check
		
	}
	//cudaDeviceSynchronize();
	auto end_t2 = NOW;
	cudaErrchk(cudaPeekAtLastError());
	transfer1=end_t1-start_t1;
	transfer2=end_t2-start_t2;
	compute=end_c-start_c;
	
	auto start_f = NOW;
	
	//free(q_l);
	//free(db_l);
	//free(q_r);
	//free(db_r);

	cudaErrchk(cudaFree(*q_l_d));
	cudaErrchk(cudaFree(*db_l_d));
	cudaErrchk(cudaFree(*seed_d_l));
	cudaErrchk(cudaFree(*scoreLeft_d));
	cudaErrchk(cudaFree(*q_r_d));
	cudaErrchk(cudaFree(*db_r_d));
	cudaErrchk(cudaFree(*seed_d_r));
	cudaErrchk(cudaFree(*scoreRight_d));

	auto end_f = NOW;
	tfree = end_f - start_f;
	std::cout << "\nTransfer time1: "<<transfer1.count()<<" Transfer time2: "<<transfer2.count() <<" Compute time: "<<compute.count()  <<" Free time: "<< tfree.count() << std::endl;	

	for(int i = 0; i < N_STREAMS/2-1; i++)
		cout<< *scoreLeft[i]+*scoreRight[i]+kmer_length<<endl;	

	

	//std::cout << MIN-GAP_EXT << std::endl;
	//int *res=(int *)malloc(sizeof(int));
	//*res = *scoreLeft + *scoreRight + kmer_length;
	//return *res;

}

