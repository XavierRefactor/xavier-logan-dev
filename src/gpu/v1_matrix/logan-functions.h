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

#define N_THREADS 1000
#define MIN -2147483648

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

__device__ inline int array_max(int *array, 
						int dim) //TODO optimize this
{
	int max = MIN;
	for(int i = 0; i < dim; i++)
	{
		if(array[i]>max)
		{
			max = array[i];
		}
	}
	//if(threadIdx.x==0)
	//	printf("MAX: %d\n", max);
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

__device__ inline void computeAntidiag(int *antiDiag1,
				int *antiDiag2,
				int *antiDiag3,
				int offset1,
				int offset2,
				int offset3,
				ExtensionDirectionL direction,
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
		int tmp = max(antiDiag2[i2-1], antiDiag2[i2]) +gapCost;
		tmp = max(tmp, antiDiag1[i1 - 1] + score(scoringScheme, querySeg[queryPos], databaseSeg[dbPos]));
		
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
	__syncthreads();
	
	//if(threadId == 0){
	//	printf("\n");
	//}
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
	__syncthreads();
	antiDiag3 = tmp;
}

__device__ inline int initAntiDiag3(int *antiDiag3,
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
	return offset;
}

__device__ inline void initAntiDiags(int *antiDiag2,
			   int *antiDiag3,
			   int *a2size,
			   int *a3size,
			   int const dropOff,
			   int const gapCost,
			   int const undefined)
{
	// antiDiagonals will be swaped in while loop BEFORE computation of antiDiag3 entries
	//  -> no initialization of antiDiag1 necessary

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
		SeedL* seed,
		char *querySeg,
		char *databaseSeg,
		ExtensionDirectionL direction,
		ScoringSchemeL scoringScheme,
		int scoreDropOff,
		int *res,
		int *antiDiag1,
		int *antiDiag2,
		int *antiDiag3,
		int qL,
		int dbL)
{
	//typedef typename Size<TQuerySegment>::Type int;
	//typedef typename SeedL<Simple,TConfig>::int int;
	
	//dimension of the antidiagonals
	int a1size = 0, a2size = 0, a3size = 0;
	
	int cols = qL+1;//querySeg.length()+1;//should be enough even if we allocate just one time the string
	int rows = dbL+1;//databaseSeg.length()+1;//
	if (rows == 1 || cols == 1)
		return;

	// int minimumVal = INT_MIN;//as the minimum value will always be calculated starting from an integer, we can fix it without the need to call the function
	//int len = 2 * max(cols, rows); // number of antidiagonals (does not change in any implementation)
	// int minErrScore = minimumVal / len; // minimal allowed error penalty
	int gapCost = scoreGap(scoringScheme);
	//printf("%d\n", gapCost);
	int undefined = MIN - gapCost;
	int minCol = 1;
	int maxCol = 2;

	int offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
	int offset2 = 0; //                                                       in antiDiag2
	int offset3 = 0; //                                                       in antiDiag3

	initAntiDiags(antiDiag2, antiDiag3, &a2size, &a3size, scoreDropOff, gapCost, undefined);
	int antiDiagNo = 1; // the currently calculated anti-diagonal

	int best = 0; // maximal score value in the DP matrix (for drop-off calculation)

	int lowerDiag = 0;
	int upperDiag = 0;
	//if(threadIdx.x == 0){
	//	printf("SIZEOF Q: %d GPU: ", qL);
	//	for(int i = 0; i < qL; i++)
	//		printf("%c", querySeg[i]);
	//	printf("\n");
	//}
	//printf("%d %d %d %d\n", getBeginPositionH(*seed), getBeginPositionV(*seed), getEndPositionH(*seed), getEndPositionV(*seed));
	while (minCol < maxCol)
	{	

		
		++antiDiagNo;
 		//if(threadIdx.x == 0){
                //        for(int i = 0; i < 20; i++){
                //                printf("%d ", antiDiag3[i]);
                //        }
                //        printf("\n");
                //}
		//swapAntiDiags(antiDiag1, antiDiag2, antiDiag3);
		
		//antidiag swap
		int *t = antiDiag1;
		antiDiag1 = antiDiag2;
		antiDiag2 = antiDiag3;
		antiDiag3 = t;
		int t_l = a1size;
		a1size = a2size;
		a2size = a3size;
		a3size = t_l;
		__syncthreads();
		
		//if(threadIdx.x == 0){
                //        for(int i = 0; i < 20; i++){
                //                printf("%d ", antiDiag3[i]);
                //        }
                //        printf("\n");
                //}
		//antiDiag2 -> antiDiag1
		//antiDiag3 -> antiDiag2
		//antiDiag1 -> antiDiag3
		offset1 = offset2;
		offset2 = offset3;
		offset3 = minCol-1;
		initAntiDiag3(antiDiag3, &a3size, offset3, maxCol, antiDiagNo, best - scoreDropOff, gapCost, undefined);

		int antiDiagBest = antiDiagNo * gapCost;	
		//__syncthreads();	
		computeAntidiag(antiDiag1,antiDiag2,antiDiag3,offset1,offset2,offset3,direction,antiDiagNo,gapCost,scoringScheme,querySeg,databaseSeg,undefined,best,scoreDropOff,cols,rows,maxCol,minCol);
	 	__syncthreads();

		antiDiagBest = array_max(antiDiag3, a3size);
		//if(threadIdx.x == 0){
		//	for(int i = 0; i < 20; i++){
		//		printf("%d ", antiDiag3[i]);
		//	}
		//	printf("\n");
		//}
        	//std::cout << '\n';
		best = (best > antiDiagBest) ? best : antiDiagBest;
		// Calculate new minCol and minCol
		while (minCol - offset3 < a3size && antiDiag3[minCol - offset3] == undefined &&
			   minCol - offset2 - 1 < a2size && antiDiag2[minCol - offset2 - 1] == undefined)
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
		calcExtendedLowerDiag(&lowerDiag, minCol, antiDiagNo);
		calcExtendedUpperDiag(&upperDiag, maxCol - 1, antiDiagNo);

		// end of databaseSeg reached?
		minCol = max(minCol,(antiDiagNo + 2 - rows));
		// end of querySeg reached?
		maxCol = min(maxCol, cols);
		__syncthreads();	
	}
	//if(threadIdx.x==0)
	//	printf("BEST: %d\n", best);	
	int longestExtensionCol = a3size + offset3 - 2;
	int longestExtensionRow = antiDiagNo - longestExtensionCol;
	int longestExtensionScore = antiDiag3[longestExtensionCol - offset3];
	//if(threadIdx.x==0)
		//printf("%d %d %d\n", a1size, a2size, a3size); 
	//__syncthreads();
	__syncthreads();
	//*res = longestExtensionScore;
	//__syncthreads();
	if (longestExtensionScore == undefined)
	{
		if (antiDiag2[a2size -2] != undefined)
		{
			// reached end of query segment
			longestExtensionCol = a2size + offset2 - 2;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
			//if(threadIdx.x==0)
                		//printf("1 %d %d %d\n", longestExtensionCol, longestExtensionRow, longestExtensionScore); 
		}
		else if (a2size > 2 && antiDiag2[a2size-3] != undefined)
		{
			// reached end of database segment
			longestExtensionCol = a2size + offset2 - 3;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
			//if(threadIdx.x==0)
                		//printf("2 %d %d %d\n", longestExtensionCol, longestExtensionRow, longestExtensionScore); 
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
				//if(threadIdx.x==0)
                                //	printf("3 %d %d %d\n", longestExtensionCol, longestExtensionRow, longestExtensionScore);
			}
		}
	}
	// update seed
	if (longestExtensionScore != undefined)//AAAA it was !=
		updateExtendedSeedL(*seed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);
	
	//__syncthreads();
	*res = longestExtensionScore;
	//if(threadIdx.x == 0)
	//	printf("%d \n", longestExtensionScore );
	//}
	__syncthreads();
}


inline int extendSeedL(SeedL &seed,
			ExtensionDirectionL direction,
			std::string const& target,
			std::string const& query,
			ScoringSchemeL & penalties,
			int const& XDrop,
			int const& kmer_length)
{
	//printf("extending");

	if(scoreGapExtend(penalties) >= 0){

		std::cout<<"Error: Logan does not support gap extension penalty >= 0\n";
		exit(1);
	}
	if(scoreGapOpen(penalties) >= 0){

		std::cout<<"Error: Logan does not support gap opening penalty >= 0\n";
		exit(1);
	}
	//assert(scoreMismatch(penalties) < 0);
	//assert(scoreMatch(penalties) > 0); 
	//assert(scoreGapOpen(penalties) == scoreGapExtend(penalties));
	SeedL *seed_ptr = &seed;
	int *scoreLeft=(int *)malloc(sizeof(int));
	int *scoreRight=(int *)malloc(sizeof(int));
	int len = max(query.length(),target.length())*2;
	int minErrScore = MIN / len ;
	setScoreGap(penalties, max(scoreGap(penalties), minErrScore));
	setScoreMismatch(penalties, max(scoreMismatch(penalties), minErrScore));

	//Result scoreFinal;
	//std::cout << seed.beginPositionH << " "<<seed.beginPositionV << " " <<seed.endPositionH << " " << seed.beginPositionH << std::endl;
	if (direction == EXTEND_LEFTL || direction == EXTEND_BOTHL){

		//AAAA maybe extracting the substring could be avoided and we can directly copy the chars in the character array instead

		std::string queryPrefix = query.substr(0, getBeginPositionV(seed));	// from read start til start seed (seed not included)
		std::string targetPrefix = target.substr(0, getBeginPositionH(seed));	// from read start til start seed (seed not included)
		
		//allocate memory on host and copy string
		char *q_l = (char *)malloc(sizeof(char)*queryPrefix.length());
		char *db_l = (char *)malloc(sizeof(char)*targetPrefix.length());
		//queryPrefix.copy(q_l, queryPrefix.length());
		//targetPrefix.copy(db_l, targetPrefix.length());
		for(int i = 0; i < queryPrefix.length(); i++){
			q_l[i]=queryPrefix[i];
		}
		//std::cout << std::endl;
		for(int i = 0; i < targetPrefix.length(); i++){
                        db_l[i]=targetPrefix[i];
                }
		//declare vars for the gpu
		char *q_l_d, *db_l_d;
		int *a1_l, *a2_l, *a3_l; //AAAA think if a fourth is necessary for the swap
		int *scoreLeft_d;
		SeedL *seed_d;
		std::chrono::duration<double>  transfer1, transfer2, compute;
		auto start_t1 = std::chrono::high_resolution_clock::now();
		//allocate memory for the antidiagonals
		cudaErrchk(cudaMalloc(&a1_l, min(queryPrefix.length(),targetPrefix.length())*sizeof(int)));
		cudaErrchk(cudaMalloc(&a2_l, min(queryPrefix.length(),targetPrefix.length())*sizeof(int)));
		cudaErrchk(cudaMalloc(&a3_l, min(queryPrefix.length(),targetPrefix.length())*sizeof(int)));

		//allocate memory for the strings and copy them
		cudaErrchk(cudaMalloc(&q_l_d, queryPrefix.length()*sizeof(char)));
		cudaErrchk(cudaMalloc(&db_l_d, targetPrefix.length()*sizeof(char)));

		cudaErrchk(cudaMemcpy(q_l_d, q_l, queryPrefix.length()*sizeof(char),cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(db_l_d, db_l, targetPrefix.length()*sizeof(char),cudaMemcpyHostToDevice));

		//allocate memory for seed and score
		cudaErrchk(cudaMalloc(&seed_d, sizeof(SeedL)));
		cudaErrchk(cudaMalloc(&scoreLeft_d, sizeof(int)));

		cudaErrchk(cudaMemcpy(seed_d, seed_ptr, sizeof(SeedL), cudaMemcpyHostToDevice));//check
		//call GPU to extend the seed
		auto end_t1 = std::chrono::high_resolution_clock::now();
		auto start_c = std::chrono::high_resolution_clock::now();
		extendSeedLGappedXDropOneDirection <<<1, N_THREADS>>> (seed_d, q_l_d, db_l_d, EXTEND_LEFTL, penalties, XDrop, scoreLeft_d, a1_l, a2_l, a3_l, queryPrefix.length(), targetPrefix.length());//check seed
		auto end_c = std::chrono::high_resolution_clock::now();
		auto start_t2 = std::chrono::high_resolution_clock::now();
		cudaErrchk(cudaPeekAtLastError());
		cudaErrchk(cudaDeviceSynchronize());

		cudaErrchk(cudaMemcpy(seed_ptr, seed_d, sizeof(SeedL), cudaMemcpyDeviceToHost));//check
		cudaErrchk(cudaMemcpy(scoreLeft, scoreLeft_d, sizeof(int), cudaMemcpyDeviceToHost));//check
		auto end_t2 = std::chrono::high_resolution_clock::now();
		transfer1=end_t1-start_t1;
		transfer2=end_t2-start_t2;
		compute=end_c-start_c;
		std::cout << "Transfer time1: "<<transfer1.count()<<" Transfer time2: "<<transfer2.count() <<" Compute time: "<<compute.count()  << std::endl;
		free(q_l);
		free(db_l);
		cudaErrchk(cudaFree(a1_l));
		cudaErrchk(cudaFree(a2_l));
		cudaErrchk(cudaFree(a3_l));
		cudaErrchk(cudaFree(q_l_d));
		cudaErrchk(cudaFree(db_l_d));
		cudaErrchk(cudaFree(seed_d));
		cudaErrchk(cudaFree(scoreLeft_d));
		
	}

	if (direction == EXTEND_RIGHTL || direction == EXTEND_BOTHL){

		std::string querySuffix = query.substr(getEndPositionV(seed), query.length());		// from end seed until the end (seed not included)
		std::string targetSuffix = target.substr(getEndPositionH(seed), target.length()); 	// from end seed until the end (seed not included)
		
		//allocate memory on host
		//allocate memory on host and copy string
		char *q_r = (char *)malloc(sizeof(char)*querySuffix.length());
		char *db_r = (char *)malloc(sizeof(char)*targetSuffix.length());
		//querySuffix.copy(q_r, querySuffix.length());
		//targetSuffix.copy(db_r, targetSuffix.length());

		for(int i = 0; i < querySuffix.length(); i++){
                        q_r[i]=querySuffix[i];
                }
                //std::cout << std::endl;
                for(int i = 0; i < targetSuffix.length(); i++){
                        db_r[i]=targetSuffix[i];
                }
		//declare vars for the gpu
		char *q_r_d, *db_r_d;
		int *a1_r, *a2_r, *a3_r; //AAAA think if a fourth is necessary for the swap
		int *scoreRight_d;
		SeedL *seed_d;
		std::chrono::duration<double>  transfer1, transfer2, compute;
                auto start_t1 = std::chrono::high_resolution_clock::now();
		//allocate memory for the antidiagonals
		cudaErrchk(cudaMalloc(&a1_r, min(querySuffix.length(),targetSuffix.length())*sizeof(int)));
		cudaErrchk(cudaMalloc(&a2_r, min(querySuffix.length(),targetSuffix.length())*sizeof(int)));
		cudaErrchk(cudaMalloc(&a3_r, min(querySuffix.length(),targetSuffix.length())*sizeof(int)));

		//allocate memory for the strings and copy them
		cudaErrchk(cudaMalloc(&q_r_d, querySuffix.length()*sizeof(char)));
		cudaErrchk(cudaMalloc(&db_r_d, targetSuffix.length()*sizeof(char)));

		cudaErrchk(cudaMemcpy(q_r_d, q_r, querySuffix.length()*sizeof(char),cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(db_r_d, db_r, targetSuffix.length()*sizeof(char),cudaMemcpyHostToDevice));

		//allocate memory for seed and score
		cudaErrchk(cudaMalloc(&seed_d, sizeof(SeedL)));
		cudaErrchk(cudaMalloc(&scoreRight_d, sizeof(int)));

		cudaErrchk(cudaMemcpy(seed_d, seed_ptr, sizeof(SeedL), cudaMemcpyHostToDevice));//check
		//call GPU to extend the seed
		auto end_t1 = std::chrono::high_resolution_clock::now();
                auto start_c = std::chrono::high_resolution_clock::now();
		extendSeedLGappedXDropOneDirection <<<1, N_THREADS>>> (seed_d, q_r_d, db_r_d, EXTEND_RIGHTL, penalties, XDrop, scoreRight_d, a1_r, a2_r, a3_r, querySuffix.length(),targetSuffix.length());//check seed
		auto end_c = std::chrono::high_resolution_clock::now();
                auto start_t2 = std::chrono::high_resolution_clock::now();
		cudaErrchk(cudaPeekAtLastError());
		cudaErrchk(cudaDeviceSynchronize());

		cudaErrchk(cudaMemcpy(seed_ptr, seed_d, sizeof(SeedL), cudaMemcpyDeviceToHost));//check
		cudaErrchk(cudaMemcpy(scoreRight, scoreRight_d, sizeof(int), cudaMemcpyDeviceToHost));//check
		auto end_t2 = std::chrono::high_resolution_clock::now();
                transfer1=end_t1-start_t1;
                transfer2=end_t2-start_t2;
                compute=end_c-start_c;
		std::cout << "Transfer time1: "<<transfer1.count()<<" Transfer time2: "<<transfer2.count() <<" Compute time: "<<compute.count()  << "\n\n";
		free(q_r);
		free(db_r);
		cudaErrchk(cudaFree(a1_r));
		cudaErrchk(cudaFree(a2_r));
		cudaErrchk(cudaFree(a3_r));
		cudaErrchk(cudaFree(q_r_d));
		cudaErrchk(cudaFree(db_r_d));
		cudaErrchk(cudaFree(seed_d));
		cudaErrchk(cudaFree(scoreRight_d));


	}


	int *res=(int *)malloc(sizeof(int));
	*res = *scoreLeft + *scoreRight + kmer_length;
	return *res;

}

