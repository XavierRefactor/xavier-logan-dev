//==================================================================
// Title:  Cuda x-drop seed-and-extend alignment algorithm
// Author: G. Guidi, A. Zeni
// Date:   6 March 2019
//==================================================================

#define N_THREADS 1024
#define N_BLOCKS 2900 
#define MIN -32768
#define BYTES_INT 4
#define XDROP 21
// #define N_STREAMS 60
#define MAX_SIZE_ANTIDIAG 1024
#define WARPSIZE 32
#define FULL_MASK 0xffffffff
#define N_WARPS N_THREADS/WARPSIZE

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

enum ExtensionDirectionL
{
	EXTEND_NONEL  = 0,
	EXTEND_LEFTL  = 1,
	EXTEND_RIGHTL = 2,
	EXTEND_BOTHL  = 3
};

__inline__ __device__ short simple_max(short *antidiag,
			  int &dim){
	int max = antidiag[0];
	for(int i = 1; i < dim; i++){
		if(antidiag[i]>max)
			max=antidiag[i];
	}
	return max;

}

__inline__ __device__ void warpReduce(volatile short *input,
                                          int myTId){
        input[myTId] = (input[myTId] > input[myTId + 32]) ? input[myTId] : input[myTId + 32]; 
        input[myTId] = (input[myTId] > input[myTId + 16]) ? input[myTId] : input[myTId + 16];
        input[myTId] = (input[myTId] > input[myTId + 8]) ? input[myTId] : input[myTId + 8]; 
        input[myTId] = (input[myTId] > input[myTId + 4]) ? input[myTId] : input[myTId + 4];
        input[myTId] = (input[myTId] > input[myTId + 2]) ? input[myTId] : input[myTId + 2];
        input[myTId] = (input[myTId] > input[myTId + 1]) ? input[myTId] : input[myTId + 1];
}

__inline__ __device__ short reduce_max(short *input, int dim){
	unsigned int myTId = threadIdx.x;   
	if(dim>32){
		for(int i = N_THREADS/2; i >32; i>>=1){
			if(myTId < i){
				        input[myTId] = (input[myTId] > input[myTId + i]) ? input[myTId] : input[myTId + i];
			}__syncthreads();
		}//__syncthreads();
	}
	if(myTId<32)
		warpReduce(input, myTId);
	__syncthreads();
	return input[0];
}
/*
__inline__ __device__
short warpReduceMax(short val) {
	for (int mask = WARPSIZE/2; mask > 0; mask /= 2){ 
		short v = __shfl_down_sync(FULL_MASK,val, mask);
		val = (v>val)?v:val;
	}
	return val;
}

__inline__ __device__
short blockReduceMax(short val) {

	static __shared__ short shared[N_WARPS]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % WARPSIZE;
	int wid = threadIdx.x / WARPSIZE;

	val = warpReduceMax(val);  

	if (lane==0)
		shared[wid]=val; 

	__syncthreads();             

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : UNDEF;

	if (wid==0) 
		val = warpReduceMax(val);
	if (threadIdx.x == 0)
		return val;

}*/

__inline__ __device__ void updateExtendedSeedL(SeedL &seed,
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
		seed.beginPositionH -= rows;
		seed.beginPositionV -= cols;
	} else {  // direction == EXTEND_RIGHTL
		// Set new lower and upper diagonals.
		int endDiag = seed.endDiagonal;
		if (getUpperDiagonal(seed) < endDiag - lowerDiag)
			setUpperDiagonal(seed, (endDiag - lowerDiag));
		if (getLowerDiagonal(seed) > (endDiag - upperDiag))
			setLowerDiagonal(seed, endDiag - upperDiag);

		// Set new end position of seed.
		seed.endPositionH += rows;
		seed.endPositionV += cols;
		
	}
}

__inline__ __device__ void computeAntidiag(short *antiDiag1,
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
	
	for(int i = 0; i < maxCol; i+=N_THREADS){

		int col = tid + minCol + i;
		int queryPos, dbPos;
		
		if(direction == EXTEND_LEFTL){
			queryPos = cols - 1 - col;
			dbPos = rows - 1 + col - antiDiagNo;
		}else{//EXTEND RIGHT
			queryPos = col - 1;
			dbPos = antiDiagNo - col - 1;
		}
		
		if(col < maxCol){
		
			int tmp = max_logan(antiDiag2[col-offset2],antiDiag2[col-offset2-1]) + GAP_EXT;
		
			int score = (querySeg[queryPos] == databaseSeg[dbPos]) ? MATCH : MISMATCH;
			
			tmp = max_logan(antiDiag1[col-offset1-1]+score,tmp);
			
			antiDiag3[tid+1+i] = (tmp < best - scoreDropOff) ? UNDEF : tmp;
		
		}
	}
}

__inline__ __device__ void calcExtendedLowerDiag(int &lowerDiag,
			  int const &minCol,
			  int const &antiDiagNo)
{
	int minRow = antiDiagNo - minCol;
	if (minCol - minRow < lowerDiag)
		lowerDiag = minCol - minRow;
}

__inline__ __device__ void calcExtendedUpperDiag(int &upperDiag,
			  int const &maxCol,
			  int const &antiDiagNo)
{
	int maxRow = antiDiagNo + 1 - maxCol;
	if (maxCol - 1 - maxRow > upperDiag)
		upperDiag = maxCol - 1 - maxRow;
}

__inline__ __device__ void initAntiDiag3(short *antiDiag3,
						   int &a3size,
						   int const &offset,
						   int const &maxCol,
						   int const &antiDiagNo,
						   int const &minScore,
						   int const &gapCost,
						   int const &undefined)
{
	a3size = maxCol + 1 - offset;

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

__inline__ __device__ void initAntiDiags(
			   short *antiDiag1,
			   short *antiDiag2,
			   short *antiDiag3,
			   int &a2size,
			   int &a3size,
			   int const &dropOff,
			   int const &gapCost,
			   int const &undefined)
{
	//antiDiag2.resize(1);
	a2size = 1;

	//resize(antiDiag2, 1);
	antiDiag2[0] = 0;

	//antiDiag3.resize(2);
	a3size = 2;

	// if (-gapCost > dropOff)
	// {
	// 	antiDiag3[0] = undefined;
	// 	antiDiag3[1] = undefined;
	// }
	// else
	// {
	 	antiDiag3[0] = gapCost;
	 	antiDiag3[1] = gapCost;
	// }
}

__global__ void extendSeedLGappedXDropOneDirection(
		SeedL *seed,
		char *querySegArray,
		char *databaseSegArray,
		ExtensionDirectionL direction,
		int scoreDropOff,
		int *res,
		int *offsetQuery,
		int *offsetTarget,
		int offAntidiag
		//,short *antidiag  //comment this line for shared/decomment for global
		)
{
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

	extern __shared__ short antidiagonals[]; //decomment this for shared/ comment for global
	short* antiDiag1 = &antidiagonals[0]; //decomment this for shared/ comment for global

	//short *antiDiag1 = &antidiag[myId*offAntidiag*3]; //decomment this for global/ comment for shared
	short* antiDiag2 = &antiDiag1[offAntidiag];
        short* antiDiag3 = &antiDiag2[offAntidiag];


	SeedL mySeed(seed[myId]);	
	//dimension of the antidiagonals
	int a1size = 0, a2size = 0, a3size = 0;
	int cols, rows;

    if(myId == 0){
            cols = offsetQuery[myId]+1;
            rows = offsetTarget[myId]+1;
    }
    else{
            cols = offsetQuery[myId]-offsetQuery[myId-1]+1;
            rows = offsetTarget[myId]-offsetTarget[myId-1]+1;
    }

	if (rows == 1 || cols == 1)
		return;

	//printf("%d\n", gapCost);
	//int undefined = UNDEF;
	int minCol = 1;
	int maxCol = 2;

	int offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
	int offset2 = 0; //                                                       in antiDiag2
	int offset3 = 0; //                                                       in antiDiag3

	initAntiDiags(antiDiag1,antiDiag2, antiDiag3, a2size, a3size, scoreDropOff, GAP_EXT, UNDEF);
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
		__shared__ short temp[N_THREADS];
		initAntiDiag3(antiDiag3, a3size, offset3, maxCol, antiDiagNo, best - scoreDropOff, GAP_EXT, UNDEF);
		
		computeAntidiag(antiDiag1, antiDiag2, antiDiag3, querySeg, databaseSeg, best, scoreDropOff, cols, rows, minCol, maxCol, antiDiagNo, offset1, offset2, direction);	 	
		__syncthreads();	
	
		int tmp, antiDiagBest = UNDEF;	
		for(int i=0; i<a3size; i+=N_THREADS){
			int size = a3size-i;
			
			if(myTId<N_THREADS){
				temp[myTId] = (myTId<size) ? antiDiag3[myTId+i]:UNDEF;				
			}
			__syncthreads();
			
			tmp = reduce_max(temp,size);
			antiDiagBest = (tmp>antiDiagBest) ? tmp:antiDiagBest;

		}
		best = (best > antiDiagBest) ? best : antiDiagBest;
		//int prova = simple_max(antiDiag3, a3size);	
		//if(prova!=antiDiagBest){
		//	if(myTId==0)
		//		printf("errore %d/%d\n", prova,antiDiagBest);
		//}
		while (minCol - offset3 < a3size && antiDiag3[minCol - offset3] == UNDEF &&
			   minCol - offset2 - 1 < a2size && antiDiag2[minCol - offset2 - 1] == UNDEF)
		{
			++minCol;
		}

		// Calculate new maxCol
		while (maxCol - offset3 > 0 && (antiDiag3[maxCol - offset3 - 1] == UNDEF) &&
									   (antiDiag2[maxCol - offset2 - 1] == UNDEF))
		{
			--maxCol;
		}
		++maxCol;

		// Calculate new lowerDiag and upperDiag of extended seed
		calcExtendedLowerDiag(lowerDiag, minCol, antiDiagNo);
		calcExtendedUpperDiag(upperDiag, maxCol - 1, antiDiagNo);
		
		// end of databaseSeg reached?
		minCol = (minCol > (antiDiagNo + 2 - rows)) ? minCol : (antiDiagNo + 2 - rows);
		// end of querySeg reached?
		maxCol = (maxCol < cols) ? maxCol : cols;
		//}
	}

	int longestExtensionCol = a3size + offset3 - 2;
	int longestExtensionRow = antiDiagNo - longestExtensionCol;
	int longestExtensionScore = antiDiag3[longestExtensionCol - offset3];
	
	if (longestExtensionScore == UNDEF)
	{
		if (antiDiag2[a2size -2] != UNDEF)
		{
			// reached end of query segment
			longestExtensionCol = a2size + offset2 - 2;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
			
		}
		else if (a2size > 2 && antiDiag2[a2size-3] != UNDEF)
		{
			// reached end of database segment
			longestExtensionCol = a2size + offset2 - 3;
			longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
			longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
			
		}
	}


	//could be parallelized in some way
	if (longestExtensionScore == UNDEF){

		// general case
		for (int i = 0; i < a1size; ++i){

			if (antiDiag1[i] > longestExtensionScore){

				longestExtensionScore = antiDiag1[i];
				longestExtensionCol = i + offset1;
				longestExtensionRow = antiDiagNo - 2 - longestExtensionCol;
			
			}
		}
	}
	
	if (longestExtensionScore != UNDEF)
		updateExtendedSeedL(mySeed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);
	seed[myId] = mySeed;
	res[myId] = longestExtensionScore;
	//}

}

inline void extendSeedL(vector<SeedL> &seeds,
			ExtensionDirectionL direction,
			vector<string> &target,
			vector<string> &query,
			vector<ScoringSchemeL> &penalties,
			int const& XDrop,
			int const& kmer_length,
			int *res)
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
	cudaStream_t stream_r, stream_l;	
	cudaStreamCreate(&stream_r);
	cudaStreamCreate(&stream_l);


	// NB N_BLOCKS should be double or close as possible to target.size()=queryu.size()
	// change here for future implementations
	int nSequences = N_BLOCKS;

	//declare score for left extension
	int *scoreLeft = (int *)malloc(nSequences * sizeof(int));
	
	//declare score for right extension
	int *scoreRight = (int *)malloc(nSequences * sizeof(int));

	//malloc data on the GPU ASAP to hide the time required by these operations
	int nSeqInt = nSequences*sizeof(int);

	//declare and allocate GPU offsets
	int *offsetLeftQ_d, *offsetLeftT_d;
	int *offsetRightQ_d, *offsetRightT_d;
	cudaErrchk(cudaMalloc(&offsetLeftQ_d, nSeqInt));
	cudaErrchk(cudaMalloc(&offsetLeftT_d, nSeqInt));
	cudaErrchk(cudaMalloc(&offsetRightQ_d, nSeqInt));
	cudaErrchk(cudaMalloc(&offsetRightT_d, nSeqInt));

	//declare result variables
	int *scoreLeft_d, *scoreRight_d;
	cudaErrchk(cudaMalloc(&scoreLeft_d, nSeqInt));
	cudaErrchk(cudaMalloc(&scoreRight_d, nSeqInt));

	//declare and allocate GPU seeds
	SeedL *seed_d_l, *seed_d_r;
	cudaErrchk(cudaMalloc(&seed_d_l, nSequences*sizeof(SeedL)));
	cudaErrchk(cudaMalloc(&seed_d_r, nSequences*sizeof(SeedL)));

	//copy seeds
	vector<SeedL> seeds_r;
	vector<SeedL> seeds_l;
	seeds_r.reserve(nSequences);
	seeds_l.reserve(nSequences);
	for (int i=0; i<seeds.size(); i++){
			seeds_r.push_back(seeds[i]);
			seeds_l.push_back(seeds[i]);
	}

	//copy seeds
	cudaErrchk(cudaMemcpyAsync(seed_d_l, &seeds_l[0], nSequences*sizeof(SeedL), cudaMemcpyHostToDevice, stream_l));
	cudaErrchk(cudaMemcpyAsync(seed_d_r, &seeds_r[0], nSequences*sizeof(SeedL), cudaMemcpyHostToDevice, stream_r));	

	//query and target suffix/prefix
	vector<string> queryPrefix(nSequences);
	vector<string> targetPrefix(nSequences);
	vector<string> querySuffix(nSequences);
	vector<string> targetSuffix(nSequences);
	
	//divide strings
	for(int i = 0; i<nSequences; i++){
		
		queryPrefix[i] = query[i].substr(0, getBeginPositionV(seeds[i]));					// from read start til start seed (seed not included)
		targetPrefix[i] = target[i].substr(0, getBeginPositionH(seeds[i]));					// from read start til start seed (seed not included)
		querySuffix[i] = query[i].substr(getEndPositionV(seeds[i]), query[i].length());		// from end seed until the end (seed not included)
		targetSuffix[i] = target[i].substr(getEndPositionH(seeds[i]), target[i].length()); 	// from end seed until the end (seed not included)

	}

	//compute and save sequences lenghts and offsets	 		
	vector<int> offsetLeftQ;
	vector<int> offsetLeftT;	
	vector<int> offsetRightQ;	
	vector<int> offsetRightT;	

	offsetLeftQ.reserve(nSequences);
	offsetLeftT.reserve(nSequences);
	offsetRightQ.reserve(nSequences);
	offsetRightT.reserve(nSequences);

	//shared_mem_size per block
    	int shared_left = 0;
    	int shared_right = 0;
	int n_cell = 0;

	for(int i = 0; i < nSequences; i++){
		offsetLeftQ.push_back(queryPrefix[i].size());
		offsetLeftT.push_back(targetPrefix[i].size());
		shared_left = std::max(std::min(offsetLeftQ[i],offsetLeftT[i]), shared_left);
		int tmp = std::min(offsetLeftQ[i],offsetLeftT[i]);
		n_cell += (tmp*tmp);
		offsetRightQ.push_back(querySuffix[i].size());
		offsetRightT.push_back(targetSuffix[i].size());
		shared_right = std::max(std::min(offsetRightQ[i], offsetRightT[i]), shared_right);
		tmp = std::min(offsetRightQ[i], offsetRightT[i]);
		n_cell += (tmp*tmp);
	}
	
	partial_sum(offsetLeftQ.begin(),offsetLeftQ.end(),offsetLeftQ.begin());	
	partial_sum(offsetLeftT.begin(),offsetLeftT.end(),offsetLeftT.begin());
	partial_sum(offsetRightQ.begin(),offsetRightQ.end(),offsetRightQ.begin());
	partial_sum(offsetRightT.begin(),offsetRightT.end(),offsetRightT.begin());

	//copy offsets
	cudaErrchk(cudaMemcpyAsync(offsetLeftQ_d, &offsetLeftQ[0], nSeqInt, cudaMemcpyHostToDevice, stream_l));
	cudaErrchk(cudaMemcpyAsync(offsetLeftT_d, &offsetLeftT[0], nSeqInt, cudaMemcpyHostToDevice, stream_l));
	cudaErrchk(cudaMemcpyAsync(offsetRightQ_d, &offsetRightQ[0], nSeqInt, cudaMemcpyHostToDevice, stream_r));
	cudaErrchk(cudaMemcpyAsync(offsetRightT_d, &offsetRightT[0], nSeqInt, cudaMemcpyHostToDevice, stream_r));

	//total lenght of the sequences
	int totalLengthQPref = offsetLeftQ[nSequences-1];
	int totalLengthTPref = offsetLeftT[nSequences-1];
	int totalLengthQSuff = offsetRightQ[nSequences-1];
	int totalLengthTSuff = offsetRightT[nSequences-1];

	//declare prefixes and suffixes
	char *prefQ, *prefT;
	char *suffQ, *suffT;

	//declare and allocate GPU antidiags
	short *ant_l, *ant_r;

	cudaErrchk(cudaMalloc(&ant_l, sizeof(short)*shared_left*3*nSequences));
	cudaErrchk(cudaMalloc(&ant_r, sizeof(short)*shared_right*3*nSequences));
	
	//declare and allocate GPU strings  
	char *prefQ_d, *prefT_d;
	char *suffQ_d, *suffT_d;

	prefQ = (char*)malloc(sizeof(char)*totalLengthQPref);
	prefT = (char*)malloc(sizeof(char)*totalLengthTPref);
	suffQ = (char*)malloc(sizeof(char)*totalLengthQSuff);
	suffT = (char*)malloc(sizeof(char)*totalLengthTSuff);

	memcpy(prefQ, queryPrefix[0].c_str(), offsetLeftQ[0]);
	for(int i = 1; i<nSequences; i++){
		char *seqptr = prefQ + offsetLeftQ[i-1];
		memcpy(seqptr, queryPrefix[i].c_str(), offsetLeftQ[i]-offsetLeftQ[i-1]);

	}

	memcpy(prefT, targetPrefix[0].c_str(), offsetLeftT[0]);
	for(int i = 1; i<nSequences; i++){
		char *seqptr = prefT + offsetLeftT[i-1];
		memcpy(seqptr, targetPrefix[i].c_str(), offsetLeftT[i]-offsetLeftT[i-1]);

	}

	memcpy(suffQ, querySuffix[0].c_str(), offsetRightQ[0]);
	for(int i = 1; i<nSequences; i++){
		char *seqptr = suffQ + offsetRightQ[i-1];
		memcpy(seqptr, querySuffix[i].c_str(), offsetRightQ[i]-offsetRightQ[i-1]);

	}

	memcpy(suffT, targetSuffix[0].c_str(), offsetRightT[0]);
	for(int i = 1; i<nSequences; i++){
		char *seqptr = suffT + offsetRightT[i-1];
		memcpy(seqptr, targetSuffix[i].c_str(), offsetRightT[i]-offsetRightT[i-1]);

	}
	
	cudaErrchk(cudaMalloc(&prefQ_d, totalLengthQPref*sizeof(char)));
	cudaErrchk(cudaMalloc(&prefT_d, totalLengthTPref*sizeof(char)));
	cudaErrchk(cudaMalloc(&suffQ_d, totalLengthQSuff*sizeof(char)));
	cudaErrchk(cudaMalloc(&suffT_d, totalLengthTSuff*sizeof(char)));
	
	//copy sequences
	cudaErrchk(cudaMemcpyAsync(prefQ_d, prefQ, totalLengthQPref*sizeof(char), cudaMemcpyHostToDevice, stream_l));
	cudaErrchk(cudaMemcpyAsync(prefT_d, prefT, totalLengthTPref*sizeof(char), cudaMemcpyHostToDevice, stream_l));
	cudaErrchk(cudaMemcpyAsync(suffQ_d, suffQ, totalLengthQSuff*sizeof(char), cudaMemcpyHostToDevice, stream_r));
	cudaErrchk(cudaMemcpyAsync(suffT_d, suffT, totalLengthTSuff*sizeof(char), cudaMemcpyHostToDevice, stream_r));

	auto end_t1 = NOW;
	duration<double> transfer1=end_t1-start_t1;
	std::cout << "Input setup time: " << transfer1.count() << std::endl;
	auto start_c = NOW;
	
	//andidiags global
	// extendSeedLGappedXDropOneDirection <<<N_BLOCKS, N_THREADS, 0, stream_l>>> (seed_d_l, prefQ_d, prefT_d, EXTEND_LEFTL, XDrop, scoreLeft_d, offsetLeftQ_d, offsetLeftT_d, shared_left, ant_l);
	// extendSeedLGappedXDropOneDirection <<<N_BLOCKS, N_THREADS, 0, stream_r>>> (seed_d_r, suffQ_d, suffT_d, EXTEND_RIGHTL, XDrop, scoreRight_d, offsetRightQ_d, offsetRightT_d, shared_right, ant_r);
	

	//antidiags shared	
	extendSeedLGappedXDropOneDirection <<<N_BLOCKS, N_THREADS, 3*shared_left*sizeof(short), stream_l>>> (seed_d_l, prefQ_d, prefT_d, EXTEND_LEFTL, XDrop, scoreLeft_d, offsetLeftQ_d, offsetLeftT_d, shared_left);
        extendSeedLGappedXDropOneDirection <<<N_BLOCKS, N_THREADS, 3*shared_right*sizeof(short), stream_r>>> (seed_d_r, suffQ_d, suffT_d, EXTEND_RIGHTL, XDrop, scoreRight_d, offsetRightQ_d, offsetRightT_d, shared_right);
	

	cudaErrchk(cudaMemcpyAsync(scoreLeft, scoreLeft_d, nSeqInt, cudaMemcpyDeviceToHost, stream_l));
	cudaErrchk(cudaMemcpyAsync(scoreRight, scoreRight_d, nSeqInt, cudaMemcpyDeviceToHost, stream_r));
	cudaErrchk(cudaMemcpyAsync(&seeds[0], seed_d_l, nSequences*sizeof(SeedL), cudaMemcpyDeviceToHost,stream_l));
	cudaErrchk(cudaMemcpyAsync(&seeds_r[0], seed_d_r, nSequences*sizeof(SeedL), cudaMemcpyDeviceToHost,stream_r));

	cudaDeviceSynchronize();

	auto end_c = NOW;
	duration<double> compute = end_c-start_c;
	std::cout << "Compute time: " << compute.count() << std::endl;

	cudaErrchk(cudaPeekAtLastError());

	auto start_f = NOW;
	
	cudaErrchk(cudaFree(prefQ_d));
	cudaErrchk(cudaFree(prefT_d));
	cudaErrchk(cudaFree(suffQ_d));
	cudaErrchk(cudaFree(suffT_d));
	cudaErrchk(cudaFree(offsetLeftQ_d));
	cudaErrchk(cudaFree(offsetLeftT_d));
	cudaErrchk(cudaFree(offsetRightQ_d));
	cudaErrchk(cudaFree(offsetRightT_d));
	cudaErrchk(cudaFree(seed_d_l));
	cudaErrchk(cudaFree(seed_d_r));
	cudaErrchk(cudaFree(scoreLeft_d));
	cudaErrchk(cudaFree(scoreRight_d));
	cudaErrchk(cudaFree(ant_l));
        cudaErrchk(cudaFree(ant_r));
	auto end_f = NOW;
	
	for(int i = 0; i < N_BLOCKS; i++){
		res[i] = scoreLeft[i]+scoreRight[i]+kmer_length;
		setEndPositionH(seeds[i], getEndPositionH(seeds_r[i]));    
		setEndPositionV(seeds[i], getEndPositionV(seeds_r[i])); 
	}
	for(int i = N_BLOCKS-29; i < N_BLOCKS; i++){
                cout<<res[i]<<endl;
        }

	std::cout << "N_CELLS: "<< n_cell<< " MAX DIM LEFT: "<<shared_left << " MAX DIM RIGHT: " << shared_right <<endl;
}
