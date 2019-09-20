//==================================================================
// Title:  LOGAN: X-Drop Adaptive Banded Alignment
// Author: G. Guidi, E. Younis
// Date:   22 April 2019
//==================================================================

#include<vector>
#include<iostream>
#include<omp.h>
#include<algorithm>
#include<inttypes.h>
#include<assert.h>
#include<iterator>
#include<x86intrin.h>
#include"utils.h"
#include"simd_utils.h"
#include"score.h"

#ifdef DEBUG
	#define myLog( var ) do { std::cerr << "LOG: " << __FILE__ << "(" << __LINE__ << ") " << #var << " = " << (var) << std::endl; } while(0)
#else
	#define myLog( var )
#endif

class LoganState
{
public:
	LoganState
	(
		SeedL& _seed,
	 	std::string const& targetSeg,
		std::string const& querySeg,
		ScoringSchemeL& scoringScheme,
		int64_t const &_scoreDropOff
	)
	{
		seed = _seed;

		hlength = targetSeg.length() + 1;
		vlength = querySeg.length()  + 1;

		if (hlength <= 1 || vlength <= 1)
			myLog("ERROR STATE, Fix this here buddy!");

		// Convert from string to int array
		// This is the entire sequences
		queryh = new int8_t[hlength];
		queryv = new int8_t[vlength];
		std::copy(targetSeg.begin(), targetSeg.end(), queryh);
		std::copy(querySeg.begin(), querySeg.end(), queryv);

		// pay attention
		// myLog( "Verify Here (ASSERT)" );

		matchCost    = scoreMatch(scoringScheme   );
		mismatchCost = scoreMismatch(scoringScheme);
		gapCost      = scoreGap(scoringScheme     );

		vmatchCost    = setOp (matchCost   );
		vmismatchCost = setOp (mismatchCost);
		vgapCost      = setOp (gapCost     );
		vzeros        = _mm256_setzero_si256();

		hoffset = LOGICALWIDTH;
		voffset = LOGICALWIDTH;

		bestScore    = 0;
		currScore    = 0;
		scoreOffset  = 0;
		scoreDropOff = _scoreDropOff;
		xDropCond   = false;
	}

	~LoganState()
	{
		delete [] queryh;
		delete [] queryv;
	}

	// i think this can be smaller than 64bit
	int64_t get_score_offset  ( void ) { return scoreOffset;  }
	int64_t get_best_score    ( void ) { return bestScore;    }
	int64_t get_curr_score    ( void ) { return currScore;    }
	int64_t get_score_dropoff ( void ) { return scoreDropOff; }

	void set_score_offset ( int64_t _scoreOffset ) { scoreOffset = _scoreOffset; }
	void set_best_score   ( int64_t _bestScore   ) { bestScore   = _bestScore;   }
	void set_curr_score   ( int64_t _currScore   ) { currScore   = _currScore;   }

	int8_t get_match_cost    ( void ) { return matchCost;    }
	int8_t get_mismatch_cost ( void ) { return mismatchCost; }
	int8_t get_gap_cost      ( void ) { return gapCost;      }

	vectorType get_vqueryh ( void ) { return vqueryh.simd; }
	vectorType get_vqueryv ( void ) { return vqueryv.simd; }

	vectorType get_antiDiag1 ( void ) { return antiDiag1.simd; }
	vectorType get_antiDiag2 ( void ) { return antiDiag2.simd; }
	vectorType get_antiDiag3 ( void ) { return antiDiag3.simd; }

	vectorType get_vmatchCost    ( void ) { return vmatchCost;    }
	vectorType get_vmismatchCost ( void ) { return vmismatchCost; }
	vectorType get_vgapCost      ( void ) { return vgapCost;      }
	vectorType get_vzeros        ( void ) { return vzeros;        }

	void update_vqueryh ( uint8_t idx, int8_t value ) { vqueryh.elem[idx] = value; }
	void update_vqueryv ( uint8_t idx, int8_t value ) { vqueryv.elem[idx] = value; }

	void update_antiDiag1 ( uint8_t idx, int8_t value ) { antiDiag1.elem[idx] = value; }
	void update_antiDiag2 ( uint8_t idx, int8_t value ) { antiDiag2.elem[idx] = value; }
	void update_antiDiag3 ( uint8_t idx, int8_t value ) { antiDiag3.elem[idx] = value; }

	void broadcast_antiDiag1 ( int8_t value ) { antiDiag1.simd = setOp( value ); }
	void broadcast_antiDiag2 ( int8_t value ) { antiDiag2.simd = setOp( value ); }
	void broadcast_antiDiag3 ( int8_t value ) { antiDiag3.simd = setOp( value ); }

	void set_antiDiag1 ( vectorType vector ) { antiDiag1.simd = vector; }
	void set_antiDiag2 ( vectorType vector ) { antiDiag2.simd = vector; }
	void set_antiDiag3 ( vectorType vector ) { antiDiag3.simd = vector; }

	void moveRight ( void )
	{
		// (a) shift to the left on query horizontal
		vqueryh = shiftLeft( vqueryh.simd );
		vqueryh.elem[LOGICALWIDTH - 1] = queryh[hoffset++];

		// (b) shift left on updated vector 1
		// this places the right-aligned vector 2 as a left-aligned vector 1
		antiDiag1.simd = antiDiag2.simd;
		antiDiag1 = shiftLeft( antiDiag1.simd );
		antiDiag2.simd = antiDiag3.simd;
	}

	void moveDown ( void )
	{
		// (a) shift to the right on query vertical
		vqueryv = shiftRight( vqueryv.simd );
		vqueryv.elem[0] = queryv[voffset++];

		// (b) shift to the right on updated vector 2
		// this places the left-aligned vector 3 as a right-aligned vector 2
		antiDiag1.simd = antiDiag2.simd;
		antiDiag2.simd = antiDiag3.simd;
		antiDiag2 = shiftRight( antiDiag2.simd );
	}

	// Seed position (define starting position and need to be updated when exiting)
	SeedL seed;

	// Sequence Lengths
	unsigned int hlength;
	unsigned int vlength;

	// Sequences as ints
	int8_t* queryh;
	int8_t* queryv;

	// Sequence pointers
	int hoffset;
	int voffset;

	// Constant Scoring Values
	int8_t matchCost;
	int8_t mismatchCost;
	int8_t gapCost;

	// Constant Scoring Vectors
	vectorType vmatchCost;
	vectorType vmismatchCost;
	vectorType vgapCost;
	vectorType vzeros;

	// Computation Vectors
	vector_union_t antiDiag1;
	vector_union_t antiDiag2;
	vector_union_t antiDiag3;

	vector_union_t vqueryh;
	vector_union_t vqueryv;

	// X-Drop Variables
	int64_t bestScore;
	int64_t currScore;
	int64_t scoreOffset;
	int64_t scoreDropOff;
	bool xDropCond;
};

// GG: max penalty = +/- 3
void
LoganPhase1(LoganState& state)
{
	myLog("Phase1");

	// we need one more space for the off-grid values and one more space for antiDiag2
	int DPmatrix[LOGICALWIDTH + 2][LOGICALWIDTH + 2];

	// DPmatrix initialization
	DPmatrix[0][0] = 0;
	for(int i = 1; i < LOGICALWIDTH + 2; i++ )
	{
		DPmatrix[0][i] = -i;
		DPmatrix[i][0] = -i;
	}

	// DPmax tracks maximum value in DPmatrix for xdrop condition
	int DPmax = 0;

	// dynamic programming loop to fill DPmatrix
	for(int i = 1; i < LOGICALWIDTH + 2; i++)
	{
		for (int j = 1; j < LOGICALWIDTH + 2; j++)
		{
			int oneF = DPmatrix[i-1][j-1];

			// comparing bases
			if(state.queryh[i-1] == state.queryv[j-1])
				oneF += state.get_match_cost();
			else
				oneF += state.get_mismatch_cost();

			int twoF = std::max(DPmatrix[i-1][j], DPmatrix[i][j-1]);
			twoF += state.get_gap_cost();

			DPmatrix[i][j] = std::max(oneF, twoF);

			// heuristic to keep track of the max in phase1
			if(DPmatrix[i][j] > DPmax)
				DPmax = DPmatrix[i][j];
		}
	}

	for(int i = 0; i < LOGICALWIDTH; ++i)
	{
		state.update_vqueryh(i, state.queryh[i + 1]);
		state.update_vqueryv(i, state.queryv[LOGICALWIDTH - i]);
	}

	state.update_vqueryh(LOGICALWIDTH, NINF);
	state.update_vqueryv(LOGICALWIDTH, NINF);

	// let's assume we fit in int8_t
	// int64_t antiDiagMin = std::numeric_limits<int64_t>::max();
	int antiDiag1Max = std::numeric_limits<int8_t>::min();

	// load DPmatrix into antiDiag1 and antiDiag2 vector and find max elem at the end of phase1 in antiDiag1
	// GG: double check loop bound
	for(int i = 1; i < LOGICALWIDTH + 1; ++i)
	{
		int value1 = DPmatrix[i][LOGICALWIDTH - i + 1];
		int value2 = DPmatrix[i + 1][LOGICALWIDTH - i + 1];

		state.update_antiDiag1(i - 1, value1);
		state.update_antiDiag2(i, value2);
		// state.update_antiDiag1(i - 1, value1 - state.get_score_offset());
		// state.update_antiDiag2(i, value2 - state.get_score_offset());

		if(value1 > antiDiag1Max)
			antiDiag1Max = value1;
	}
	state.update_antiDiag1(LOGICALWIDTH, NINF);
	state.update_antiDiag2(0, NINF);

	// clear antiDiag3
	state.broadcast_antiDiag3(NINF);

	state.set_best_score(DPmax);
	state.set_curr_score(antiDiag1Max);

	// myLog(DPmax);
	// myLog(antiDiag1Max);

	// check x-drop condition
	if(antiDiag1Max < DPmax - state.get_score_dropoff())
	{
		state.xDropCond = true;
		return;
	}

	// I'm gonna assume this is fine for now
	// if(antiDiagMax > CUTOFF)
	//	state.set_score_offset(state.get_score_offset() + antiDiagMin);

	// antiDiag2 going right, first computation of antiDiag3 is going down
}

void
LoganPhase2(LoganState& state)
{
	myLog("Phase2");

	while(state.hoffset < state.hlength && state.voffset < state.vlength)
	{
		// antiDiag1F (final)
		// POST-IT: -1 for a match and 0 for a mismatch
		vectorType match = cmpeqOp(state.get_vqueryh(), state.get_vqueryv());
		match = blendvOp(state.get_vmismatchCost(), state.get_vmatchCost(), match);
		vectorType antiDiag1F = addOp(match, state.get_antiDiag1());

		// antiDiag2S (shift)
		// TODO: vectorType not vector_union_t;
		// redo left/right shift to take and return vectorType
		vector_union_t antiDiag2S = shiftLeft(state.get_antiDiag2());

		// antiDiag2M (pairwise max)
		vectorType antiDiag2M = maxOp(antiDiag2S.simd, state.get_antiDiag2());

		// antiDiag2F (final)
		vectorType antiDiag2F = addOp(antiDiag2M, state.get_vgapCost());

		// Compute antiDiag3
		state.set_antiDiag3(maxOp(antiDiag1F, antiDiag2F));

		// we need to have always antiDiag3 left-aligned
		state.update_antiDiag3(LOGICALWIDTH, NINF);

		// TODO: x-drop termination
		// Note: Don't need to check x drop every time
		// Create custom max_element that also returns position to save computation
		int8_t  antiDiagBest = *std::max_element( state.antiDiag3.elem, state.antiDiag3.elem + VECTORWIDTH );
		state.set_curr_score(antiDiagBest + state.get_score_offset());
		// int64_t current_best_score = antiDiagBest + state.get_score_offset();
		int64_t scoreThreshold = state.get_best_score() - state.get_score_dropoff();

		if ( state.get_curr_score() < scoreThreshold )
		{
			state.xDropCond = true;
			return; // GG: it's a void function and the values are saved in LoganState object
		}

		if (antiDiagBest > CUTOFF)
		{
			int8_t min = *std::min_element(state.antiDiag3.elem, state.antiDiag3.elem + LOGICALWIDTH);
			state.set_antiDiag2(subOp(state.get_antiDiag2(), setOp(min)));
			state.set_antiDiag3(subOp(state.get_antiDiag3(), setOp(min)));
			state.set_score_offset(state.get_score_offset() + min);
		}

		// update best
		if (state.get_curr_score() > state.get_best_score())
			state.set_best_score(state.get_curr_score());

		// TODO : optimize this
		int maxpos, max = 0;
		for (int i = 0; i < VECTORWIDTH; ++i)
		{
			if (state.antiDiag3.elem[i] > max)
			{
				maxpos = i;
				max = state.antiDiag3.elem[i];
			}
		}

		if (maxpos > MIDDLE)
			state.moveRight();
		else
			state.moveDown();
	}

	// TODO: check here
	setBeginPositionH(state.seed, 0);
	setBeginPositionV(state.seed, 0);
	// TODO: check here
	setEndPositionH(state.seed, state.hoffset);
	setEndPositionV(state.seed, state.voffset);
}

void
LoganPhase4(LoganState& state)
{
	myLog("Phase4");

	int dir = state.hoffset >= state.hlength ? goDOWN : goRIGHT;

	for (int i = 0; i < (LOGICALWIDTH - 3); i++)
	{
		// antiDiag1F (final)
		// POST-IT: -1 for a match and 0 for a mismatch
		vectorType match = cmpeqOp(state.get_vqueryh(), state.get_vqueryv());
		match = blendvOp(state.get_vmismatchCost(), state.get_vmatchCost(), match);
		vectorType antiDiag1F = addOp(match, state.get_antiDiag1());

		// antiDiag2S (shift)
		vector_union_t antiDiag2S = shiftLeft(state.get_antiDiag2());

		// antiDiag2M (pairwise max)
		vectorType antiDiag2M = maxOp(antiDiag2S.simd, state.get_antiDiag2());

		// antiDiag2F (final)
		vectorType antiDiag2F = addOp(antiDiag2M, state.get_vgapCost());

		// Compute antiDiag3
		state.set_antiDiag3(maxOp(antiDiag1F, antiDiag2F));

		// we need to have always antiDiag3 left-aligned
		state.update_antiDiag3(LOGICALWIDTH, NINF);

		// TODO: x-drop termination
		// note: Don't need to check x drop every time
		// Create custom max_element that also returns position to save computation
		int8_t antiDiagBest = *std::max_element(state.antiDiag3.elem, state.antiDiag3.elem + VECTORWIDTH);
		state.set_curr_score(antiDiagBest + state.get_score_offset());

		// int64_t current_best_score = antiDiagBest + state.get_score_offset();
		int64_t scoreThreshold = state.get_best_score() - state.get_score_dropoff();

		if (state.get_curr_score() < scoreThreshold)
		{
			state.xDropCond = true;
			return; // GG: it's a void function and the values are saved in LoganState object
		}

		if (antiDiagBest > CUTOFF)
		{
			int8_t min = *std::min_element(state.antiDiag3.elem, state.antiDiag3.elem + LOGICALWIDTH);
			state.set_antiDiag2(subOp(state.get_antiDiag2(), setOp(min)));
			state.set_antiDiag3(subOp(state.get_antiDiag3(), setOp(min)));
			state.set_score_offset(state.get_score_offset() + min);
		}

		// update best
		if (state.get_curr_score() > state.get_best_score())
			state.set_best_score(state.get_curr_score());

		// antiDiag swap, offset updates, and new base load
		short nextDir = dir ^ 1;

		if (nextDir == goRIGHT)
			state.moveRight();
		else
			state.moveDown();

		// direction update
		dir = nextDir;
	}

	// TODO: check here
	setBeginPositionH(state.seed, 0);
	setBeginPositionV(state.seed, 0);
	// TODO: check here
	setEndPositionH(state.seed, state.hoffset);
	setEndPositionV(state.seed, state.voffset);
}

//======================================================================================
// X-DROP ADAPTIVE BANDED ALIGNMENT
//======================================================================================

void operator+=(LoganState& state1, const LoganState& state2)
{
	state1.bestScore = state1.bestScore + state2.bestScore;
	state1.currScore = state1.currScore + state2.currScore;
}

void
LoganOneDirection (LoganState& state) {

	// PHASE 1 (initial values load using dynamic programming)
	LoganPhase1(state);
	if(state.xDropCond) return;

	// PHASE 2 (core vectorized computation)
	LoganPhase2(state);
	if(state.xDropCond) return;

	// PHASE 3 (align on one edge)
	// GG: Phase3 removed to read to code easily (can be recovered from simd/ folder or older commits)

	// PHASE 4 (reaching end of sequences)
	LoganPhase4(state);
	if(state.xDropCond) return;
}

std::pair<int, int>
LoganXDrop
(
	SeedL& seed,
	ExtDirectionL direction,
	std::string const& target,
	std::string const& query,
	ScoringSchemeL& scoringScheme,
	int const &scoreDropOff
)
{
	// TODO: add check scoring scheme correctness/input parameters

	if (direction == LOGAN_EXTEND_LEFT)
	{
		SeedL _seed = seed; // need temporary datastruct

		std::string targetPrefix = target.substr (0, getEndPositionH(seed));	// from read start til start seed (seed included)
		std::string queryPrefix  = query.substr  (0, getEndPositionV(seed));	// from read start til start seed (seed included)
		std::reverse (targetPrefix.begin(), targetPrefix.end());
		std::reverse (queryPrefix.begin(),  queryPrefix.end());

		LoganState result (_seed, targetPrefix, queryPrefix, scoringScheme, scoreDropOff);
		LoganOneDirection (result);

		setBeginPositionH(result.seed, getEndPositionH(seed) - getEndPositionH(result.seed));
		setBeginPositionV(result.seed, getEndPositionV(seed) - getEndPositionV(result.seed));

		return std::make_pair(result.get_best_score(), result.get_curr_score());
	}
	else if (direction == LOGAN_EXTEND_RIGHT)
	{
		SeedL _seed = seed; // need temporary datastruct

		std::string targetSuffix = target.substr (getBeginPositionH(seed), target.length()); 	// from end seed until the end (seed included)
		std::string querySuffix  = query.substr  (getBeginPositionV(seed), query.length());		// from end seed until the end (seed included)

		LoganState result (_seed, targetSuffix, querySuffix, scoringScheme, scoreDropOff);
		LoganOneDirection (result);

		setEndPositionH (result.seed, getBeginPositionH(seed) + getEndPositionH(result.seed));
		setEndPositionV (result.seed, getBeginPositionV(seed) + getEndPositionV(result.seed));

		return std::make_pair(result.get_best_score(), result.get_curr_score());
	}
	else
	{
		SeedL _seed1 = seed; // need temporary datastruct
		SeedL _seed2 = seed; // need temporary datastruct

		std::string targetPrefix = target.substr (0, getEndPositionH(seed));	// from read start til start seed (seed not included)
		std::string queryPrefix  = query.substr  (0, getEndPositionV(seed));	// from read start til start seed (seed not included)

		std::reverse (targetPrefix.begin(), targetPrefix.end());
		std::reverse (queryPrefix.begin(),  queryPrefix.end());

		LoganState result1(_seed1, targetPrefix, queryPrefix, scoringScheme, scoreDropOff);
		LoganOneDirection (result1);

		std::string targetSuffix = target.substr (getEndPositionH(seed), target.length()); 	// from end seed until the end (seed included)
		std::string querySuffix  = query.substr  (getEndPositionV(seed), query.length());	// from end seed until the end (seed included)

		LoganState result2(_seed2, targetSuffix, querySuffix, scoringScheme, scoreDropOff);
		LoganOneDirection (result2);

		setBeginPositionH (result1.seed, getEndPositionH(seed) - getEndPositionH(result1.seed));
		setBeginPositionV (result1.seed, getEndPositionV(seed) - getEndPositionV(result1.seed));

		setEndPositionH (result1.seed, getEndPositionH(seed) + getEndPositionH(result2.seed));
		setEndPositionV (result1.seed, getEndPositionV(seed) + getEndPositionV(result2.seed));

		// seed already updated and saved in result1
		// this operation sums up best and exit scores for result1 and result2 and stores them in result1
		result1 += result2;
		return std::make_pair(result1.get_best_score(), result1.get_curr_score());
	}
}