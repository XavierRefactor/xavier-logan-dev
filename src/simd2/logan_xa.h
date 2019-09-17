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
	#define log( var ) do { std::cerr << "LOG: " << __FILE__ << "(" << __LINE__ << ") " << #var << " = " << (var) << std::endl; } while(0)
#else
	#define log( var )
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
			log( "ERROR STATE, Fix this here buddy!" );
			// return std::make_pair(0, 0);

		// Convert from string to int array
		// This is the entire sequences
		queryh = new int8_t[hlength];
		queryv = new int8_t[vlength];
		std::copy(targetSeg.begin(), targetSeg.end(), queryh);
		std::copy(querySeg.begin(), querySeg.end(), queryv);

		// pay attention
		log( "Verify Here (ASSERT)" );

		matchCost    = scoreMatch(scoringScheme   );
		mismatchCost = scoreMismatch(scoringScheme);
		gapCost      = scoreGap(scoringScheme     );

		vmatchCost    = set1_func (matchCost   );
		vmismatchCost = set1_func (mismatchCost);
		vgapCost      = set1_func (gapCost     );
		vzeros        = _mm256_setzero_si256();

		hoffset = LOGICALWIDTH;
		voffset = LOGICALWIDTH;

		bestScore    = 0;
		scoreOffset  = 0;
		scoreDropOff = _scoreDropOff;
	}

	~LoganState()
	{
		delete [] queryh;
		delete [] queryv;
	}

	int64_t get_score_offset  ( void ) { return scoreOffset;  }
	int64_t get_best_score    ( void ) { return bestScore;    }
	int64_t get_score_dropoff ( void ) { return scoreDropOff; }

	void set_score_offset ( int64_t _scoreOffset ) { scoreOffset = _scoreOffset; }
	void set_best_score   ( int64_t _bestScore   ) { bestScore = _bestScore; }

	int8_t get_match_cost    ( void ) { return matchCost;    }
	int8_t get_mismatch_cost ( void ) { return mismatchCost; }
	int8_t get_gap_cost      ( void ) { return gapCost;      }

	vector_t get_vqueryh ( void ) { return vqueryh.simd; }
	vector_t get_vqueryv ( void ) { return vqueryv.simd; }

	vector_t get_antiDiag1 ( void ) { return antiDiag1.simd; }
	vector_t get_antiDiag2 ( void ) { return antiDiag2.simd; }
	vector_t get_antiDiag3 ( void ) { return antiDiag3.simd; }

	vector_t get_vmatchCost    ( void ) { return vmatchCost;    }
	vector_t get_vmismatchCost ( void ) { return vmismatchCost; }
	vector_t get_vgapCost      ( void ) { return vgapCost;      }
	vector_t get_vzeros        ( void ) { return vzeros;        }

	void update_vqueryh ( uint8_t idx, int8_t value ) { vqueryh.elem[idx] = value; }
	void update_vqueryv ( uint8_t idx, int8_t value ) { vqueryv.elem[idx] = value; }

	void update_antiDiag1 ( uint8_t idx, int8_t value ) { antiDiag1.elem[idx] = value; }
	void update_antiDiag2 ( uint8_t idx, int8_t value ) { antiDiag2.elem[idx] = value; }
	void update_antiDiag3 ( uint8_t idx, int8_t value ) { antiDiag3.elem[idx] = value; }

	void broadcast_antiDiag1 ( int8_t value ) { antiDiag1.simd = set1_func( value ); }
	void broadcast_antiDiag2 ( int8_t value ) { antiDiag2.simd = set1_func( value ); }
	void broadcast_antiDiag3 ( int8_t value ) { antiDiag3.simd = set1_func( value ); }

	void set_antiDiag1 ( vector_t vector ) { antiDiag1.simd = vector; }
	void set_antiDiag2 ( vector_t vector ) { antiDiag2.simd = vector; }
	void set_antiDiag3 ( vector_t vector ) { antiDiag3.simd = vector; }

// private:

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
	vector_t vmatchCost;
	vector_t vmismatchCost;
	vector_t vgapCost;
	vector_t vzeros;

	// Computation Vectors
	vector_union_t antiDiag1;
	vector_union_t antiDiag2;
	vector_union_t antiDiag3;

	vector_union_t vqueryh;
	vector_union_t vqueryv;

	// X-Drop Variables
	int64_t bestScore;
	int64_t scoreOffset;
	int64_t scoreDropOff;
};

void 
LoganPhase1 ( LoganState& state )
{
	log( "Phase1" );

	// we need one more space for the off-grid values and one more space for antiDiag2
	// TODO: worry about overflow in phase 1 with int8_t (depends on scoring matrix)
	int8_t dp_matrix[LOGICALWIDTH + 2][LOGICALWIDTH + 2];

	// dp_matrix initialization
	dp_matrix[0][0] = 0;
	for ( int i = 1; i < LOGICALWIDTH + 2; i++ )
	{
		dp_matrix[0][i] = -i;
		dp_matrix[i][0] = -i;
	}

	// dynamic programming loop to fill dp_matrix
	for ( int i = 1; i < LOGICALWIDTH + 2; i++ )
	{
		for ( int j = 1; j < LOGICALWIDTH + 2; j++ )
		{
			int8_t onef = dp_matrix[i-1][j-1];

			if ( state.queryh[i-1] == state.queryv[j-1] )
				onef += state.get_match_cost();
			else
				onef += state.get_mismatch_cost();

			int8_t twof = std::max( dp_matrix[i-1][j], dp_matrix[i][j-1] );
			twof += state.get_gap_cost();

			dp_matrix[i][j] = std::max(onef, twof);
		}
	}

	#ifdef DEBUG // print dp_matrix
		for ( int i = 1; i < LOGICALWIDTH + 2; i++ )
		{
			for ( int j = 1; j < LOGICALWIDTH + 2; j++ )
				std::cout << dp_matrix[i][j] << '\t';
			std::cout << std::endl;
		}
	#endif

	for ( int i = 0; i < LOGICALWIDTH; ++i )
	{
		state.update_vqueryh( i, state.queryh[i + 1] );
		state.update_vqueryv( i, state.queryv[LOGICALWIDTH - i] );
	}

	state.update_vqueryh( LOGICALWIDTH, NINF );
	state.update_vqueryv( LOGICALWIDTH, NINF );

	// load dp_matrix into antiDiag1 and antiDiag2 vector
	for ( int i = 1; i <= LOGICALWIDTH; ++i ) 
	{
		state.update_antiDiag1( i - 1, dp_matrix[i][LOGICALWIDTH - i + 1] );
		state.update_antiDiag2( i, dp_matrix[i + 1][LOGICALWIDTH - i + 1] );
	}
	state.update_antiDiag1( LOGICALWIDTH, NINF );
	state.update_antiDiag2( 0, NINF );

	// Clear antiDiag3
	state.broadcast_antiDiag3( NINF );

	// antiDiag2 going right, first computation of antiDiag3 is going down.
}

void
LoganPhase2 ( LoganState& state )
{
	log( "Phase2" );

	while ( state.hoffset < state.hlength && state.voffset < state.vlength )
	{
		// antiDiag1F (final)
		// POST-IT: -1 for a match and 0 for a mismatch
		vector_t match = cmpeq_func( state.get_vqueryh(), state.get_vqueryv() );
		match = blendv_func( state.get_vmismatchCost(), state.get_vmatchCost(), match );
		vector_t antiDiag1F = add_func( match, state.get_antiDiag1() );

		// antiDiag2S (shift)
		// TODO: vector_t not vector_union_t;
		// redo left/right shift to take and return vector_t
		vector_union_t antiDiag2S = leftShift( state.get_antiDiag2() );

		// antiDiag2M (pairwise max)
		vector_t antiDiag2M = max_func( antiDiag2S.simd, state.get_antiDiag2() );

		// antiDiag2F (final)
		vector_t antiDiag2F = add_func( antiDiag2M, state.get_vgapCost() );

		// Compute antiDiag3
		state.set_antiDiag3( max_func( antiDiag1F, antiDiag2F ) );

		// we need to have always antiDiag3 left-aligned
		state.update_antiDiag3( LOGICALWIDTH, NINF );

		// TODO: x-drop termination
		// Note: Don't need to check x drop every time
		// Create custom max_element that also returns position to save computation
		int8_t  antiDiagBest = *std::max_element( state.antiDiag3.elem, state.antiDiag3.elem + VECTORWIDTH );
		int64_t current_best_score = antiDiagBest + state.get_score_offset();
		int64_t score_threshold = state.get_best_score() - state.get_score_dropoff();

		if ( current_best_score < score_threshold )
			return; // GG: it's a void function and the values are saved in LoganState object

		if ( antiDiagBest > CUTOFF )
		{
			int8_t min = *std::min_element(  state.antiDiag3.elem, state.antiDiag3.elem + LOGICALWIDTH);
			state.set_antiDiag2( sub_func( state.get_antiDiag2(), set1_func( min ) ) );
			state.set_antiDiag3( sub_func( state.get_antiDiag3(), set1_func( min ) ) );
			state.set_score_offset( state.get_score_offset() + min );
		}

		// update best
		if ( current_best_score > state.get_best_score() )
			state.set_best_score( current_best_score );

		// CHECKPOINT

		// antiDiag swap, offset updates, and new base load
		// TODO : optimize this
		int maxpos, max = 0;
		for ( int i = 0; i < VECTORWIDTH; ++i )
		{
			if ( state.antiDiag3.elem[i] > max )
			{
				maxpos = i;
				max = state.antiDiag3.elem[i];
			}
		}

		if( maxpos > MIDDLE )
		{
			moveRight (state.antiDiag1, state.antiDiag2, state.antiDiag3, state.hoffset, state.voffset, state.vqueryh, state.vqueryv, state.queryh, state.queryv);
		}
		else
		{
			moveDown (state.antiDiag1, state.antiDiag2, state.antiDiag3, state.hoffset, state.voffset, state.vqueryh, state.vqueryv, state.queryh, state.queryv);
		}
	}
}

// GG LoganPhase3 not neeeded for BELLA (our highest priority)

void 
LoganPhase4 ( LoganState& state )
{
	log("Phase4");

	int dir = state.hoffset >= state.hlength ? myDOWN : myRIGHT;

	for (int i = 0; i < (LOGICALWIDTH - 3); i++)
	{
		// antiDiag1F (final)
		// POST-IT: -1 for a match and 0 for a mismatch
		vector_t match = cmpeq_func( state.get_vqueryh(), state.get_vqueryv() );
		match = blendv_func( state.get_vmismatchCost(), state.get_vmatchCost(), match );
		vector_t antiDiag1F = add_func( match, state.get_antiDiag1() );

		// antiDiag2S (shift)
		// TODO: vector_t not vector_union_t;
		// redo left/right shift to take and return vector_t
		vector_union_t antiDiag2S = leftShift( state.get_antiDiag2() );

		// antiDiag2M (pairwise max)
		vector_t antiDiag2M = max_func( antiDiag2S.simd, state.get_antiDiag2() );

		// antiDiag2F (final)
		vector_t antiDiag2F = add_func( antiDiag2M, state.get_vgapCost() );

		// Compute antiDiag3
		state.set_antiDiag3( max_func( antiDiag1F, antiDiag2F ) );

		// we need to have always antiDiag3 left-aligned
		state.update_antiDiag3( LOGICALWIDTH, NINF );

		// TODO: x-drop termination
		// Note: Don't need to check x drop every time
		// Create custom max_element that also returns position to save computation
		int8_t  antiDiagBest = *std::max_element( state.antiDiag3.elem, state.antiDiag3.elem + VECTORWIDTH );
		int64_t current_best_score = antiDiagBest + state.get_score_offset();
		int64_t score_threshold = state.get_best_score() - state.get_score_dropoff();

		if ( current_best_score < score_threshold )
			return; // GG: it's a void function and the values are saved in LoganState object

		if ( antiDiagBest > CUTOFF )
		{
			int8_t min = *std::min_element(  state.antiDiag3.elem, state.antiDiag3.elem + LOGICALWIDTH);
			state.set_antiDiag2( sub_func( state.get_antiDiag2(), set1_func( min ) ) );
			state.set_antiDiag3( sub_func( state.get_antiDiag3(), set1_func( min ) ) );
			state.set_score_offset( state.get_score_offset() + min );
		}

		// update best
		if ( current_best_score > state.get_best_score() )
			state.set_best_score( current_best_score );

		// antiDiag swap, offset updates, and new base load
		short nextDir = dir ^ 1;
		// antiDiag swap, offset updates, and new base load
		if (nextDir == myRIGHT)
		{
			moveRight (state.antiDiag1, state.antiDiag2, state.antiDiag3, state.hoffset, 
				state.voffset, state.vqueryh, state.vqueryv, state.queryh, state.queryv);
		}
		else
		{
			moveDown (state.antiDiag1, state.antiDiag2, state.antiDiag3, state.hoffset, 
				state.voffset, state.vqueryh, state.vqueryv, state.queryh, state.queryv);
		}
		// direction update
		dir = nextDir;
	}

	// find positions of longest extension and update seed
	setBeginPositionH(state.seed, 0);
	setBeginPositionV(state.seed, 0);
	// TODO : fix rthis
	setEndPositionH(state.seed, state.hoffset);
	setEndPositionV(state.seed, state.voffset);

}

//======================================================================================
// X-DROP ADAPTIVE BANDED ALIGNMENT
//======================================================================================

std::pair<int64_t, int64_t>
LoganOneDirection
(
	SeedL& seed,
	std::string const& targetSeg,
	std::string const& querySeg,
	ScoringSchemeL& scoringScheme,
	int64_t const &scoreDropOff
)
{
	LoganState state(seed, targetSeg, querySeg, scoringScheme, scoreDropOff);

	//==================================================================
	// PHASE I (initial values load using dynamic programming)
	//==================================================================

	LoganPhase1(state);

	//==================================================================
	// PHASE II (core vectorized computation)
	//==================================================================

	LoganPhase2(state);

	//==================================================================
	// PHASE IV (reaching end of sequences)
	//==================================================================

	LoganPhase4(state);
}

// GG PhaseIII removed to read to code easily (can be recovered from simd/ folder or older commits)

std::pair<int64_t, int64_t>
LoganXDrop
(
	SeedL& seed,
	ExtDirectionL direction,
	std::string const& target,
	std::string const& query,
	ScoringSchemeL& scoringScheme,
	int64_t const &scoreDropOff
)
{
	// TODO: check scoring scheme correctness/input parameters
	if (direction == LOGAN_EXTEND_LEFT)
	{
		std::string targetPrefix = target.substr (0, getEndPositionH(seed));	// from read start til start seed (seed included)
		std::string queryPrefix = query.substr (0, getEndPositionV(seed));		// from read start til start seed (seed included)
		std::reverse (targetPrefix.begin(), targetPrefix.end());
		std::reverse (queryPrefix.begin(), queryPrefix.end());
		return LoganOneDirection (seed, targetPrefix, queryPrefix, scoringScheme, scoreDropOff);
	}
	else if (direction == LOGAN_EXTEND_RIGHT)
	{
		std::string targetSuffix = target.substr (getBeginPositionH(seed), target.length()); 	// from end seed until the end (seed included)
		std::string querySuffix = query.substr (getBeginPositionV(seed), query.length());		// from end seed until the end (seed included)
		return LoganOneDirection (seed, targetSuffix, querySuffix, scoringScheme, scoreDropOff);
	}
	else
	{
		std::pair<int64_t, int64_t> extLeft;
		std::pair<int64_t, int64_t> extRight;

		std::string targetPrefix = target.substr (0, getBeginPositionH(seed));	// from read start til start seed (seed not included)
		std::string queryPrefix = query.substr (0, getBeginPositionV(seed));	// from read start til start seed (seed not included)
		std::reverse (targetPrefix.begin(), targetPrefix.end());
		std::reverse (queryPrefix.begin(), queryPrefix.end());
		extLeft = LoganOneDirection (seed, targetPrefix, queryPrefix, scoringScheme, scoreDropOff);

		std::string targetSuffix = target.substr (getBeginPositionH(seed), target.length()); 	// from end seed until the end (seed included)
		std::string querySuffix = query.substr (getBeginPositionV(seed), query.length());		// from end seed until the end (seed included)
		extRight = LoganOneDirection (seed, targetSuffix, querySuffix, scoringScheme, scoreDropOff);

		return extLeft + extRight;
	}
}
