//==================================================================
// Title:  C++ x-drop seed-and-extend alignment algorithm
// Author: G. Guidi, A. Zeni
// Date:   8 March 2019
//==================================================================

typedef struct 
{
		int match_score;      // match
		int mismatch_score;   // substitution
		int gap_extend_score; // gap extension (indels)
		int gap_open_score;   // gap opening (indels)

		ScoringScheme()
				: match_score(1), mismatch_score(-1), gap_extend_score(-1),
					data_gap_open(-1) {
		}

		// liner gap penalty
		ScoringScheme(int _match, int _mismatch, int _gap)
				: match_score(_match), mismatch_score(_mismatch),
					gap_extend_score(_gap), gap_open_score(_gap) {
		}

		// affine gap penalty
		ScoringScheme(int _match, int _mismatch, int _gap_extend, int _gap_open) 
				: match_score(_match), mismatch_score(_mismatch),
					gap_extend_score(_gap_extend), gap_open_score(_gap_open) {
		}
} ScoringScheme;


inline int
scoreMatch(ScoringScheme const & me) {
	return me.match_score;
}

inline int
scoreMismatch(ScoringScheme const & me) {
	return me.mismatch_score;
}

inline int
scoreGapExtend(ScoringScheme const & me) {
	return me.gap_extend_score;
}

inline int
scoreGapOpen(ScoringScheme const & me) {
	return me.gap_open_score;
}