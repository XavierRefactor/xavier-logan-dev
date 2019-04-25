//==================================================================
// Title:  LOGAN: X-Drop Adaptive Banded Alignment
// Author: G. Guidi, E. Younis
// Date:   23 April 2019
//==================================================================

#include <vector>
#include <iostream>
#include <string>
#include <omp.h>
#include <algorithm>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <iterator>
#include <x86intrin.h>
#include <seqan/sequence.h>
#include <seqan/align.h>
#include <seqan/seeds.h>
#include <seqan/score.h>
#include <seqan/modifier.h>
#include "logan.cpp"
#include "ksw2/ksw2.h"
#include "ksw2/ksw2_extz2_sse.c" // global and extension with SSE intrinsics; Suzuki'
#ifdef __cplusplus
extern "C" {
#endif
#include "libgaba/gaba.h" 		 // sometimes the forefront vector will not reach the end 
								 // of the sequences. It is more likely to occur when the input 
								 // sequence lengths greatly differ
#ifdef __cplusplus
}
#endif
//======================================================================================
// READ SIMULATOR
//======================================================================================

// Logan AVVX2 can achieve at most a score of 32,767
// Future work: remove this limitation
#define LEN1 (32767)	// read length (this is going to be a distribution of length in
						// the adaptive version)
#define LEN2 (200)		// 2nd read length
#define MAT	( 1)		// match score
#define MIS	(-1)		// mismatch score
#define GAP	(-1)		// gap score
#define XDROP (LEN1)	// so high so it won't be triggered in SeqAn
#define LOGAN
#define KSW2
//#define LIBGABA
#define SEQAN

void 
readSimulator (std::string& readh, std::string& readv)
{
	char bases[4] = {'A', 'T', 'C', 'G'}; 

	// read horizontal
	for (int i = 0; i < LEN1; i++)
	{
		int test = rand();
		readh = readh + bases[test % 4];
		readv = readv + bases[test % 4];
	}

	// read vertical
	//for (int i = 0; i < LEN2; i++)
	//{
	//	readv = readv + bases[rand() % 4];
	//}
}

//======================================================================================
// BENCHMARK CODE
//======================================================================================

int main(int argc, char const *argv[])
{
	std::string targetSeg, querySeg;

	// Simulate pair of read
	readSimulator(targetSeg, querySeg);
	//std::cout << targetSeg << "\n" << std::endl;
	//std::cout << querySeg << "\n" << std::endl;

	//======================================================================================
	// LOGAN
	//======================================================================================

#ifdef LOGAN
	ScoringSchemeL scoringSchemeLogan(MAT, MIS, GAP);
	// 1st prototype without seed and x-drop termination
	std::chrono::duration<double> diff1;
	auto start1 = std::chrono::high_resolution_clock::now();
	LoganAVX2(targetSeg, querySeg, scoringSchemeLogan);
	auto end1 = std::chrono::high_resolution_clock::now();
	diff1 = end1-start1;
	// score off by factor of 5
	std::cout << " in " << diff1.count() << " sec " << std::endl;
#endif

	//======================================================================================
	// KSW2 GLOBAL AND EXTENSION, SSE4.1
	//======================================================================================

#ifdef KSW2
	int8_t a = MAT, b = MIS < 0? MIS : -MIS; // a>0 and b<0
	int8_t mat[25] = { a,b,b,b,0, b,a,b,b,0, b,b,a,b,0, b,b,b,a,0, 0,0,0,0,0 };
	int tl = strlen(targetSeg.c_str()), ql = strlen(querySeg.c_str());
	uint8_t *ts, *qs, c[256];
	ksw_extz_t ez;

	memset(&ez, 0, sizeof(ksw_extz_t));
	memset(c, 4, 256);

	// build the encoding table
	c['A'] = c['a'] = 0; c['C'] = c['c'] = 1;
	c['G'] = c['g'] = 2; c['T'] = c['t'] = 3;
	ts = (uint8_t*)malloc(tl);
	qs = (uint8_t*)malloc(ql);

	// encode to 0/1/2/3
	for (int i = 0; i < tl; ++i)
	{
		ts[i] = c[(uint8_t)targetSeg[i]];
	}
	for (int i = 0; i < ql; ++i)
	{
		qs[i] = c[(uint8_t)querySeg[i]];
	}

	std::chrono::duration<double> diff2;
	auto start2 = std::chrono::high_resolution_clock::now();

	ksw_extz2_sse(0, ql, qs, tl, ts, 5, mat, 0, -GAP, -1, -1, 0, KSW_EZ_SCORE_ONLY, &ez);

	auto end2 = std::chrono::high_resolution_clock::now();
	diff2 = end2-start2;

	free(ts); free(qs);

	std::cout << "ksw2's best " << ez.score << " in " << diff2.count() << " sec " << std::endl;
#endif

	//======================================================================================
	// LIBGABA (SegFault)
	//======================================================================================

#ifdef GABA
	gaba_t *ctx = gaba_init(GABA_PARAMS(
		// match award, mismatch penalty, gap open penalty (G_i), and gap extension penalty (G_e)
		GABA_SCORE_SIMPLE(2, 2, -GAP, -GAP),
		gfa : 0,
		gfb : 0,
		xdrop : 100,
		filter_thresh : 0,
	));

	std::chrono::duration<double> diff3;
	auto start3 = std::chrono::high_resolution_clock::now();

	char const t[64] = {0};	// tail array
	gaba_section_t asec = gaba_build_section(0, (uint8_t const *)targetSeg.c_str(), (uint32_t)targetSeg.size());
	gaba_section_t bsec = gaba_build_section(2, (uint8_t const *)querySeg.c_str(),  (uint32_t)querySeg.size());
	gaba_section_t tail = gaba_build_section(4, t, 64);

	// create thread-local object
	gaba_dp_t *dp = gaba_dp_init(ctx);	// dp[0] holds a 64-cell-wide context

	// init section pointers
	gaba_section_t const *ap = &asec, *bp = &bsec;
	gaba_fill_t const *f = gaba_dp_fill_root(dp, // dp -> &dp[_dp_ctx_index(band_width)] makes the band width selectable
		ap, 0,					// a-side (reference side) sequence and start position
		bp, 0,					// b-side (query)
		UINT32_MAX				// max extension length
	);

	// until x-drop condition is detected
	// x-drop has to be within [-127, 127] in libagaba
	gaba_fill_t const *m = f;
	// track max
	while((f->status & GABA_TERM) == 0) {
		// substitute the pointer by the tail section's if it reached the end
		if(f->status & GABA_UPDATE_A) { ap = &tail; }
		if(f->status & GABA_UPDATE_B) { bp = &tail; }

		f = gaba_dp_fill(dp, f, ap, bp, UINT32_MAX);	// extend the banded matrix
		m = f->max > m->max ? f : m;					// swap if maximum score was updated
	}

	// alignment path
	gaba_alignment_t *r = gaba_dp_trace(dp,
		m,		// section with the max
		NULL	// custom allocator: see struct gaba_alloc_s in gaba.h
	);

	auto end3 = std::chrono::high_resolution_clock::now();
	diff3 = end3-start3;

	// clean up
	gaba_dp_res_free(dp, r); gaba_dp_clean(dp);
	gaba_clean(ctx);

	std::cout << "Libgaba's best " << r->score << " in " << diff3.count() << " sec " << std::endl;
#endif

	//======================================================================================
	// SEQAN
	//======================================================================================

#ifdef SEQAN
	// SeqAn
	seqan::Score<int, seqan::Simple> scoringSchemeSeqAn(MAT, MIS, GAP);
	seqan::Seed<seqan::Simple> seed(0, 0, 0);
	std::chrono::duration<double> diff4;
	auto start4 = std::chrono::high_resolution_clock::now();
	int score = seqan::extendSeed(seed, targetSeg, querySeg, seqan::EXTEND_RIGHT, 
		scoringSchemeSeqAn, XDROP, seqan::GappedXDrop(), 0);
	auto end4 = std::chrono::high_resolution_clock::now();
	diff4 = end4-start4;

	// SeqAn is doing more computation
	std::cout << "SeqAn's best " << score << " in " << diff4.count() << " sec " << std::endl;
#endif

	return 0;
}
