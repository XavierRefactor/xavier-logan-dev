//==================================================================
// Title:  LOGAN: X-Drop Adaptive Banded Alignment
// Author: G. Guidi, E. Younis
// Date:   22 April 2019
//==================================================================

#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <vector>
#include <sstream>
#include <omp.h>
#include <chrono>
#include "logan.h"
#include <seqan/align.h>
#include <seqan/sequence.h>
#include <seqan/align.h>
#include <seqan/seeds.h>
#include <seqan/score.h>
#include <seqan/modifier.h>
#include <seqan/basic.h>
#include <seqan/stream.h>
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

#define MAT		( 1)		// match score
#define MIS		(-1)		// mismatch score
#define GAP		(-1)		// gap score

using namespace std;

// This is the basic unit of work
struct seed_pair
{
	uint32_t id1;
	uint32_t id2;
	uint32_t seed1;
	uint32_t seed2;
};

struct read_data
{
	uint32_t          num_reads;
	uint32_t          kmer_len;
	vector<string>    reads;
	vector<seed_pair> work_array;
};

read_data read_file (string filename)
{
	ifstream file(filename);
	read_data data;

	file >> data.num_reads;
	file >> data.kmer_len;

	data.reads.resize(data.num_reads);

	for (int i = 0; i < data.num_reads; ++i)
	{
		uint32_t id;
		string   sequence;

		file >> id;
		file >> sequence;

		data.reads[ id ] = sequence;
	}

	while (!file.eof())
	{
		seed_pair spair;

		file >> spair.id1;
		file >> spair.id2;
		file >> spair.seed1;
		file >> spair.seed2;

		data.work_array.push_back( spair );
	}

	return data;
}

int main(int argc, char const *argv[])
{
	read_data data = read_file( "input-logan-small.txt" );

	std::chrono::duration<double> diff1;
	auto start1 = std::chrono::high_resolution_clock::now();

	int xdrop = stoi(argv[1]);
	//std::cout << "Logan" << std::endl;

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < data.work_array.size(); ++i)
	{
		seed_pair work = data.work_array[i];

		SeedL seed( 0, 0, 0 );
		ScoringSchemeL scoringSchemeLogan(MAT, MIS, GAP);

		LoganXDrop(seed, LOGAN_EXTEND_RIGHT, data.reads[ work.id1 ], data.reads[ work.id2 ], scoringSchemeLogan, xdrop);
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	diff1 = end1-start1;
	cout << endl;
	cout << diff1.count() << "\tseconds"<< endl;
	cout << data.work_array.size() << "\talignments"<< endl;
	cout << (double)data.work_array.size() / diff1.count() << "\talignments/seconds"<< endl;
//	std::cout << "SeqAn" << std::endl;
//	std::chrono::duration<double> diff2;
//	auto start2 = std::chrono::high_resolution_clock::now();

//#pragma omp parallel for schedule(dynamic)
//	for (int i = 0; i < data.work_array.size(); ++i)
//	{
//		seed_pair work = data.work_array[i];
//		// SeqAn
//		seqan::Score<int, seqan::Simple> scoringSchemeSeqAn(MAT, MIS, GAP);
//		seqan::Seed<seqan::Simple> seed1(0, 0, 0);
//		std::chrono::duration<double> diff4;
//		auto start4 = std::chrono::high_resolution_clock::now();
//		int score = seqan::extendSeed(seed1, data.reads[ work.id1 ], data.reads[ work.id2 ], seqan::EXTEND_RIGHT,
//			scoringSchemeSeqAn, xdrop, seqan::GappedXDrop(), 0);
//	}
//
//	auto end2 = std::chrono::high_resolution_clock::now();
//	diff2 = end2-start2;
//
//	cout << diff2.count() << "\tseconds"<< endl;
//	cout << data.work_array.size() << "\talignments"<< endl;
//	cout << (double)data.work_array.size() / diff2.count() << "\talignments/seconds"<< endl;
//
//	std::cout << "ksw2" << std::endl;
	std::chrono::duration<double> diff3;
	auto start3 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < data.work_array.size(); ++i)
	{
		seed_pair work = data.work_array[i];
		// SeqAn
		int8_t a = MAT, b = MIS < 0? MIS : -MIS; // a>0 and b<0
		int8_t mat[25] = { a,b,b,b,0, b,a,b,b,0, b,b,a,b,0, b,b,b,a,0, 0,0,0,0,0 };
		int tl = strlen(data.reads[ work.id1 ].c_str()), ql = strlen(data.reads[ work.id2 ].c_str());
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
			ts[i] = c[(uint8_t)data.reads[ work.id1 ][i]];
		}
		for (int i = 0; i < ql; ++i)
		{
			qs[i] = c[(uint8_t)data.reads[ work.id2 ][i]];
		}

		ksw_extz2_sse(0, ql, qs, tl, ts, 5, mat, 0, -GAP, xdrop, -1, 0, KSW_EZ_SCORE_ONLY, &ez);

		free(ts); free(qs);
	}

		auto end3 = std::chrono::high_resolution_clock::now();
		diff3 = end3-start3;

		cout << diff3.count() << "\tseconds"<< endl;
		cout << data.work_array.size() << "\talignments"<< endl;
		cout << (double)data.work_array.size() / diff3.count() << "\talignments/seconds"<< endl;

		std::chrono::duration<double> diff4;
		auto start4 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < data.work_array.size(); ++i)
	{
		seed_pair work = data.work_array[i];
		// SeqAn
		gaba_t *ctx = gaba_init(GABA_PARAMS(
			// match award, mismatch penalty, gap open penalty (G_i), and gap extension penalty (G_e)
			GABA_SCORE_SIMPLE(MAT, -MIS, 0, -GAP),
			gfa : 0,
			gfb : 0,
			xdrop : (int8_t)xdrop,
			filter_thresh : 0,
		));

		char const t[64] = {0};	// tail array
		gaba_section_t asec = gaba_build_section(0, (uint8_t const *)data.reads[ work.id1 ].c_str(), (uint32_t)data.reads[ work.id1 ].size());
		gaba_section_t bsec = gaba_build_section(2, (uint8_t const *)data.reads[ work.id2 ].c_str(), (uint32_t)data.reads[ work.id2 ].size());
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

		if(r == NULL)
		{
			std::cout << "Libgaba seg fault with the following pair of sequences:" << std::endl;
			std::cout << data.reads[ work.id1 ] << std::endl;
			std::cout << data.reads[ work.id2 ] << std::endl;
		}

		// clean up
		gaba_dp_res_free(dp, r); gaba_dp_clean(dp);
		gaba_clean(ctx);
	}
		auto end4 = std::chrono::high_resolution_clock::now();
		diff4 = end4-start4;

		cout << diff4.count() << "\tseconds"<< endl;
		cout << data.work_array.size() << "\talignments"<< endl;
		cout << (double)data.work_array.size() / diff4.count() << "\talignments/seconds"<< endl;
		cout << endl;

	return 0;
}