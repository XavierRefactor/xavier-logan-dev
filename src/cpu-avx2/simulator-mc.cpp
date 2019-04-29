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
#include "logan.cpp"
#include <seqan/align.h>
#include <seqan/sequence.h>
#include <seqan/align.h>
#include <seqan/seeds.h>
#include <seqan/score.h>
#include <seqan/modifier.h>
#include <seqan/basic.h>
#include <seqan/stream.h>

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

#pragma omp parallel for
	for (int i = 0; i < data.work_array.size(); ++i)
	{
		seed_pair work = data.work_array[i];

		SeedL seed( 0, 0, 0 );
		ScoringSchemeL scoringSchemeLogan(MAT, MIS, GAP);

		LoganAVX2(seed, EXTEND_MC_RIGHT, data.reads[ work.id1 ], data.reads[ work.id2 ], scoringSchemeLogan, xdrop);
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	diff1 = end1-start1;

	cout << diff1.count() << "\tseconds"<< endl;
	cout << data.work_array.size() << "\talignments"<< endl;
	cout << (double)data.work_array.size() / diff1.count() << "\talignments/seconds"<< endl;

	std::chrono::duration<double> diff2;
	auto start2 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
	for (int i = 0; i < data.work_array.size(); ++i)
	{
		seed_pair work = data.work_array[i];
		// SeqAn
		seqan::Score<int, seqan::Simple> scoringSchemeSeqAn(MAT, MIS, GAP);
		seqan::Seed<seqan::Simple> seed1(0, 0, 0);
		std::chrono::duration<double> diff4;
		auto start4 = std::chrono::high_resolution_clock::now();
		int score = seqan::extendSeed(seed1, data.reads[ work.id1 ], data.reads[ work.id2 ], seqan::EXTEND_RIGHT,
			scoringSchemeSeqAn, xdrop, seqan::GappedXDrop(), 0);
	}

	auto end2 = std::chrono::high_resolution_clock::now();
	diff2 = end2-start2;

	cout << diff2.count() << "\tseconds"<< endl;
	cout << data.work_array.size() << "\talignments"<< endl;
	cout << (double)data.work_array.size() / diff2.count() << "\talignments/seconds"<< endl;

	return 0;
}