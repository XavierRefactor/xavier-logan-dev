#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <vector>
#include <sstream>
#include <omp.h>
#include <chrono>

#include "logan.cpp"

#define LEN1 	(10000)		// read length (this is going to be a distribution of length in
							// the adaptive version)
#define LEN2 	(10050)		// 2nd read length
#define MAT		( 1)		// match score
#define MIS		(-1)		// mismatch score
#define GAP		(-1)		// gap score
#define XDROP 	(21)		// so high so it won't be triggered in SeqAn
#define PMIS 	(0.03)		// substitution probability
#define PGAP 	(0.12)		// insertion/deletion probability
#define BW 		(32)		// bandwidth (the alignment path of the input sequence and the result does not go out of the band)
#define LOGAN

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

read_data read_file ( string filename )
{
	ifstream file( filename );
	read_data data;

	file >> data.num_reads;
	file >> data.kmer_len;

	data.reads.resize( data.num_reads );

	for ( int i = 0; i < data.num_reads; ++i )
	{
		uint32_t id;
		string   sequence;

		file >> id;
		file >> sequence;

		data.reads[ id ] = sequence;
	}

	while ( !file.eof() )
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

int main ( void )
{
	read_data data = read_file( "input-logan-small.txt" );

	std::chrono::duration<double> diff1;
	auto start1 = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for
	for ( int i = 0; i < data.work_array.size(); ++i )
	{
		seed_pair work = data.work_array[i];

		SeedL seed( 0, 0, 0 );
		ScoringSchemeL scoringSchemeLogan( MAT, MIS, GAP );

		LoganAVX2( seed, EXTEND_MC_RIGHT, data.reads[ work.id1 ], data.reads[ work.id2 ], scoringSchemeLogan, XDROP );
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	diff1 = end1-start1;

	cout << diff1.count() << "s" << endl;
	return 0;
}