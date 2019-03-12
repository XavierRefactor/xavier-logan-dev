
//========================================================================================================
// Title:  C++ program to assest quality and performance of LOGAN wrt to original SeqAn implementation
// Author: G. Guidi
// Date:   12 March 2019
//========================================================================================================

#include <string.h>
#include <stdio.h>
#include <numeric>
#include <string>
#include <chrono>
#include <seqan/sequence.h>
#include <seqan/align.h>
#include <seqan/seeds.h>
#include <seqan/score.h>
#include <seqan/modifier.h>
#include "../cpu-nosimd/logan.h"

using namespace std;
using namespace seqan;

//=======================================================================
// 
// Common functions
// 
//=======================================================================

typedef Seed<Simple> TSeed;
typedef std::tuple< int, int, int, int, double > myinfo;	// score, start seed, end seed, runtime
typedef std::tuple< int, myinfo, myinfo > myresult;			// pair identifier, myinfo seqan, myinfo logan

char complement (char n)
{   
	switch(n)
	{   
	case 'A':
		return 'T';
	case 'T':
		return 'A';
	case 'G':
		return 'C';
	case 'C':
		return 'G';
	}   
	assert(false);
	return ' ';
}

vector<std::string> split (const std::string &s, char delim)
{
	std::vector<std::string> result;
	std::stringstream ss (s);
	std::string item;

	while (std::getline (ss, item, delim))
	{
		result.push_back (item);
	}

	return result;
}

//=======================================================================
// 
// SeqAn and LOGAN function calls
// 
//=======================================================================

// typedef std::tuple< int, int, int, int, double > myinfo;	// score, start seed, end seed, runtime
myinfo seqanXdrop(Dna5String& readV, Dna5String& readH, int posV, int posH, int8_t mat, int8_t mis, int8_t gap int kmerLen, int xdrop)
{

	Score<int, Simple> scoringScheme(mat, mis, gap);
	int score;

	chrono::duration<double> diff;
	TSeed seed(posH, posV, kmerLen);

	// perform match extension	
	auto start = std::chrono::system_clock::now();
	score = extendSeed(seed, readH, readV, EXTEND_BOTH, scoringScheme, xdrop, GappedXDrop());
	auto end = std::chrono::system_clock::now();
	diff = end-start;

	double time = diff.count();
}

// typedef std::tuple< int, int, int, int, double > myinfo;	// score, start seed, end seed, runtime
myinfo loganXdrop(std::string& readV, std::string& readH, int posV, int posH, int8_t mat, int8_t mis, int8_t gap int kmerLen, int xdrop)
{

	ScoringScheme penalties(mat, mis, gap);
	Result result(kmerLen);

	chrono::duration<double> diff;
	Seed seed(posH, posV, kmerLen);

	// perform match extension	
	auto start = std::chrono::system_clock::now();
	result = extendSeed(seed, readH, readV, EXTEND_BOTH, scoringScheme, xdrop, GappedXDrop());
	auto end = std::chrono::system_clock::now();
	diff = end-start;

	double time = diff.count();
}

//=======================================================================
// 
// Function call main
// 
//=======================================================================

int main(int argc, char **argv)
{
	// add optlist library later		
	ifstream input(argv[1]);		// file name with sequences and seed positions
	int kmerLen = atoi(argv[2]);	// kmerLen
	int xdrop = atoi(argv[3]);		// xdrop
	int mat = 1, mis = gap = -1;	// GGGG: make these input parameters
	std::string filename = "benchmark.out"; // GGGG: make this input parameter

	int maxt = 1;
#pragma omp parallel
	{
		maxt = omp_get_num_threads();
	}

	uint64_t numpair = std::count(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>(), '\n');
	input.seekg(0, std::ios_base::beg);

	vector<std::string> entries;
	vector<myresult> local(maxt);		

	/* read input file */
	if(input)
		for (int i = 0; i < numpair; ++i)
		{
			std::string line;
			std::getline(input, line);
			entries.push_back(line);
		}
	input.close();

	// compute pairwise alignments
#pragma omp parallel for
	for(uint64_t i = 0; i < numpair; i++) 
	{
		int ithread = omp_get_thread_num();
		// format: seqV, posV, seqH, posH, strand -- GGGG: generate this input with BELLA
		std::vector<std::string> v = split (entries[i], '\t');

		int posV = stoi(v[1]); 
		int posH = stoi(v[3]);		
		std::string seqV = v[0];		
		std::string seqH = v[2];
		std::string strand = v[4];

		// reverse complement (use horizontal read) if needed
		if(strand == "c")
		{
			std::transform(
				std::begin(seqH),
				std::end(seqH),
				std::begin(seqH),
			complement);
			posH = seqH.length()-posH-kmerLen;

			Dna5String seqHseqan(seqH), seqVseqan(seqV);

			// GGGG: call alignment
			myinfo seqanresult;
			myinfo loganresult;

			seqanresult = seqanXdrop(seqVseqan, seqHseqan, posV, posH, mat, mis, gap, kmerLen, xdrop);
			loganresult = loganXdrop(seqV, seqH, posV, posH, mat, mis, gap, kmerLen, xdrop);

			local[ithread].push_back(std::make_tuple(i, seqanresult, loganresult)); // local thread information
		}
		else
		{
			Dna5String seqHseqan(seqH), seqVseqan(seqV);

			// GGGG: call alignment
			myinfo seqanresult;
			myinfo loganresult;

			seqanresult = seqanXdrop(seqVseqan, seqHseqan, posV, posH, mat, mis, gap, kmerLen, xdrop);
			loganresult = loganXdrop(seqV, seqH, posV, posH, mat, mis, gap, kmerLen, xdrop);

			local[ithread].push_back(std::make_tuple(i, seqanresult, loganresult)); // local thread information
		}
	}

	// write to a new file
	int64_t * bytes = new int64_t[maxt];
	for(int i = 0; i < maxt; ++i)
	{
		local[i].seekg(0, ios::end);
		bytes[i] = local[i].tellg();
		local[i].seekg(0, ios::beg);
	}
	int64_t bytestotal = std::accumulate(bytes, bytes + maxt, static_cast<int64_t>(0));

	std::ofstream output(filename, std::ios::binary | std::ios::app);
#ifdef PRINT
	cout << "Creating or appending to output file with " << (double)bytestotal/(double)(1024 * 1024) << " MB" << endl;
#endif
	output.seekp(bytestotal - 1);
	// this will likely create a sparse file so the actual disks won't spin yet 	
	output.write("", 1); 
	output.close();

	#pragma omp parallel
	{
		int ithread = omp_get_thread_num(); 

		FILE *ffinal;
		// then everyone fills it 	
		if ((ffinal = fopen(filename, "rb+")) == NULL) 
		{
			fprintf(stderr, "File %s failed to open at thread %d\n", filename, ithread);
		}
		int64_t bytesuntil = std::accumulate(bytes, bytes + ithread, static_cast<int64_t>(0));
		fseek (ffinal , bytesuntil , SEEK_SET);
		std::string text = local[ithread].str();
		fwrite(text.c_str(),1, bytes[ithread], ffinal);
		fflush(ffinal);
		fclose(ffinal);
	}

	delete [] bytes;
	return 0;
}
