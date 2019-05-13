
//========================================================================================================
// Title:  C++ program to assest quality and performance of LOGAN wrt to original SeqAn implementation
// Author: G. Guidi
// Date:   12 March 2019
//========================================================================================================

//#include <omp.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <math.h>
#include <limits.h>
#include <bitset>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <ctype.h> 
#include <sstream>
#include <set>
#include <memory>
#include <typeinfo>
#include <pthread.h>
#include <vector>
// #include <seqan/sequence.h>
// #include <seqan/align.h>
// #include <seqan/seeds.h>
// #include <seqan/score.h>
// #include <seqan/modifier.h>
// #include "../gpu/v0_antidiag/logan-functions.h"
#include "logan-functions.h"
using namespace std;
//using namespace seqan;

#define NOW std::chrono::high_resolution_clock::now()

//=======================================================================
// 
// Common functions
// 
//=======================================================================

//typedef seqan::Seed<seqan::Simple> TSeed;
typedef std::tuple< int, int, int, int, int, double > myinfo;	// score, start seedV, end seedV, start seedH, end seedH, runtime

char dummycomplement (char n)
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

//typedef std::tuple< int, int, int, int, double > myinfo;	// score, start seed, end seed, runtime
// myinfo seqanXdrop(seqan::Dna5String& readV, seqan::Dna5String& readH, int posV, int posH, int mat, int mis, int gap, int kmerLen, int xdrop)
// {

// 	seqan::Score<int, seqan::Simple> scoringScheme(mat, mis, -2, gap);
// 	int score;
// 	myinfo seqanresult;

// 	std::chrono::duration<double>  diff;
// 	TSeed seed(posH, posV, kmerLen);

// 	// perform match extension	
// 	auto start = std::chrono::high_resolution_clock::now();
// 	score = seqan::extendSeed(seed, readH, readV, seqan::EXTEND_BOTH, scoringScheme, xdrop, seqan::GappedXDrop(), kmerLen);
// 	auto end = std::chrono::high_resolution_clock::now();
// 	diff = end-start;

// 	std::cout << "seqan score:\t" << score << "\tseqan time:\t" <<  diff.count() <<std::endl;
// 	//double time = diff.count();
// 	seqanresult = std::make_tuple(score, beginPositionV(seed), endPositionV(seed), beginPositionH(seed), endPositionH(seed), diff.count());
// 	return seqanresult;
// }

// typedef std::tuple< int, int, int, int, double > myinfo;	// score, start seed, end seed, runtime
void loganXdrop(std::vector< std::vector<std::string> > &v, int mat, int mis, int gap, int kmerLen, int xdrop)
{
	
	
	//Result result(kmerLen);
	int n_align = v.size();
	//int result;
	//myinfo loganresult;
	vector<ScoringSchemeL> penalties(n_align);
	vector<int> posV(n_align);
	vector<int> posH(n_align);
	vector<string> seqV(n_align);
	vector<string> seqH(n_align);
	vector<SeedL> seeds(n_align);
	for(int i = 0; i < v.size(); i++){
		ScoringSchemeL tmp_sscheme(mat, mis, -1, gap);
		penalties[i]=tmp_sscheme;
		posV[i]=stoi(v[i][1]); 
		posH[i]=stoi(v[i][3]);		
		seqV[i]=v[i][0];		
		seqH[i]=v[i][2];
		std::string strand = v[i][4];

		if(strand == "c"){
			std::transform(
					std::begin(seqH[i]),
					std::end(seqH[i]),
					std::begin(seqH[i]),
					dummycomplement);
			posH[i] = seqH[i].length()-posH[i]-kmerLen;
		}
		SeedL tmp_seed(posH[i], posV[i], kmerLen);
		seeds[i] = tmp_seed;
	}
		

	std::chrono::duration<double>  diff_l;
	

		// perform match extension	
	auto start_l = NOW;
		// GGGG: double check call function
	extendSeedL(seeds, EXTEND_BOTHL, seqH, seqV, penalties, xdrop, kmerLen);
	auto end_l = NOW;
	diff_l = end_l-start_l;

	cout << "logan time:\t" <<  diff_l.count() <<endl;
	
	
	//loganresult = std::make_tuple(result, getBeginPositionV(seed), getEndPositionV(seed), getBeginPositionH(seed), getEndPositionH(seed), diff_l.count());
	//return loganresult;
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
	int mat = 1, mis = -1, gap = -1;	// GGGG: make these input parameters
	const char* filename =  (char*) malloc(20 * sizeof(char));
	std::string temp = "benchmark.txt"; // GGGG: make filename input parameter
	filename = temp.c_str();
	std::cout << "starting benchmark" << std::endl;
	int maxt = 30;
	
	//setting up the gpu environment
	cudaFree(0);
	// omp_set_nested(1);
//#pragma omp parallel
	{
		//maxt = omp_get_num_threads();
	}

	uint64_t numpair = std::count(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>(), '\n');
	input.seekg(0, std::ios_base::beg);

	vector<std::string> entries;
	vector<std::stringstream> local(maxt);      

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
//#pragma omp parallel for
	vector< vector<string> > v(numpair);
	for(uint64_t i = 0; i < numpair; i++) 
	{
		
		//int ithread = i;//omp_get_thread_num();
		vector<string> temp = split(entries[i], '\t');
		// format: seqV, posV, seqH, posH, strand -- GGGG: generate this input with BELLA
		v[i]=temp;
	}
/*		int posV = stoi(v[1]); 
		int posH = stoi(v[3]);		
		std::string seqV = v[0];		
		std::string seqH = v[2];
		std::string strand = v[4];
	
		if(strand == "c")
		{
			std::transform(
				std::begin(seqH),
				std::end(seqH),
				std::begin(seqH),
			dummycomplement);
			posH = seqH.length()-posH-kmerLen;

			//AAAA change here if using 4 bases and to new type 
			//Dna5String seqHLogan(seqH), seqVLogan(seqV);

			//myinfo seqanresult;
			//myinfo loganresult;

			
			//cout << "seqan ok" << endl;
			loganXdrop(v, mat, mis, gap, kmerLen, xdrop);
			//loganresult = loganXdrop(seqV, seqH, posV, posH, mat, mis, gap, kmerLen, xdrop);
			
			//cout << "logan ok" << endl;
			// GGGG: use a custom data struct instead of tuples 	(readability)
			local[ithread] << i << "\t" << get<0>(seqanresult) << "\t" << get<1>(seqanresult) << "\t" 
				<< get<2>(seqanresult) << "\t" << get<3>(seqanresult) << "\t" << get<4>(seqanresult)
					<< "\t" << get<5>(seqanresult) << "\t" << get<0>(loganresult) << "\t" << get<1>(loganresult) << "\t" << 
						get<2>(loganresult) << "\t" << get<3>(loganresult) << "\t" << get<4>(loganresult) 
							<< "\t" << get<5>(loganresult) << endl;
		}
		else
		{
			//AAAA change here if using 4 bases and to new type 
			//Dna5String seqHLogan(seqH), seqVLogan(seqV);

			//myinfo seqanresult;
			//myinfo loganresult;

			
			//cout << "seqan ok" << endl;
*/
			loganXdrop(v, mat, mis, gap, kmerLen, xdrop);
			//loganresult = loganXdrop(seqV, seqH, posV, posH, mat, mis, gap, kmerLen, xdrop);
			
			//cout << "logan ok" << endl;
			// GGGG: use a custom data struct instead of tuples 	
			/*local[ithread] << i << "\t" << get<0>(seqanresult) << "\t" << get<1>(seqanresult) << "\t" 
				<< get<2>(seqanresult) << "\t" << get<3>(seqanresult) << "\t" << get<4>(seqanresult)
					<< "\t" << get<5>(seqanresult) << "\t" << get<0>(loganresult) << "\t" << get<1>(loganresult) << "\t" << 
						get<2>(loganresult) << "\t" << get<3>(loganresult) << "\t" << get<4>(loganresult) 
							<< "\t" << get<5>(loganresult) << endl;*/
//		}

	
//	}
	// write to a new file 	
	int64_t* bytes = new int64_t[maxt];
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
	for(int i =0; i<maxt; i++)
	//#pragma omp parallel
	{
		int ithread = i; 

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
