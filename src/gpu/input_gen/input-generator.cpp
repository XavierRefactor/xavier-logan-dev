
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <time.h> 

using namespace std;

int main(int argc, char const *argv[])
{
	srand(time(NULL));

  	int nseq = stoi(argv[1]);
  	int minlen = stoi(argv[2]);
  	int maxlen = stoi(argv[3]);
	char dictionary[] = {'A','C','G','T'};
	vector<string> sequences;
	for(int i = 0; i < nseq; i++){
		string tmp;
		int len1 = rand()%(maxlen-minlen) + minlen;
		int seed1 = rand()%(len1/2)+len1/4;
		int len2 = rand()%(maxlen-minlen) + minlen;
		int seed2 = rand()%(len2/2)+len2/4;
		for(int j=0; j < len1; j++)
			tmp+=dictionary[rand()%4];
		tmp.append("\t");
		tmp+=to_string(seed1);
		tmp.append("\t");
		for(int j=0; j < len2; j++)
			tmp+=dictionary[rand()%4];
		tmp.append("\t");
		tmp+=to_string(seed2);
		tmp.append("\t");
		tmp += (rand()%2==0) ? "c":"n";
		tmp.append("\n");
		sequences.push_back(tmp);
	}
	ofstream myfile;
	myfile.open ("input.txt");

	for(int i = 0; i < nseq; i++){
		myfile << sequences[i];
	}
	myfile.close();
	return 0;

}
