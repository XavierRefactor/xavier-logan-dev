#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <vector>


using namespace std;

int main(int argc, char const *argv[])
{
	std::ifstream infile("roofline.txt");
	ofstream myfile;
	//std::vector<std::string> result;
 	//myfile.open ("example.txt");
  	//myfile << "Writing this to a file.\n";
  	//myfile.close();
  	vector<unsigned long long> antidiags(20000);
  	vector<unsigned long long> counter_antidiags(20000);
  	for(int i=0; i<15000; i++){
  		antidiags[i]=0;
  		counter_antidiags[i]=0;
  	}
  	std::string line;
	while (std::getline(infile, line)){
		int index, val;
		std::vector<std::string> result;
		std::istringstream iss(line);
		for(std::string s; iss >> s; )
    		result.push_back(s);
    	index = atoi(result[1].c_str());
    	val = atoi(result[3].c_str());
    	antidiags[index]+=(val);
    	counter_antidiags[index]++;
	}
	for(int i=0; i<15000; i++)
		cout<<i<<","<<antidiags[i]<<","<<counter_antidiags[i]<<endl;
	myfile.close();
  	
	return 0;
}

