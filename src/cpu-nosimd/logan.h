//==================================================================
// Title:  C++ x-drop seed-and-extend alignment algorithm
// Author: G. Guidi, A. Zeni
// Date:   6 March 2019
//==================================================================

// -----------------------------------------------------------------
// Function extendSeed                         [GappedXDrop, noSIMD]
// -----------------------------------------------------------------

namespace LOGAN
{
	class Seed { 	
	public: 

		// we might need uint32_t 	
		int begpT, endpT;
		int begpQ, endpQ;
		int seedLen; 		

		// default constructors 	
		Seed(int& beginT, int& beginQ, int& len) 
		{ 
			seedLen = len;

			begpT = beginT; 
			endpT = begpT + seedLen - 1; 	
			begpQ = beginQ; 
			endpQ = begpQ + seedLen - 1; 	

		} 	
		Seed(Seed& myseed) 
		{ 
			begpT = myseed.begpT;
			begpQ = myseed.begpQ; 
			endpT = myseed.endpT;
			endpQ = myseed.endpQ;

			seedLen = myseed.seedLen;
		} 
		// member functions		
		// modify after its initialization	
		void setSeed(int& a, int& b, int& c) { begp = a; endp = b; leng = c; }//needs to be checked I think it's broken
		void setBegpT(int p) {begpT = p}
		void setBegpQ(int p) {begpQ = p}
		void setEndpT(int p) {endpT = p}
		void setEndpQ(int p) {endpQ = p}

		int getBegpT() 	{ return begpT; }
		int getBegpQ() 	{ return begpQ; }
		int getEndpT() 	{ return endpT; }
		int getEndpQ() 	{ return endpQ; }
		int getLength()	{ return seedLen; }

	};

	class Result { 		
	public: 
		// we might need uint32_t 	
		Seed seed;
		int score; 		// extension/alignment score	
		int length;		// extension/alignment length

		// default constructors 	
		Result(Seed& myseed) 
		{ 
			seed(myseed); 	
			score = 0;
			length = 0;
		} 			

		// member functions		
		int getAlignScore() 	{ return score; }
		int getAlignLength()	{ return length; } 		// seed.length should be == lenght after extending seed extension 		
		int getAlignBegpT() 	{ return seed.begpT; }
		int getAlignBegpQ() 	{ return seed.begpQ; }
		int getAlignEndpT() 	{ return seed.endpT; }
		int getAlignEndpQ() 	{ return seed.endpQ; }
	};
}

