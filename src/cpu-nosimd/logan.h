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
		void setSeed(int& a, int& b, int& c) { begp = a; endp = b; leng = c; }
		int getBegin() 	{ return begp; }
		int getEnd() 	{ return endp; }
		int getLength()	{ return leng; }

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
		int getScore() 		{ return score; }
		int getOverlap()	{ return length; } 	// seed.length should be == lenght after extending seed extension 		
		int getBegin() 		{ return seed.begp; }
		int getEnd() 		{ return seed.endp; }
	};
}

