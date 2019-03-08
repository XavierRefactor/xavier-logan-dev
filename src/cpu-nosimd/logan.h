//==================================================================
// Title:  C++ x-drop seed-and-extend alignment algorithm
// Author: G. Guidi, A. Zeni
// Date:   6 March 2019
//==================================================================

#include<string.h>

typedef struct
{
	int beginPositionH;
	int beginPositionV;
	int endPositionH;
	int endPositionV;
	int seedLength;
	int lowerDiagonal;  // GGGG: it might possibly be a std::string
	int upperDiagonal;  // GGGG: it might possibly be a std::string
	int score;

    Seed() : beginPositionH(0), beginPositionV(0), endPositionH(0), endPositionV(0), lowerDiagonal(0), upperDiagonal(0),
             score(0)
    {}

    Seed(int beginPositionH, int beginPositionV, int seedLength) :
            beginPositionH(beginPositionH), beginPositionV(beginPositionV), endPositionH(beginPositionH + seedLength),
            endPositionV(beginPositionV + seedLength), lowerDiagonal((beginPositionH - beginPositionV)),
            upperDiagonal((beginPositionH - beginPositionV)), score(0)
    {
        assert(upperDiagonal >= lowerDiagonal);
    }

      Seed(int beginPositionH, int beginPositionV, int endPositionH, int endPositionV) :
            beginPositionH(beginPositionH),
            beginPositionV(beginPositionV),
            endPositionH(endPositionH),
            endPositionV(endPositionV),
            lowerDiagonal(std::min((beginPositionH - beginPositionV), (endPositionH - endPositionV))),
            upperDiagonal(std::max((beginPositionH - beginPositionV), (endPositionH - endPositionV))),
            score(0)
    {
        assert(upperDiagonal >= lowerDiagonal);
    }

    Seed(Seed const& other) :
              beginPositionH(beginPositionH(other)),
              beginPositionV(beginPositionV(other)),
              endPositionH(endPositionH(other)),
              endPositionV(endPositionV(other)),
              lowerDiagonal(lowerDiagonal(other)),
              upperDiagonal(upperDiagonal(other)),
              score(0)
    {
        assert(upperDiagonal >= lowerDiagonal);
    }

} Seed;

typedef struct 
{
	Seed myseed;
	int score; 			// alignment score
	int length;			// overlap length / max extension

	Result() : Seed(), score(0), length(0)
	{}

	Result(int kmerLen) : Seed(), score(0), length(kmerLen)
	{}

} Result;

// GGGG we can think about this later
//int getAlignScore() 	{ return score; }
//int getAlignLength()	{ return length; }
//int getAlignBegpT() 	{ return seed.begpT; }
//int getAlignBegpQ() 	{ return seed.begpQ; }
//int getAlignEndpT() 	{ return seed.endpT; }
//int getAlignEndpQ() 	{ return seed.endpQ; }
