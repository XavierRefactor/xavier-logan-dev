//==================================================================
// Title:  C++ x-drop seed-and-extend alignment algorithm
// Author: G. Guidi, A. Zeni
// Date:   6 March 2019
//==================================================================

#include<string>
#include <algorithm> 
#include <cassert>

struct Seed
{
	int beginPositionH;
	int beginPositionV;
	int endPositionH;
	int endPositionV;
	int seedLength;
	int lowerDiagonal;  // GGGG: it might possibly be a std::string
	int upperDiagonal;  // GGGG: it might possibly be a std::string
	int beginDiagonal;
	int endDiagonal;
	int score;

	Seed(): beginPositionH(0), beginPositionV(0), endPositionH(0), endPositionV(0), lowerDiagonal(0), upperDiagonal(0), score(0)
	{}

	Seed(int beginPositionH, int beginPositionV, int seedLength):
		beginPositionH(beginPositionH), beginPositionV(beginPositionV), endPositionH(beginPositionH + seedLength),
		endPositionV(beginPositionV + seedLength), lowerDiagonal((beginPositionH - beginPositionV)),
		upperDiagonal((beginPositionH - beginPositionV)), beginDiagonal(beginPositionH - beginPositionV),
		endDiagonal(endPositionH - endPositionV), score(0)
	{
		assert(upperDiagonal >= lowerDiagonal);
	}

	Seed(int beginPositionH, int beginPositionV, int endPositionH, int endPositionV):
		beginPositionH(beginPositionH),
		beginPositionV(beginPositionV),
		endPositionH(endPositionH),
		endPositionV(endPositionV),
		lowerDiagonal(std::min((beginPositionH - beginPositionV), (endPositionH - endPositionV))),
		upperDiagonal(std::max((beginPositionH - beginPositionV), (endPositionH - endPositionV))),
		beginDiagonal((beginPositionH - beginPositionV)),
		endDiagonal((endPositionH - endPositionV)),
		score(0)
	{
		assert(upperDiagonal >= lowerDiagonal);
	}

	Seed(Seed const& other):
		beginPositionH(other.beginPositionH),
		beginPositionV(other.beginPositionV),
		endPositionH(other.endPositionH),
		endPositionV(other.endPositionV),
		lowerDiagonal(other.lowerDiagonal),
		upperDiagonal(other.upperDiagonal),
		beginDiagonal(other.beginDiagonal),
		endDiagonal(other.endDiagonal),
		score(0)
	{
		assert(upperDiagonal >= lowerDiagonal);
	}

};

struct Result
{
	Seed myseed;
	int score; 			// alignment score
	int length;			// overlap length / max extension

	Result() : score(0), length(0)//check
	{
		myseed=Seed();
	}

	Result(int kmerLen) : score(0), length(kmerLen)
	{
		myseed=Seed();
	}

};

// GGGG we can think about this later
// AAAA add setter also

inline int
getAlignScore(Seed const &myseed){
	return myseed.score;
}

inline int
getBeginPositionH(Seed const &myseed){
	return myseed.beginPositionH;
}

inline int
getBeginPositionV(Seed const &myseed){
	return myseed.beginPositionV;
}

inline int
getEndPositionH(Seed const &myseed){
	return myseed.endPositionH;
}

inline int
getEndPositionV(Seed const &myseed){
	return myseed.endPositionV;
}

inline int
getSeedLength(Seed const &myseed){
	return myseed.seedLength;
}

inline int
getLowerDiagonal(Seed const &myseed){
	return myseed.lowerDiagonal;
}

inline int
getUpperDiagonal(Seed const &myseed){
	return myseed.upperDiagonal;
}

inline int
getBeginDiagonal(Seed const &myseed){
	return myseed.beginDiagonal;
}

inline int
getEndDiagonal(Seed const &myseed){
	return myseed.endDiagonal;
}

inline void
setAlignScore(Seed &myseed,int const value){
	myseed.score = value;
}

inline void
setBeginPositionH(Seed &myseed,int const value){
	myseed.beginPositionH = value;
}

inline void
setBeginPositionV(Seed &myseed,int const value){
	myseed.beginPositionV = value;
}

inline void
setEndPositionH(Seed &myseed,int const value){
	myseed.endPositionH = value;
}

inline void
setEndPositionV(Seed &myseed,int const value){
	myseed.endPositionV = value;
}

inline void
setSeedLength(Seed &myseed,int const value){
	myseed.seedLength = value;
}

inline void
setLowerDiagonal(Seed &myseed,int const value){
	myseed.lowerDiagonal = value;
}

inline void
setUpperDiagonal(Seed &myseed,int const value){
	myseed.upperDiagonal = value;
}

inline void
setBeginDiagonal(Seed &myseed,int const value){
	myseed.beginDiagonal = value;
}

inline void
setEndDiagonal(Seed &myseed,int const value){
	myseed.endDiagonal = value;
}

