//==================================================================
// Title:  C++ x-drop seed-and-extend alignment algorithm
// Author: G. Guidi, A. Zeni
// Date:   6 March 2019
//==================================================================

#include<algorithm> 
#include<cassert>

template<typename Tx_>
const Tx_&  min(const Tx_& _Left, const Tx_& Right_)
{   // return smaller of _Left and Right_
    if (_Left < Right_)
        return _Left;
    else
        return Right_;
}

template<typename Tx_, typename Ty_>
Tx_  min(const Tx_& _Left, const Ty_& Right_)
{   // return smaller of _Left and Right_
    return (Right_ < _Left ? Right_ : _Left);
}
template<typename Ty_>
Ty_ const &
max(const Ty_& _Left, const Ty_& Right_)
{   // return larger of _Left and Right_
    if (_Left < Right_)
        return Right_;
    else
        return _Left;
}

template<typename Tx_, typename Ty_>
Tx_
max(const Tx_& _Left, const Ty_& Right_)
{   // return smaller of _Left and Right_
    return (Right_ < _Left ? _Left : Right_);
}


//GPU vector
template<typename T>
class gpuVector
{
private:
    T* m_begin;
    T* m_end;

    size_t capacity;
    size_t length;
    void expand() {
        capacity *= 2;
        size_t tempLength = (m_end - m_begin);
        T* tempBegin = new T[capacity];

        memcpy(tempBegin, m_begin, tempLength * sizeof(T));
        delete[] m_begin;
        m_begin = tempBegin;
        m_end = m_begin + tempLength;
        length = static_cast<size_t>(m_end - m_begin);
    }
public:
    explicit gpuVector() : length(0), capacity(16) {
        m_begin = new T[capacity];
        m_end = m_begin;
    }
    T& operator[] (unsigned int index) {
        return *(m_begin + index);//*(begin+index)
    }
    T* begin() {
        return m_begin;
    }
    T* end() {
        return m_end;
    }
    ~gpuVector()
    {
        delete[] m_begin;
        m_begin = nullptr;
    }

    void add(T t) {

        if ((m_end - m_begin) >= capacity) {
            expand();
        }

        new (m_end) T(t);
        m_end++;
        length++;
    }
    T pop() {
        T endElement = (*m_end);
        delete m_end;
        m_end--;
        return endElement;
    }

    size_t size() {
        return length;
    }

    void resize(size_t n){
    	capacity=n;
    	T* tempBegin = new T[capacity];
    	delete[] m_begin;
    	m_begin = tempBegin;
        m_end = m_begin + n;
    	length = static_cast<size_t>(m_end - m_begin);
    }

};


struct SeedL
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

	SeedL(): beginPositionH(0), beginPositionV(0), endPositionH(0), endPositionV(0), lowerDiagonal(0), upperDiagonal(0), score(0)
	{}

	SeedL(int beginPositionH, int beginPositionV, int seedLength):
		beginPositionH(beginPositionH), beginPositionV(beginPositionV), endPositionH(beginPositionH + seedLength),
		endPositionV(beginPositionV + seedLength), lowerDiagonal((beginPositionH - beginPositionV)),
		upperDiagonal((beginPositionH - beginPositionV)), beginDiagonal(beginPositionH - beginPositionV),
		endDiagonal(endPositionH - endPositionV), score(0)
	{
		assert(upperDiagonal >= lowerDiagonal);
	}

	SeedL(int beginPositionH, int beginPositionV, int endPositionH, int endPositionV):
		beginPositionH(beginPositionH),
		beginPositionV(beginPositionV),
		endPositionH(endPositionH),
		endPositionV(endPositionV),
		lowerDiagonal(min((beginPositionH - beginPositionV), (endPositionH - endPositionV))),
		upperDiagonal(max((beginPositionH - beginPositionV), (endPositionH - endPositionV))),
		beginDiagonal((beginPositionH - beginPositionV)),
		endDiagonal((endPositionH - endPositionV)),
		score(0)
	{
		assert(upperDiagonal >= lowerDiagonal);
	}

	SeedL(SeedL const& other):
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
	SeedL myseed;
	int score; 			// alignment score
	int length;			// overlap length / max extension

	Result() : score(0), length(0)//check
	{
		//myseed=SeedL();
	}

	Result(int kmerLen) : score(0), length(kmerLen)
	{
		//myseed=SeedL();
	}

};

// GGGG we can think about this later
// AAAA add setter also

int
getAlignScore(SeedL const &myseed){
	return myseed.score;
}

int
getBeginPositionH(SeedL const &myseed){
	return myseed.beginPositionH;
}

int
getBeginPositionV(SeedL const &myseed){
	return myseed.beginPositionV;
}

int
getEndPositionH(SeedL const &myseed){
	return myseed.endPositionH;
}

int
getEndPositionV(SeedL const &myseed){
	return myseed.endPositionV;
}

int
getSeedLLength(SeedL const &myseed){
	return myseed.seedLength;
}

int
getLowerDiagonal(SeedL const &myseed){
	return myseed.lowerDiagonal;
}

int
getUpperDiagonal(SeedL const &myseed){
	return myseed.upperDiagonal;
}

int
getBeginDiagonal(SeedL const &myseed){
	return myseed.beginDiagonal;
}

int
getEndDiagonal(SeedL const &myseed){
	return myseed.endDiagonal;
}

void
setAlignScore(SeedL &myseed,int const value){
	myseed.score = value;
}

void
setBeginPositionH(SeedL &myseed,int const value){
	myseed.beginPositionH = value;
}

void
setBeginPositionV(SeedL &myseed,int const value){
	myseed.beginPositionV = value;
}

void
setEndPositionH(SeedL &myseed,int const value){
	myseed.endPositionH = value;
}

void
setEndPositionV(SeedL &myseed,int const value){
	myseed.endPositionV = value;
}

void
setSeedLLength(SeedL &myseed,int const value){
	myseed.seedLength = value;
}

void
setLowerDiagonal(SeedL &myseed,int const value){
	myseed.lowerDiagonal = value;
}

void
setUpperDiagonal(SeedL &myseed,int const value){
	myseed.upperDiagonal = value;
}

void
setBeginDiagonal(SeedL &myseed,int const value){
	myseed.beginDiagonal = value;
}

void
setEndDiagonal(SeedL &myseed,int const value){
	myseed.endDiagonal = value;
}