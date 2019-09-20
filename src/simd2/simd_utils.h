//==================================================================
// Title:  LOGAN: X-Drop Adaptive Banded Alignment
// Author: G. Guidi, E. Younis
// Date:   30 April 2019
//==================================================================

#include <cstdint>

#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

//======================================================================================
// GLOBAL FUNCTION DECLARATION
//======================================================================================
#ifdef  __AVX2__ 	// Compile flag: -mavx2
#define VECTORWIDTH  (32)
#define LOGICALWIDTH (VECTORWIDTH - 1)
#define vectorType    __m256i
#define addOp    	_mm256_adds_epi8  	// saturated arithmetic
#define subOp    	_mm256_subs_epi8  	// saturated arithmetic
#define maxOp    	_mm256_max_epi8   	// max
#define setOp   	_mm256_set1_epi8  	// set1 operation
#define blendvOp	 _mm256_blendv_epi8 // blending operation
#define cmpeqOp 	 _mm256_cmpeq_epi8 	// compare equality operation
#elif __SSE4_2__ 	// Compile flag: -msse4.2
#define VECTORWIDTH  (8)
#define LOGICALWIDTH (VECTORWIDTH - 1)
#define vectorType    __m128i
#define addOp    	_mm_adds_epi16 	// saturated arithmetic
#define subOp    	_mm_subs_epi16  // saturated arithmetic
#define maxOp    	_mm_max_epi16  	// max
#define setOp   	_mm_set1_epi16  // set1 operation
#define blendvOp 	_mm_blendv_epi8 // blending operation
#define cmpeqOp  	_mm_cmpeq_epi16 // compare equality operation
#endif

//======================================================================================
// GLOBAL VARIABLE DEFINITION
//======================================================================================

#define NINF  	(std::numeric_limits<int8_t>::min())
#define goRIGHT (0)
#define goDOWN  (1)
#define MIDDLE 	(LOGICALWIDTH / 2)

#define CUTOFF	(std::numeric_limits<int8_t>::max() - 25)

//======================================================================================
// SIMD UTILS
//======================================================================================

typedef int8_t element_t;

typedef union {
	vectorType  simd;
	element_t 	elem[VECTORWIDTH];
} vector_union_t;

void
print_vector_c(vectorType a) {

	vector_union_t tmp;
	tmp.simd = a;

	printf("{");
	for (int i = 0; i < VECTORWIDTH-1; ++i)
		printf("%c,", tmp.elem[i]);
	printf("%c}\n", tmp.elem[VECTORWIDTH-1]);
}

void
print_vector_d(vectorType a) {

	vector_union_t tmp;
	tmp.simd = a;

	printf("{");
	for (int i = 0; i < VECTORWIDTH-1; ++i)
		printf("%d,", tmp.elem[i]);
	printf("%d}\n", tmp.elem[VECTORWIDTH-1]);
}

enum ExtDirectionL
{
	LOGAN_EXTEND_NONE  = 0,
	LOGAN_EXTEND_LEFT  = 1,
	LOGAN_EXTEND_RIGHT = 2,
	LOGAN_EXTEND_BOTH  = 3
};

template <typename T,typename U>
std::pair<T,U> operator+(const std::pair<T,U> & l,const std::pair<T,U> & r) {
	return {l.first+r.first,l.second+r.second};
}

// TODO: REWRITE shiftLeft/RightShift for int8_t
// TODO: Verify
#ifdef __AVX2__
inline vector_union_t
shiftLeft (const vectorType& _a) { // this work for avx2
	vector_union_t a;
	a.simd = _a;

	vector_union_t b;
	// https://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avx
	b.simd = _mm256_alignr_epi8(_mm256_permute2x128_si256(a.simd, a.simd, _MM_SHUFFLE(2, 0, 0, 1)), a.simd, 1);
	b.elem[VECTORWIDTH - 1] = NINF;
	return b;
}
inline vector_union_t
shiftRight (const vectorType& _a) { // this work for avx2
	vector_union_t a;
	a.simd = _a;

	vector_union_t b;
	// https://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avx
	b.simd = _mm256_alignr_epi8(a.simd, _mm256_permute2x128_si256(a.simd, a.simd, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 1);
	b.elem[0] = NINF;
	return b;
}
#elif __SSE4_2__
inline vector_union_t
shiftLeft (const vectorType& _a) { // this work for avx2
	vector_union_t a;
	a.simd = _a;

	vector_union_t b;
	// https://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avx
	b.simd = _mm256_alignr_epi8(_mm256_permute2x128_si256(a.simd, a.simd, _MM_SHUFFLE(2, 0, 0, 1)), a.simd, 2);
	b.elem[VECTORWIDTH - 1] = NINF;
	return b;
}
inline vector_union_t
shiftRight (const vectorType& _a) { // this work for avx2
	vector_union_t a;
	a.simd = _a;

	vector_union_t b;
	// https://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avx
	b.simd = _mm256_alignr_epi8(a.simd, _mm256_permute2x128_si256(a.simd, a.simd, _MM_SHUFFLE(0, 0, 2, 0)), 16 - 2);
	b.elem[0] = NINF;
	return b;
}
#endif

#endif