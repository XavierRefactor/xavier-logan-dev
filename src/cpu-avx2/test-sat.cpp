#include<vector>
#include<iostream>
#include<omp.h>
#include<algorithm>
#include<inttypes.h>
#include<assert.h>
#include<iterator>
#include"logan.h"
#include"score.h"
#include <immintrin.h> // For AVX instructions
// #include <bits/stdc++.h>

typedef union {
	__m256i simd;
	int16_t elem[16] = {-1}; // 
} _m256i_16_t;

void print_m256i_16(__m256i a) {

	_m256i_16_t t;
	t.simd = a;

	printf("{%d,%d,%d,%d,%d,%d,%d,%d,"
			"%d,%d,%d,%d,%d,%d,%d,%d}\n",
			t.elem[ 0], t.elem[ 1], t.elem[ 2], t.elem[ 3],
			t.elem[ 4], t.elem[ 5], t.elem[ 6], t.elem[ 7],
			t.elem[ 8], t.elem[ 9], t.elem[10], t.elem[11],
			t.elem[12], t.elem[13], t.elem[14], t.elem[15]
			);
}
// max(-int,x) works! 		
int main(int argc, char const *argv[])
{
	short ninf = -32768;
	short x = -300;

	__m256i a = _mm256_set1_epi16(ninf);
	__m256i b = _mm256_set1_epi16(x);

	__m256i c = _mm256_adds_epi16 (a, b);
	__m256i d = _mm256_subs_epi16 (a, b);
	__m256i e = _mm256_max_epi16 (a, b);

	print_m256i_16(a);
	print_m256i_16(b);
	print_m256i_16(c);
	print_m256i_16(d);
	print_m256i_16(e);

	return 0;
}