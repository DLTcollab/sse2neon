#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <math.h>

// SSE2NEONTEST performs a set of 'unit tests' making sure that each SSE call
// provides the output we expect.  If this fires an assert, then something didn't match up.


#ifdef WIN32
#include <xmmintrin.h>
#include <emmintrin.h>

#include <malloc.h>
#include <crtdbg.h>

static inline float roundf(float val)
{    
	return floorf(val + 0.5f);
}

static void* platformAlignedAlloc(size_t size)
{
	return _aligned_malloc(size, 16);	
}

static void platformAlignedFree(void* ptr)
{
	_aligned_free(ptr);
}

#else

#include "SSE2NEON.h"

static void* platformAlignedAlloc(size_t size)
{
	return ::memalign(16, size);
}

static void platformAlignedFree(void* ptr)
{
	::free(ptr);
}

#endif

static float ranf(void)
{
	uint32_t ir = rand()&0x7FFF;
	return (float)ir*(1.0f/32768.0f);
}

static float ranf(float low,float high)
{
	return ranf()*(high-low)+low;
}

void validateInt(__m128i a,int32_t x,int32_t y,int32_t z,int32_t w)
{
	const int32_t *t = (const int32_t *)&a;
	assert( t[3] == x );
	assert( t[2] == y );
	assert( t[1] == z );
	assert( t[0] == w );
}

void validateInt16(__m128i a,int16_t d0,int16_t d1,int16_t d2,int16_t d3,int16_t d4,int16_t d5,int16_t d6,int16_t d7)
{
	const int16_t *t = (const int16_t *)&a;
	assert( t[0] == d0 );
	assert( t[1] == d1 );
	assert( t[2] == d2 );
	assert( t[3] == d3 );
	assert( t[4] == d4 );
	assert( t[5] == d5 );
	assert( t[6] == d6 );
	assert( t[7] == d7 );

}


void validateFloat(__m128 a,float x,float y,float z,float w)
{
	const float *t = (const float *)&a;
	assert( t[3] == x );
	assert( t[2] == y );
	assert( t[1] == z );
	assert( t[0] == w );
}

void validateFloatEpsilon(__m128 a,float x,float y,float z,float w,float epsilon)
{
	const float *t = (const float *)&a;
	float dx = fabsf( t[3] - x );
	float dy = fabsf( t[2] - y);
	float dz = fabsf( t[1] - z );
	float dw = fabsf( t[0] - w );
	assert( dx < epsilon );
	assert( dy < epsilon );
	assert( dz < epsilon );
	assert( dw < epsilon );
}


__m128i test_mm_setzero_si128(void)
{
	__m128i a = _mm_setzero_si128();
	validateInt(a,0,0,0,0);
	return a;
}

__m128 test_mm_setzero_ps(void)
{
	__m128 a = _mm_setzero_ps();
	validateFloat(a,0,0,0,0);
	return a;
}

__m128 test_mm_set1_ps(float w)
{
	__m128 a = _mm_set1_ps(w);
	validateFloat(a,w,w,w,w);
	return a;
}

__m128 test_mm_set_ps(float x,float y,float z,float w)
{
	__m128 a = _mm_set_ps(x,y,z,w);
	validateFloat(a,x,y,z,w);
	return a;
}

__m128i test_mm_set1_epi32(int32_t i)
{
	__m128i a = _mm_set1_epi32(i);
	validateInt(a,i,i,i,i);
	return a;
}

__m128i test_mm_set_epi32(int32_t x,int32_t y,int32_t z,int32_t w)
{
	__m128i a = _mm_set_epi32(x,y,z,w);
	validateInt(a,x,y,z,w);
	return a;
}

void test_mm_store_ps(float *p,float x,float y,float z,float w)
{
	__m128 a = _mm_set_ps(x,y,z,w);
	_mm_store_ps(p,a);
	assert( p[0] == w );
	assert( p[1] == z );
	assert( p[2] == y );
	assert( p[3] == x );
}

void test_mm_store_ps(int32_t *p,int32_t x,int32_t y,int32_t z,int32_t w)
{
	__m128i a = _mm_set_epi32(x,y,z,w);
	_mm_store_ps((float *)p,*(const __m128 *)&a);
	assert( p[0] == w );
	assert( p[1] == z );
	assert( p[2] == y );
	assert( p[3] == x );
}

__m128 test_mm_load1_ps(const float *p)
{
	__m128 a = _mm_load1_ps(p);
	validateFloat(a,p[0],p[0],p[0],p[0]);
	return a;
}

__m128 test_mm_load_ps(const float *p)
{
	__m128 a = _mm_load_ps(p);
	validateFloat(a,p[3],p[2],p[1],p[0]);
	return a;
}

__m128i test_mm_load_ps(const int32_t *p)
{
	__m128 a = _mm_load_ps((const float *)p);
	__m128i ia = *(const __m128i *)&a;
	validateInt(ia,p[3],p[2],p[1],p[0]);
	return ia;
}


//r0 := ~a0 & b0
//r1 := ~a1 & b1
//r2 := ~a2 & b2
//r3 := ~a3 & b3
__m128 test_mm_andnot_ps(const float *_a,const float *_b)
{
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);
	__m128 c = _mm_andnot_ps(a,b);
	// now for the assertion...
	const uint32_t *ia = (const uint32_t *)&a;
	const uint32_t *ib = (const uint32_t *)&b;
	uint32_t r0 = ~ia[0] & ib[0];
	uint32_t r1 = ~ia[1] & ib[1];
	uint32_t r2 = ~ia[2] & ib[2];
	uint32_t r3 = ~ia[3] & ib[3];
	__m128i ret = test_mm_set_epi32(r3,r2,r1,r0);
	validateInt(*(const __m128i *)&c,r3,r2,r1,r0);
	validateInt(ret,r3,r2,r1,r0);
	return *(const __m128 *)&ret;
}

__m128 test_mm_and_ps(const float *_a,const float *_b)
{
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);
	__m128 c = _mm_and_ps(a,b);
	// now for the assertion...
	const uint32_t *ia = (const uint32_t *)&a;
	const uint32_t *ib = (const uint32_t *)&b;
	uint32_t r0 = ia[0] & ib[0];
	uint32_t r1 = ia[1] & ib[1];
	uint32_t r2 = ia[2] & ib[2];
	uint32_t r3 = ia[3] & ib[3];
	__m128i ret = test_mm_set_epi32(r3,r2,r1,r0);
	validateInt(*(const __m128i *)&c,r3,r2,r1,r0);
	validateInt(ret,r3,r2,r1,r0);
	return *(const __m128 *)&ret;
}

__m128 test_mm_or_ps(const float *_a,const float *_b)
{
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);
	__m128 c = _mm_or_ps(a,b);
	// now for the assertion...
	const uint32_t *ia = (const uint32_t *)&a;
	const uint32_t *ib = (const uint32_t *)&b;
	uint32_t r0 = ia[0] | ib[0];
	uint32_t r1 = ia[1] | ib[1];
	uint32_t r2 = ia[2] | ib[2];
	uint32_t r3 = ia[3] | ib[3];
	__m128i ret = test_mm_set_epi32(r3,r2,r1,r0);
	validateInt(*(const __m128i *)&c,r3,r2,r1,r0);
	validateInt(ret,r3,r2,r1,r0);
	return *(const __m128 *)&ret;
}


__m128i test_mm_andnot_si128(const int32_t *_a,const int32_t *_b)
{
	__m128i a = test_mm_load_ps(_a);
	__m128i b = test_mm_load_ps(_b);
	__m128 fc = _mm_andnot_ps(*(const __m128 *)&a,*(const __m128 *)&b);
	__m128i c = *(const __m128i *)&fc;
	// now for the assertion...
	const uint32_t *ia = (const uint32_t *)&a;
	const uint32_t *ib = (const uint32_t *)&b;
	uint32_t r0 = ~ia[0] & ib[0];
	uint32_t r1 = ~ia[1] & ib[1];
	uint32_t r2 = ~ia[2] & ib[2];
	uint32_t r3 = ~ia[3] & ib[3];
	__m128i ret = test_mm_set_epi32(r3,r2,r1,r0);
	validateInt(c,r3,r2,r1,r0);
	validateInt(ret,r3,r2,r1,r0);
	return ret;
}

__m128i test_mm_and_si128(const int32_t *_a,const int32_t *_b)
{
	__m128i a = test_mm_load_ps(_a);
	__m128i b = test_mm_load_ps(_b);
	__m128 fc = _mm_and_ps(*(const __m128 *)&a,*(const __m128 *)&b);
	__m128i c = *(const __m128i *)&fc;
	// now for the assertion...
	const uint32_t *ia = (const uint32_t *)&a;
	const uint32_t *ib = (const uint32_t *)&b;
	uint32_t r0 = ia[0] & ib[0];
	uint32_t r1 = ia[1] & ib[1];
	uint32_t r2 = ia[2] & ib[2];
	uint32_t r3 = ia[3] & ib[3];
	__m128i ret = test_mm_set_epi32(r3,r2,r1,r0);
	validateInt(c,r3,r2,r1,r0);
	validateInt(ret,r3,r2,r1,r0);
	return ret;
}

__m128i test_mm_or_si128(const int32_t *_a,const int32_t *_b)
{
	__m128i a = test_mm_load_ps(_a);
	__m128i b = test_mm_load_ps(_b);
	__m128 fc = _mm_or_ps(*(const __m128 *)&a,*(const __m128 *)&b);
	__m128i c = *(const __m128i *)&fc;
	// now for the assertion...
	const uint32_t *ia = (const uint32_t *)&a;
	const uint32_t *ib = (const uint32_t *)&b;
	uint32_t r0 = ia[0] | ib[0];
	uint32_t r1 = ia[1] | ib[1];
	uint32_t r2 = ia[2] | ib[2];
	uint32_t r3 = ia[3] | ib[3];
	__m128i ret = test_mm_set_epi32(r3,r2,r1,r0);
	validateInt(c,r3,r2,r1,r0);
	validateInt(ret,r3,r2,r1,r0);
	return ret;
}

int test_mm_movemask_ps(const float *p)
{
	int ret = 0;

	const uint32_t *ip = (const uint32_t *)p;
	if ( ip[0] & 0x80000000 )
	{
		ret|=1;
	}
	if ( ip[1] & 0x80000000 )
	{
		ret|=2;
	}
	if ( ip[2] & 0x80000000 )
	{
		ret|=4;
	}
	if ( ip[3] & 0x80000000 )
	{
		ret|=8;
	}
	__m128 a = test_mm_load_ps(p);
	int val = _mm_movemask_ps(a);
	assert( val == ret );
	return ret;
}

__m128 test_mm_shuffle_ps(const float *_a,const float *_b)
{
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);

	__m128 ret = _mm_shuffle_ps(a,b,_MM_SHUFFLE(0,1,2,3));
	validateFloat(ret,_b[0],_b[1],_a[2],_a[3]);

	ret = _mm_shuffle_ps(a,b,_MM_SHUFFLE(3,2,1,0));
	validateFloat(ret,_b[3],_b[2],_a[1],_a[0]);

	ret = _mm_shuffle_ps(a,b,_MM_SHUFFLE(0,0,1,1));
	validateFloat(ret,_b[0],_b[0],_a[1],_a[1]);

	ret = _mm_shuffle_ps(a,b,_MM_SHUFFLE(3,1,0,2));
	validateFloat(ret,_b[3],_b[1],_a[0],_a[2]);


	return ret;
}

int test_mm_movemask_epi8(const int32_t *_a)
{
	__m128i a = test_mm_load_ps(_a);

	const uint8_t *ip = (const uint8_t *)_a;
	int ret = 0;
	uint32_t mask = 1;
	for (uint32_t i=0; i<16; i++)
	{
		if ( ip[i] & 0x80 )
		{
			ret|=mask;
		}
		mask = mask<<1;
	}
	int test = _mm_movemask_epi8(a);
	assert( test == ret );
	return ret;
}

__m128 test_mm_sub_ps(const float *_a,const float *_b)
{
	float dx = _a[0] - _b[0];
	float dy = _a[1] - _b[1];
	float dz = _a[2] - _b[2];
	float dw = _a[3] - _b[3];
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);

	__m128 c = _mm_sub_ps(a,b);
	validateFloat(c,dw,dz,dy,dx);
	return c;
}

__m128i test_mm_sub_epi32(const int32_t *_a,const int32_t *_b)
{
	int32_t dx = _a[0] - _b[0];
	int32_t dy = _a[1] - _b[1];
	int32_t dz = _a[2] - _b[2];
	int32_t dw = _a[3] - _b[3];
	__m128i a = test_mm_load_ps(_a);
	__m128i b = test_mm_load_ps(_b);

	__m128i c = _mm_sub_epi32(a,b);
	validateInt(c,dw,dz,dy,dx);
	return c;
}

__m128 test_mm_add_ps(const float *_a,const float *_b)
{
	float dx = _a[0] + _b[0];
	float dy = _a[1] + _b[1];
	float dz = _a[2] + _b[2];
	float dw = _a[3] + _b[3];
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);

	__m128 c = _mm_add_ps(a,b);
	validateFloat(c,dw,dz,dy,dx);
	return c;
}

__m128i test_mm_add_epi32(const int32_t *_a,const int32_t *_b)
{
	int32_t dx = _a[0] + _b[0];
	int32_t dy = _a[1] + _b[1];
	int32_t dz = _a[2] + _b[2];
	int32_t dw = _a[3] + _b[3];
	__m128i a = test_mm_load_ps(_a);
	__m128i b = test_mm_load_ps(_b);

	__m128i c = _mm_add_epi32(a,b);
	validateInt(c,dw,dz,dy,dx);
	return c;
}

__m128i test_mm_mullo_epi16(const int16_t *_a,const int16_t *_b)
{
	int16_t d0 = _a[0] * _b[0];
	int16_t d1 = _a[1] * _b[1];
	int16_t d2 = _a[2] * _b[2];
	int16_t d3 = _a[3] * _b[3];
	int16_t d4 = _a[4] * _b[4];
	int16_t d5 = _a[5] * _b[5];
	int16_t d6 = _a[6] * _b[6];
	int16_t d7 = _a[7] * _b[7];

	__m128i a = test_mm_load_ps((const int32_t *)_a);
	__m128i b = test_mm_load_ps((const int32_t *)_b);

	__m128i c = _mm_mullo_epi16(a,b);
	validateInt16(c,d0,d1,d2,d3,d4,d5,d6,d7);

	return c;
}

__m128 test_mm_mul_ps(const float *_a,const float *_b)
{
	float dx = _a[0] * _b[0];
	float dy = _a[1] * _b[1];
	float dz = _a[2] * _b[2];
	float dw = _a[3] * _b[3];
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);

	__m128 c = _mm_mul_ps(a,b);
	validateFloat(c,dw,dz,dy,dx);
	return c;
}

__m128 test_mm_rcp_ps(const float *_a)
{
	float dx = 1.0f / _a[0];
	float dy = 1.0f / _a[1];
	float dz = 1.0f / _a[2];
	float dw = 1.0f / _a[3];
	__m128 a = test_mm_load_ps(_a);
	__m128 c = _mm_rcp_ps(a);
	validateFloatEpsilon(c,dw,dz,dy,dx,300.0f);
	return c;
}

__m128 test_mm_max_ps(const float *_a,const float *_b)
{
	float c[4];

	c[0] = _a[0] > _b[0] ? _a[0] : _b[0];
	c[1] = _a[1] > _b[1] ? _a[1] : _b[1];
	c[2] = _a[2] > _b[2] ? _a[2] : _b[2];
	c[3] = _a[3] > _b[3] ? _a[3] : _b[3];

	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);

	__m128 ret = _mm_max_ps(a,b);
	validateFloat(ret,c[3],c[2],c[1],c[0]);
	return ret;
}

__m128 test_mm_min_ps(const float *_a,const float *_b)
{
	float c[4];

	c[0] = _a[0] < _b[0] ? _a[0] : _b[0];
	c[1] = _a[1] < _b[1] ? _a[1] : _b[1];
	c[2] = _a[2] < _b[2] ? _a[2] : _b[2];
	c[3] = _a[3] < _b[3] ? _a[3] : _b[3];

	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);

	__m128 ret = _mm_min_ps(a,b);
	validateFloat(ret,c[3],c[2],c[1],c[0]);
	return ret;
}

__m128i test_mm_min_epi16(const int16_t *_a,const int16_t *_b)
{
	int16_t d0 = _a[0] < _b[0] ? _a[0] : _b[0];
	int16_t d1 = _a[1] < _b[1] ? _a[1] : _b[1];
	int16_t d2 = _a[2] < _b[2] ? _a[2] : _b[2];
	int16_t d3 = _a[3] < _b[3] ? _a[3] : _b[3];
	int16_t d4 = _a[4] < _b[4] ? _a[4] : _b[4];
	int16_t d5 = _a[5] < _b[5] ? _a[5] : _b[5];
	int16_t d6 = _a[6] < _b[6] ? _a[6] : _b[6];
	int16_t d7 = _a[7] < _b[7] ? _a[7] : _b[7];

	__m128i a = test_mm_load_ps((const int32_t *)_a);
	__m128i b = test_mm_load_ps((const int32_t *)_b);

	__m128i c = _mm_min_epi16(a,b);
	validateInt16(c,d0,d1,d2,d3,d4,d5,d6,d7);

	return c;
}

__m128i test_mm_mulhi_epi16(const int16_t *_a,const int16_t *_b)
{
	int16_t d[8];
	for (uint32_t i=0; i<8; i++)
	{
		int32_t m = (int32_t)_a[i]*(int32_t)_b[i];
		d[i] = (int16_t)(m>>16);
	}

	__m128i a = test_mm_load_ps((const int32_t *)_a);
	__m128i b = test_mm_load_ps((const int32_t *)_b);

	__m128i c = _mm_mulhi_epi16(a,b);
	validateInt16(c,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);

	return c;
}

__m128 test_mm_cmplt_ps(const float *_a,const float *_b)
{
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);

	int32_t result[4];
	result[0] = _a[0] < _b[0] ? -1 : 0;
	result[1] = _a[1] < _b[1] ? -1 : 0;
	result[2] = _a[2] < _b[2] ? -1 : 0;
	result[3] = _a[3] < _b[3] ? -1 : 0;

	__m128 ret = _mm_cmplt_ps(a,b);
	__m128i iret = *(const __m128i *)&ret;
	validateInt(iret,result[3],result[2],result[1],result[0]);

	return ret;
}

__m128 test_mm_cmpgt_ps(const float *_a,const float *_b)
{
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);

	int32_t result[4];
	result[0] = _a[0] > _b[0] ? -1 : 0;
	result[1] = _a[1] > _b[1] ? -1 : 0;
	result[2] = _a[2] > _b[2] ? -1 : 0;
	result[3] = _a[3] > _b[3] ? -1 : 0;

	__m128 ret = _mm_cmpgt_ps(a,b);
	__m128i iret = *(const __m128i *)&ret;
	validateInt(iret,result[3],result[2],result[1],result[0]);

	return ret;
}

__m128 test_mm_cmpge_ps(const float *_a,const float *_b)
{
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);

	int32_t result[4];
	result[0] = _a[0] >= _b[0] ? -1 : 0;
	result[1] = _a[1] >= _b[1] ? -1 : 0;
	result[2] = _a[2] >= _b[2] ? -1 : 0;
	result[3] = _a[3] >= _b[3] ? -1 : 0;

	__m128 ret = _mm_cmpge_ps(a,b);
	__m128i iret = *(const __m128i *)&ret;
	validateInt(iret,result[3],result[2],result[1],result[0]);

	return ret;
}

__m128 test_mm_cmple_ps(const float *_a,const float *_b)
{
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);

	int32_t result[4];
	result[0] = _a[0] <= _b[0] ? -1 : 0;
	result[1] = _a[1] <= _b[1] ? -1 : 0;
	result[2] = _a[2] <= _b[2] ? -1 : 0;
	result[3] = _a[3] <= _b[3] ? -1 : 0;

	__m128 ret = _mm_cmple_ps(a,b);
	__m128i iret = *(const __m128i *)&ret;
	validateInt(iret,result[3],result[2],result[1],result[0]);

	return ret;
}

__m128 test_mm_cmpeq_ps(const float *_a,const float *_b)
{
	__m128 a = test_mm_load_ps(_a);
	__m128 b = test_mm_load_ps(_b);

	int32_t result[4];
	result[0] = _a[0] == _b[0] ? -1 : 0;
	result[1] = _a[1] == _b[1] ? -1 : 0;
	result[2] = _a[2] == _b[2] ? -1 : 0;
	result[3] = _a[3] == _b[3] ? -1 : 0;

	__m128 ret = _mm_cmpeq_ps(a,b);
	__m128i iret = *(const __m128i *)&ret;
	validateInt(iret,result[3],result[2],result[1],result[0]);

	return ret;
}


__m128i test_mm_cmplt_epi32(const int32_t *_a,const int32_t *_b)
{
	__m128i a = test_mm_load_ps(_a);
	__m128i b = test_mm_load_ps(_b);

	int32_t result[4];
	result[0] = _a[0] < _b[0] ? -1 : 0;
	result[1] = _a[1] < _b[1] ? -1 : 0;
	result[2] = _a[2] < _b[2] ? -1 : 0;
	result[3] = _a[3] < _b[3] ? -1 : 0;

	__m128i iret = _mm_cmplt_epi32(a,b);
	validateInt(iret,result[3],result[2],result[1],result[0]);

	return iret;
}

__m128i test_mm_cmpgt_epi32(const int32_t *_a,const int32_t *_b)
{
	__m128i a = test_mm_load_ps(_a);
	__m128i b = test_mm_load_ps(_b);

	int32_t result[4];

	result[0] = _a[0] > _b[0] ? -1 : 0;
	result[1] = _a[1] > _b[1] ? -1 : 0;
	result[2] = _a[2] > _b[2] ? -1 : 0;
	result[3] = _a[3] > _b[3] ? -1 : 0;

	__m128i iret = _mm_cmpgt_epi32(a,b);
	validateInt(iret,result[3],result[2],result[1],result[0]);

	return iret;
}

__m128i test_mm_cvttps_epi32(const float *_a)
{
	__m128 a = test_mm_load_ps(_a);
	int32_t trun[4];
	for (uint32_t i=0; i<4; i++)
	{
		trun[i] = (int32_t)_a[i];
	}

	__m128i ret = _mm_cvttps_epi32(a);
	validateInt(ret,trun[3],trun[2],trun[1],trun[0]);

	return ret;
}

__m128 test_mm_cvtepi32_ps(const int32_t *_a)
{
	__m128i a = test_mm_load_ps(_a);
	float trun[4];
	for (uint32_t i=0; i<4; i++)
	{
		trun[i] = (float)_a[i];
	}

	__m128 ret = _mm_cvtepi32_ps(a);
	validateFloat(ret,trun[3],trun[2],trun[1],trun[0]);

	return ret;

}

__m128i test_mm_cvtps_epi32(const float *_a)
{
	__m128 a = test_mm_load_ps(_a);
	int32_t trun[4];
	for (uint32_t i=0; i<4; i++)
	{
		trun[i] = (int32_t)(roundf(_a[i]));
	}

	__m128i ret = _mm_cvtps_epi32(a);
	validateInt(ret,trun[3],trun[2],trun[1],trun[0]);

	return ret;
}


#define MAX_TEST_VALUE 100

void SSE2NEONTEST(void)
{
	float *testFloatPointer1 = (float *)platformAlignedAlloc(sizeof(__m128));
	float *testFloatPointer2 = (float *)platformAlignedAlloc(sizeof(__m128));
	int32_t *testIntPointer1 = (int32_t *)platformAlignedAlloc(sizeof(__m128i));
	int32_t *testIntPointer2 = (int32_t *)platformAlignedAlloc(sizeof(__m128i));


	float testFloats[MAX_TEST_VALUE];
	int32_t testInts[MAX_TEST_VALUE];
	srand(0);
	for (uint32_t i=0; i<MAX_TEST_VALUE; i++)
	{
		testFloats[i] = ranf(-100000,100000);
		testInts[i] = (int32_t)ranf(-100000,100000);
	}

	test_mm_setzero_si128();
	test_mm_setzero_ps();

	for (uint32_t i=0; i<(MAX_TEST_VALUE-8); i++)
	{
		test_mm_set1_ps(testFloats[i]);
		test_mm_set_ps(testFloats[i],testFloats[i+1],testFloats[i+2],testFloats[i+3]);
		test_mm_store_ps(testFloatPointer1,testFloats[i],testFloats[i+1],testFloats[i+2],testFloats[i+3]);
		test_mm_store_ps(testFloatPointer2,testFloats[i+4],testFloats[i+5],testFloats[i+6],testFloats[i+7]);
		test_mm_load1_ps(testFloatPointer1);
		test_mm_load_ps(testFloatPointer1);
		test_mm_andnot_ps(testFloatPointer1,testFloatPointer2);
		test_mm_and_ps(testFloatPointer1,testFloatPointer2);
		test_mm_or_ps(testFloatPointer1,testFloatPointer2);
		test_mm_movemask_ps(testFloatPointer1);
		test_mm_shuffle_ps(testFloatPointer1,testFloatPointer2);
		test_mm_sub_ps(testFloatPointer1,testFloatPointer2);
		test_mm_add_ps(testFloatPointer1,testFloatPointer2);
		test_mm_mul_ps(testFloatPointer1,testFloatPointer2);

		test_mm_max_ps(testFloatPointer1,testFloatPointer2);
		test_mm_min_ps(testFloatPointer1,testFloatPointer2);

		test_mm_cmplt_ps(testFloatPointer1,testFloatPointer2);
		test_mm_cmpgt_ps(testFloatPointer1,testFloatPointer2);

		test_mm_cvttps_epi32(testFloatPointer1);
		test_mm_cvtps_epi32(testFloatPointer1);

		testFloatPointer1[3] = testFloatPointer2[3]; // make sure at least one items if ==
		test_mm_cmpge_ps(testFloatPointer1,testFloatPointer2);
		test_mm_cmple_ps(testFloatPointer1,testFloatPointer2);
		test_mm_cmpeq_ps(testFloatPointer1,testFloatPointer2);

		// Take the reciprocol of the test number to test the reciprocol approximation
		testFloatPointer1[0] = 1.0f / testFloatPointer1[0];
		testFloatPointer1[1] = 1.0f / testFloatPointer1[1];
		testFloatPointer1[2] = 1.0f / testFloatPointer1[2];
		testFloatPointer1[3] = 1.0f / testFloatPointer1[3];
		test_mm_rcp_ps(testFloatPointer1);
	}

	for (uint32_t i=0; i<(MAX_TEST_VALUE-8); i++)
	{
		test_mm_set1_epi32(testInts[i]);
		test_mm_set_epi32(testInts[i],testInts[i+1],testInts[i+2],testInts[i+3]);
		test_mm_store_ps(testIntPointer1,testInts[i],testInts[i+1],testInts[i+2],testInts[i+3]);
		test_mm_store_ps(testIntPointer2,testInts[i+4],testInts[i+5],testInts[i+6],testInts[i+7]);
		test_mm_andnot_si128(testIntPointer1,testIntPointer2);
		test_mm_and_si128(testIntPointer1,testIntPointer2);
		test_mm_or_si128(testIntPointer1,testIntPointer2);
		test_mm_movemask_epi8(testIntPointer1);
		test_mm_sub_epi32(testIntPointer1,testIntPointer2);
		test_mm_add_epi32(testIntPointer1,testIntPointer2);
		test_mm_mullo_epi16((const int16_t *)testIntPointer1,(const int16_t *)testIntPointer2);
		test_mm_min_epi16((const int16_t *)testIntPointer1,(const int16_t *)testIntPointer2);
		test_mm_mulhi_epi16((const int16_t *)testIntPointer1,(const int16_t *)testIntPointer2);
		test_mm_cmplt_epi32(testIntPointer1,testIntPointer2);
		test_mm_cmpgt_epi32(testIntPointer1,testIntPointer2);
		test_mm_cvtepi32_ps(testIntPointer1);
	}

	platformAlignedFree(testFloatPointer1);
	platformAlignedFree(testFloatPointer2);
	platformAlignedFree(testIntPointer1);
	platformAlignedFree(testIntPointer2);

}
