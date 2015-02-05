#ifndef SSE2NEON_H
#define SSE2NEON_H

// This header file provides a simple API translation layer
// between SSE intrinsics to their corresponding ARM NEON versions
//
// This header file does not (yet) translate *all* of the SSE intrinsics.
// Since this is in support of a specific porting effort, I have only
// included the intrinsics I needed to get my port to work.
//
// Note, not all methods have yet been converted to NEON
// Those which have not been converted yet are implemented in
// standard C code and produce the expected results.
// The methods still to be converted are designated with a
// 'TODO' marker.
//
// Questions/Comments/Feedback send to: jratcliffscarab@gmail.com
//
// If you want to improve or add to this project, send me an
// email and I will probably approve your access to the depot.
//
// Project is located here:
//
//	https://code.google.com/p/sse2neon/
//
// TipJar: 1PzgWDSyq4pmdAXRH8SPUtta4SWGrt4B1p

#include "arm_neon.h"

/*******************************************************/
/* MACRO for shuffle parameter for _mm_shuffle_ps().   */
/* Argument fp3 is a digit[0123] that represents the fp*/
/* from argument "b" of mm_shuffle_ps that will be     */
/* placed in fp3 of result. fp2 is the same for fp2 in */
/* result. fp1 is a digit[0123] that represents the fp */
/* from argument "a" of mm_shuffle_ps that will be     */
/* places in fp1 of result. fp0 is the same for fp0 of */
/* result                                              */
/*******************************************************/
#define _MM_SHUFFLE(fp3,fp2,fp1,fp0) (((fp3) << 6) | ((fp2) << 4) | \
	((fp1) << 2) | ((fp0)))

typedef float32x4_t __m128;
typedef int32x4_t __m128i;

// ******************************************
// Set/get methods
// ******************************************

// Sets the 128-bit value to zero https://msdn.microsoft.com/en-us/library/vstudio/ys7dw0kh(v=vs.100).aspx
inline __m128i _mm_setzero_si128 ()
{
	return vdupq_n_s32(0);
}

// Clears the four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/vstudio/tk1t2tbz(v=vs.100).aspx
inline __m128 _mm_setzero_ps(void) 
{
	return vdupq_n_f32(0);
}

// Sets the four single-precision, floating-point values to w. https://msdn.microsoft.com/en-us/library/vstudio/2x1se8ha(v=vs.100).aspx
inline __m128 _mm_set1_ps(float _w) 
{
	return vdupq_n_f32(_w);
}

// Sets the four single-precision, floating-point values to the four inputs. https://msdn.microsoft.com/en-us/library/vstudio/afh0zf75(v=vs.100).aspx
inline __m128 _mm_set_ps(float x, float y , float z , float w ) 
{
	float __attribute__ ((aligned (16))) data[4] = { w, z, y, x };
	return vld1q_f32(data);
}

 // Sets the 4 signed 32-bit integer values to i. https://msdn.microsoft.com/en-us/library/vstudio/h4xscxat(v=vs.100).aspx
inline __m128i _mm_set1_epi32 (int _i)
{
	return vdupq_n_s32(_i);
}

// Sets the 4 signed 32-bit integer values. https://msdn.microsoft.com/en-us/library/vstudio/019beekt(v=vs.100).aspx
inline __m128i _mm_set_epi32 (int i0, int i1,int i2, int i3)
{
	int32_t __attribute__ ((aligned (16))) data[4] = { i3, i2, i1, i0 };
	return vld1q_s32(data);
}

// TODO : Need a NEON version
// Stores four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/vstudio/s3h4ay6y(v=vs.100).aspx
inline void _mm_store_ps(float *p, __m128 a ) 
{
	p[0] = a[0];
	p[1] = a[1];
	p[2] = a[2];
	p[3] = a[3];
}

 // Loads a single single-precision, floating-point value, copying it into all four words https://msdn.microsoft.com/en-us/library/vstudio/5cdkf716(v=vs.100).aspx
inline __m128 _mm_load1_ps(const float * p )
{
	return vld1q_dup_f32(p);
}

 // Loads four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/vstudio/zzd50xxt(v=vs.100).aspx
inline __m128 _mm_load_ps(const float * p )
{
	return vld1q_f32(p);
}


// ******************************************
// Logic/Binary operations
// ******************************************

// Computes the bitwise AND-NOT of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/68h7wd02(v=vs.100).aspx
inline __m128 _mm_andnot_ps(__m128 a , __m128 b ) 
{
	return (__m128)vbicq_s32((__m128i)b,(__m128i)a); // *NOTE* argument swap
}

// Computes the bitwise AND of the 128-bit value in b and the bitwise NOT of the 128-bit value in a. https://msdn.microsoft.com/en-us/library/vstudio/1beaceh8(v=vs.100).aspx
inline __m128i _mm_andnot_si128 (__m128i a, __m128i b) 
{
	return (__m128i)vbicq_s32(b,a); // *NOTE* argument swap
}

// Computes the bitwise AND of the 128-bit value in a and the 128-bit value in b. https://msdn.microsoft.com/en-us/library/vstudio/6d1txsa8(v=vs.100).aspx
inline __m128i _mm_and_si128 (__m128i a, __m128i b)	
{
	return (__m128i)vandq_s32(a,b);
}

// Computes the bitwise AND of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/73ck1xc5(v=vs.100).aspx
inline __m128 _mm_and_ps(__m128 a , __m128 b ) 
{
	return (__m128)vandq_s32((__m128i)a,(__m128i)b);
}

// Computes the bitwise OR of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/7ctdsyy0(v=vs.100).aspx
inline __m128 _mm_or_ps(__m128 a , __m128 b ) 
{
	return (__m128)vorrq_s32((__m128i)a,(__m128i)b);
}

//Computes the bitwise OR of the 128-bit value in a and the 128-bit value in b. https://msdn.microsoft.com/en-us/library/vstudio/ew8ty0db(v=vs.100).aspx
inline __m128i _mm_or_si128 (__m128i a, __m128i b) 
{
	return (__m128i)vorrq_s32(a,b);
}

// NEON does not provide this method
// NOTE: TODO temporarily using C++ implementation
 // Creates a 4-bit mask from the most significant bits of the four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/vstudio/4490ys29(v=vs.100).aspx
inline int _mm_movemask_ps( __m128 a )
{
	int ret = 0;

	const uint32_t *ip = (const uint32_t *)&a;
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
	return ret;
}

// NEON does not support a general purpose permute intrinsic
// This could be emulated in normal C++ code, which would be slow.
// Not sure the proper fix for this.
// TODO: Emulated in C++ temporarily
// Selects four specific single-precision, floating-point values from a and b, based on the mask i. https://msdn.microsoft.com/en-us/library/vstudio/5f0858x0(v=vs.100).aspx
inline __m128 _mm_shuffle_ps(__m128 a , __m128 b , int i ) 
{
	__m128 ret;

	ret[0] = a[i&0x3];
	ret[1] = a[(i>>2)&0x3];

	ret[2] = b[(i>>4)&0x03];
	ret[3] = b[(i>>6)&0x03];

	return ret;
}

#define _mm_slli_epi32(a,b) (__m128i)vshlq_n_s32(a,b)

// TODO : May need a more effient implementation than the one I found online
// NEON does not provide a version of this function, here is an article about some ways to repro the results.
// http://stackoverflow.com/questions/11870910/sse-mm-movemask-epi8-equivalent-method-for-arm-neon
// Creates a 16-bit mask from the most significant bits of the 16 signed or unsigned 8-bit integers in a and zero extends the upper bits. https://msdn.microsoft.com/en-us/library/vstudio/s090c8fk(v=vs.100).aspx
inline int _mm_movemask_epi8 (__m128i _a)
{
	uint8x16_t input = (uint8x16_t)_a;
	const int8_t __attribute__ ((aligned (16))) xr[8] = {-7,-6,-5,-4,-3,-2,-1,0};
	uint8x8_t mask_and = vdup_n_u8(0x80);
	int8x8_t mask_shift = vld1_s8(xr);

	uint8x8_t lo = vget_low_u8(input);
	uint8x8_t hi = vget_high_u8(input);

	lo = vand_u8(lo, mask_and);
	lo = vshl_u8(lo, mask_shift);

	hi = vand_u8(hi, mask_and);
	hi = vshl_u8(hi, mask_shift);

	lo = vpadd_u8(lo,lo);
	lo = vpadd_u8(lo,lo);
	lo = vpadd_u8(lo,lo);

	hi = vpadd_u8(hi,hi);
	hi = vpadd_u8(hi,hi);
	hi = vpadd_u8(hi,hi);

	return ((hi[0] << 8) | (lo[0] & 0xFF));
}


// ******************************************
// Math operations
// ******************************************

 // Subtracts the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/1zad2k61(v=vs.100).aspx
inline __m128 _mm_sub_ps(__m128 a , __m128 b )
{
	return vsubq_f32(a,b);
}

// Subtracts the 4 signed or unsigned 32-bit integers of b from the 4 signed or unsigned 32-bit integers of a. https://msdn.microsoft.com/en-us/library/vstudio/fhh866h0(v=vs.100).aspx
inline __m128i _mm_sub_epi32 (__m128i a, __m128i b)
{
	return vsubq_s32(a,b);
}

// Adds the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/c9848chc(v=vs.100).aspx
inline __m128 _mm_add_ps(__m128 a , __m128 b ) 
{
	return vaddq_f32(a,b);
}

 // Adds the 4 signed or unsigned 32-bit integers in a to the 4 signed or unsigned 32-bit integers in b. https://msdn.microsoft.com/en-us/library/vstudio/09xs4fkk(v=vs.100).aspx
inline __m128i _mm_add_epi32 (__m128i a, __m128i b)
{
	return vaddq_s32(a,b);
}

// Multiplies the 8 signed or unsigned 16-bit integers from a by the 8 signed or unsigned 16-bit integers from b. https://msdn.microsoft.com/en-us/library/vstudio/9ks1472s(v=vs.100).aspx
inline __m128i _mm_mullo_epi16 (__m128i a, __m128i b)
{
	return (__m128i)vmulq_s16((int16x8_t)a,(int16x8_t)b);
}

// Multiplies the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/22kbk6t9(v=vs.100).aspx
inline __m128 _mm_mul_ps(__m128 a , __m128 b )
{
	return vmulq_f32(a,b);
}

// TODO: While this NEON function does compute the reciprocol it has far less accuracy than what is generated by SSE.  Therefore, due to low precision, it could produce unexpected results comparted to the SSE version.
// Computes the approximations of reciprocals of the four single-precision, floating-point values of a. https://msdn.microsoft.com/en-us/library/vstudio/796k1tty(v=vs.100).aspx
inline __m128 _mm_rcp_ps(__m128 a )
{
	return vrecpeq_f32(a);
}

// Computes the maximums of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/ff5d607a(v=vs.100).aspx
inline __m128 _mm_max_ps(__m128 a , __m128 b ) 
{
	return vmaxq_f32(a,b);
}

// Computes the minima of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/wh13kadz(v=vs.100).aspx
inline __m128 _mm_min_ps(__m128 a , __m128 b ) 
{
	return vminq_f32(a,b);
}

// Computes the pairwise minima of the 8 signed 16-bit integers from a and the 8 signed 16-bit integers from b. https://msdn.microsoft.com/en-us/library/vstudio/6te997ew(v=vs.100).aspx
inline __m128i _mm_min_epi16 (__m128i a, __m128i b) 
{
	return (__m128i)vminq_s16((int16x8_t)a,(int16x8_t)b);
}

// TODO : Needs a NEON implementation
// Discussion about how there is no exact match for this instruction
// http://stackoverflow.com/questions/11292884/neon-equivalent-to-sse-intrinsics
 // Multiplies the 8 signed 16-bit integers from a by the 8 signed 16-bit integers from b. https://msdn.microsoft.com/en-us/library/vstudio/59hddw1d(v=vs.100).aspx
inline __m128i _mm_mulhi_epi16 (__m128i a, __m128i b)
{
	__m128i ret;
	int16_t *d = (int16_t *)&ret;
	const int16_t *_a = (const int16_t *)&a;
	const int16_t *_b = (const int16_t *)&b;
	for (uint32_t i=0; i<8; i++)
	{
		int32_t m = (int32_t)_a[i]*(int32_t)_b[i];
		d[i] = (int16_t)(m>>16);
	}
	return ret;
}

// ******************************************
// Compare operations
// ******************************************

// Compares for less than https://msdn.microsoft.com/en-us/library/vstudio/f330yhc8(v=vs.100).aspx
inline __m128 _mm_cmplt_ps(__m128 a, __m128 b ) 
{
	return (__m128)vcltq_f32(a,b);
}

// Compares for greater than. https://msdn.microsoft.com/en-us/library/vstudio/11dy102s(v=vs.100).aspx
inline __m128 _mm_cmpgt_ps(__m128 a, __m128 b ) 
{
	return (__m128)vcgtq_f32(a,b);
}

// Compares for greater than or equal. https://msdn.microsoft.com/en-us/library/vstudio/fs813y2t(v=vs.100).aspx
inline __m128 _mm_cmpge_ps(__m128 a, __m128 b ) 
{
	return (__m128)vcgeq_f32(a,b);
}

// Compares for less than or equal. https://msdn.microsoft.com/en-us/library/vstudio/1s75w83z(v=vs.100).aspx
inline __m128 _mm_cmple_ps(__m128 a , __m128 b ) 
{
	return (__m128)vcleq_f32(a,b);
}

// Compares for equality. https://msdn.microsoft.com/en-us/library/vstudio/36aectz5(v=vs.100).aspx
inline __m128 _mm_cmpeq_ps(__m128 a , __m128 b ) 
{
	return (__m128)vceqq_f32(a,b);
}

// Compares the 4 signed 32-bit integers in a and the 4 signed 32-bit integers in b for less than. https://msdn.microsoft.com/en-us/library/vstudio/4ak0bf5d(v=vs.100).aspx
inline __m128i _mm_cmplt_epi32 (__m128i a, __m128i b) 
{
	return (__m128i)vcltq_s32(a,b);
}

// Compares the 4 signed 32-bit integers in a and the 4 signed 32-bit integers in b for greater than. https://msdn.microsoft.com/en-us/library/vstudio/1s9f2z0y(v=vs.100).aspx
inline __m128i _mm_cmpgt_epi32 (__m128i a, __m128i b) 
{
	return (__m128i)vcgtq_s32(a,b);
}

// ******************************************
// Conversions
// ******************************************

// Converts the four single-precision, floating-point values of a to signed 32-bit integer values using truncate. https://msdn.microsoft.com/en-us/library/vstudio/1h005y6x(v=vs.100).aspx
inline __m128i _mm_cvttps_epi32 (__m128 a) 
{
	return vcvtq_s32_f32(a);
}

// Converts the four signed 32-bit integer values of a to single-precision, floating-point values https://msdn.microsoft.com/en-us/library/vstudio/36bwxcx5(v=vs.100).aspx
inline __m128 _mm_cvtepi32_ps (__m128i a) 
{
	return vcvtq_f32_s32(a);
}

// TODO: Need NEON implementation!
// Converts the four single-precision, floating-point values of a to signed 32-bit integer values. https://msdn.microsoft.com/en-us/library/vstudio/xdc42k5e(v=vs.100).aspx
inline __m128i _mm_cvtps_epi32 (__m128 a) 
{
	__m128i ret;
	for (uint32_t i=0; i<4; i++)
	{
		ret[i] = (int32_t)(roundf(a[i]));
	}
	return ret;
}


#endif
