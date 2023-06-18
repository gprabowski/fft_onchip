#pragma once 

#define gemm8x8x4(a, b, c1, c2)                                                \
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "              \
               "{%0, %1},"                                                     \
               "{%2},"                                                         \
               "{%3},"                                                         \
               "{%4, %5};\n"                                                   \
               : "=d"(c1), "=d"(c2)                                            \
               : "d"(a), "d"(b), "d"(c1), "d"(c2));


#define gemm8x8x8(a1, a2, b1, b2, c1, c2)                                      \
  gemm8x8x4(a1, b1, c1, c2);                                                   \
  gemm8x8x4(a2, b2, c1, c2);


#define complex_gemm8x8x8(a1, a2, b1, b2, c1r, c1i, c2r, c2i) \
    gemm8x8x8(a1.real(), a2.real(), b1.real(), b2.real(), c1r, c2r); \
    gemm8x8x8(a1.imag(), a2.imag(), b1.real(), b2.real(), c1i, c2i); \
    gemm8x8x8(a1.real(), a2.real(), b1.imag(), b2.imag(), c1i, c2i); \
    gemm8x8x8(-a1.imag(), -a2.imag(), b1.imag(), b2.imag(), c1r, c2r);


// More register intesive, but only 3 dmma version of the above GEMM
// to get the result from karatsuba, perform:
// c1r = s21 - s11;
// c1i = s31 - s21 - s11;
// c2r = s22 - s21;
// c2i = s32 - s22 - s21;
/////////////////////////////////////////////////////////////////////////
#define karatsuba_gemm8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32)           \
  gemm8x8x8(a1.real(), a2.real(), b1.real(), b2.real(), s11, s12);             \
  gemm8x8x8(a1.imag(), a2.imag(), b1.imag(), b2.imag(), s21, s22);             \
  gemm8x8x8(a1.real() + a1.imag(), a2.real() + a2.imag(),                      \
            b1.real() + b1.imag(), b2.real() + b2.imag(), s31, s32);

