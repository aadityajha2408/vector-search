#pragma once
#include <cstddef>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace vs {

// Inner product. Vectors are L2-normalised, so dot == cosine similarity.
//
// With AVX2 (gcc/clang -mavx2, MSVC /arch:AVX2) we use explicit 256-bit intrinsics: 8 floats per
// instruction, two independent accumulators to hide FMA latency. MSVC does not reliably
// auto-vectorise float reductions, so relying on the compiler left this kernel scalar.
// Without AVX2 we fall back to 8 independent scalar accumulators.
#if defined(__AVX2__)

#if defined(__FMA__) || defined(_MSC_VER)
#define VS_FMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define VS_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif

inline float dot(const float* a, const float* b, size_t d) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        acc0 = VS_FMA(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = VS_FMA(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= d; i += 8) acc0 = VS_FMA(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc0 = _mm256_add_ps(acc0, acc1);

    __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float s = _mm_cvtss_f32(sum4);
    for (; i < d; ++i) s += a[i] * b[i];  // tail (d not a multiple of 8)
    return s;
}

#else

inline float dot(const float* a, const float* b, size_t d) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        s0 += a[i] * b[i];         s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4]; s5 += a[i + 5] * b[i + 5];
        s6 += a[i + 6] * b[i + 6]; s7 += a[i + 7] * b[i + 7];
    }
    float s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
}

#endif

}  // namespace vs