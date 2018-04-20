//! Algorithms for signed integers
#![allow(unused_attributes)]

/// 128-bit-wide algorithm for slices of signed integers
///
/// This algorithm has been adapted from HeroicKatora's reddit post:
/// https://www.reddit.com/r/cpp/comments/8bkaj3/is_sorted_using_simd_instructions/dx7jj8u/
///
/// Note:
///
/// * `_mm_load_si128` requires `SSE2`
/// * `_mm_alignr_epi8` requires `SSSE3`
/// * `_mm_or_si128` requires `SSE2`
/// * `_mm_test_all_zeros` requires `SSE4.1`
macro_rules! signed_128 {
    ($name:ident, $cpuid:tt, $id:ident, $nlanes:expr, $cmpgt:ident,
     $head:ident, $tail:ident) => {
        #[inline]
        #[target_feature(enable = $cpuid)]
        pub unsafe fn $name(s: &[$id]) -> usize {
            #[cfg(target_arch = "x86")]
            use ::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use ::arch::x86_64::*;

            // The alignment requirements for 128-bit wide vectors is 16 bytes
            const ALIGNMENT: usize = 16;
            let mut i: usize = $head!(s, $id, ALIGNMENT);
            // ^^^^^^ i is the index of the first element aligned to an ALIGNMENT boundary
            let n = s.len();
            let ap = |o| (s.as_ptr().offset(o as isize)) as *const __m128i;

            // Unroll factor: #of 128-bit vectors processed per loop iteration
            const NVECS: usize = 4;
            // #lanes in each 128-bit vector
            const NLANES: usize = $nlanes;
            // Stride: number of elements processed in each loop iteration
            // unroll_factor * #lane per vector
            const STRIDE: usize = NLANES * NVECS;
            // Minimum number of elements required for explicit vectorization.
            // Since we need one extra vector to get the last element,
            // this is #lanes * (unroll + 1) == stride + #lanes
            const MIN_LEN: usize = NLANES * (NVECS + 1);
            // Width of the vector lanes in bytes
            const EWIDTH: i32 = 128 / 8 / NLANES as i32;
            if (n - i) >= MIN_LEN {
                let mut current = _mm_load_si128(ap(i + 0 * NLANES)); // [a0,..,a3]
                while i < n - STRIDE {
                    let next0 = _mm_load_si128(ap(i + 1 * NLANES)); // [a4,..,a7]
                    let next1 = _mm_load_si128(ap(i + 2 * NLANES)); // [a8,..,a11]
                    let next2 = _mm_load_si128(ap(i + 3 * NLANES)); // [a12,..,a15]
                    let next3 = _mm_load_si128(ap(i + 4 * NLANES)); // [a16,..a19]

                    let compare0 = _mm_alignr_epi8(next0, current, EWIDTH); // [a1,..,a4]
                    let compare1 = _mm_alignr_epi8(next1, next0, EWIDTH); // [a5,..,a8]
                    let compare2 = _mm_alignr_epi8(next2, next1, EWIDTH); // [a9,..,a12]
                    let compare3 = _mm_alignr_epi8(next3, next2, EWIDTH); // [a13,..,a16]

                    // [a0 > a1,..,a3 > a4]
                    let mask0 = $cmpgt(current, compare0);
                    // [a4 > a5,..,a7 > a8]
                    let mask1 = $cmpgt(next0, compare1);
                    // [a8 > a9,..,a11 > a12]
                    let mask2 = $cmpgt(next1, compare2);
                    // [a12 > a13,..,a15 > a16]
                    let mask3 = $cmpgt(next2, compare3);

                    // mask = mask0 | mask1 | mask2 | mask3
                    let mask = _mm_or_si128(
                        _mm_or_si128(mask0, mask1),
                        _mm_or_si128(mask2, mask3),
                    );

                    // mask & mask == 0: if some gt comparison was true, the
                    // mask will have some bits set. The result of bitwise & of
                    // the mask with itself is only zero if all of the bits of
                    // the mask are zero. Therefore, if some comparison
                    // succeeded, there will be some non-zero bit, and all
                    // zeros would return false (aka 0).
                    if _mm_test_all_zeros(mask, mask) == 0 {
                        return i;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            $tail!(s, n, i)
        }
    }
}

pub mod sse42 {
    // `_mm_cmpgt_epi64` requires `SSE4.2`
    signed_128!(
        is_sorted_lt_i64,
        "sse4.2",
        i64,
        2,
        _mm_cmpgt_epi64,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );

    #[cfg(target_arch = "x86")]
    use arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use arch::x86_64::*;

    #[target_feature(enable = "sse4.2")]
    unsafe fn _mm_cmplt_epi64(x: __m128i, y: __m128i) -> __m128i {
        let a = _mm_cmpgt_epi64(x, y); // x < y SSE4.2
        let b = _mm_cmpeq_epi64(x, y); // x == y SSE4.1
        let c = _mm_or_si128(a, b); // x <= y <=> x < y || == y  -- SSE2
        let ones = _mm_set1_epi64x(-1_i64);
        _mm_andnot_si128(c, ones) // !(c) & 1  -- SSE2
    }
    // `_mm_cmplt_epi64` requires `SSE4.2`
    signed_128!(
        is_sorted_gt_i64,
        "sse4.2",
        i64,
        2,
        _mm_cmplt_epi64,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );
}

pub mod sse41 {
    // `_mm_cmpgt_epi32` requires `SSE2`
    signed_128!(
        is_sorted_lt_i32,
        "sse4.1",
        i32,
        4,
        _mm_cmpgt_epi32,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );
    // `_mm_cmpgt_epi16` requires `SSE2`
    signed_128!(
        is_sorted_lt_i16,
        "sse4.1",
        i16,
        8,
        _mm_cmpgt_epi16,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );
    // `_mm_cmpgt_epi8` requires `SSE2`
    signed_128!(
        is_sorted_lt_i8,
        "sse4.1",
        i8,
        16,
        _mm_cmpgt_epi8,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );

    // `_mm_cmplt_epi32` requires `SSE2`
    signed_128!(
        is_sorted_gt_i32,
        "sse4.1",
        i32,
        4,
        _mm_cmplt_epi32,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );
    // `_mm_cmplt_epi16` requires `SSE2`
    signed_128!(
        is_sorted_gt_i16,
        "sse4.1",
        i16,
        8,
        _mm_cmplt_epi16,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );
    // `_mm_cmplt_epi8` requires `SSE2`
    signed_128!(
        is_sorted_gt_i8,
        "sse4.1",
        i8,
        16,
        _mm_cmplt_epi8,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );
}

/// 256-bit wide algorithm for slices of signed integers
///
/// Note:
///
/// * `_mm256_load_si256` requires `AVX`
/// * `_mm256_loadu_si256` requires `AVX`
/// * `_mm256_and_si256` requires `AVX2`
/// * `_mm256_testc_si256` requires `AVX`
/// * `_mm256_set1_epi64x` requires `AVX`
macro_rules! signed_256 {
    ($name:ident, $cpuid:tt, $id:ident, $nlanes:expr, $cmpgt:ident, $head:ident, $tail:ident) => {
        #[inline]
        #[target_feature(enable = $cpuid)]
        pub unsafe fn $name(s: &[$id]) -> usize {
            #[cfg(target_arch = "x86")]
            use ::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use ::arch::x86_64::*;

            // The alignment requirements for 256-bit wide vectors is 32 bytes
            const ALIGNMENT: usize = 32;
            let mut i: usize = $head!(s, $id, ALIGNMENT);
            // ^^^^^^ i is the index of the first element aligned to an ALIGNMENT boundary
            let n = s.len();
            let ap = |o| (s.as_ptr().offset(o as isize)) as *const __m256i;

            // Unroll factor: #of 256-bit vectors processed per loop iteration
            const NVECS: usize = 4;
            // #lanes in each 256-bit vector
            const NLANES: usize = $nlanes;
            // Stride: number of elements processed in each loop iteration
            // unroll_factor * #lane per vector
            const STRIDE: usize = NLANES * NVECS;
            // Minimum number of elements required for explicit vectorization.
            // Since we need one extra vector to get the last element,
            // this is #lanes * (unroll + 1) == stride + #lanes
            const MIN_LEN: usize = NLANES * (NVECS + 1);
            if (n - i) >= MIN_LEN {
                while i < n - STRIDE {
                    let current = _mm256_load_si256(ap(i + 0 * NLANES)); // [a0,..,a7]
                    let next0 = _mm256_load_si256(ap(i + 1 * NLANES)); // [a8,..,a16]
                    let next1 = _mm256_load_si256(ap(i + 2 * NLANES)); // [a16,..,a23]
                    let next2 = _mm256_load_si256(ap(i + 3 * NLANES)); // [a24,..,a31]

                    let compare0 = _mm256_loadu_si256(ap(i + 0 * NLANES + 1)); // [a1,..,a8]
                    let compare1 = _mm256_loadu_si256(ap(i + 1 * NLANES + 1)); // [a9,..,a16]
                    let compare2 = _mm256_loadu_si256(ap(i + 2 * NLANES + 1)); // [a17,..,a23]
                    let compare3 = _mm256_loadu_si256(ap(i + 3 * NLANES + 1)); // [a25,..,a32]

                    // [a0 > a1,..,a7 > a8]
                    let mask0 = $cmpgt(current, compare0);
                    // [a8 > a9,..,a15 > a16]
                    let mask1 = $cmpgt(next0, compare1);
                    // [a16 > a17,..,a23 > a24]
                    let mask2 = $cmpgt(next1, compare2);
                    // [a24 > a25,..,a31 > a32]
                    let mask3 = $cmpgt(next2, compare3);

                    // mask = mask0 | mask1 | mask2 | mask3
                    let mask = _mm256_or_si256(
                        _mm256_or_si256(mask0, mask1),
                        _mm256_or_si256(mask2, mask3),
                    );

                    // mask & mask == 0: if some gt comparison was true, the
                    // mask will have some bits set. The result of bitwise & of
                    // the mask with itself is only zero if all of the bits of
                    // the mask are zero. Therefore, if some comparison
                    // succeeded, there will be some non-zero bit, and all
                    // zeros would return false (aka 0).
                    if _mm256_testz_si256(mask, mask) == 0 {
                        return i;
                    }

                    i += STRIDE;
                }
            }

            $tail!(s, n, i)
        }
    }
}

pub mod avx2 {
    // `_mm256_cmpgt_epi64` requires `AVX2`
    signed_256!(
        is_sorted_lt_i64,
        "avx2",
        i64,
        4,
        _mm256_cmpgt_epi64,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );
    // `_mm256_cmpgt_epi32` requires `AVX2`
    signed_256!(
        is_sorted_lt_i32,
        "avx2",
        i32,
        8,
        _mm256_cmpgt_epi32,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );
    // `_mm256_cmpgt_epi16` requires `AVX2`
    signed_256!(
        is_sorted_lt_i16,
        "avx2",
        i16,
        16,
        _mm256_cmpgt_epi16,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );
    // `_mm256_cmpgt_epi8` requires `AVX2`
    signed_256!(
        is_sorted_lt_i8,
        "avx2",
        i8,
        32,
        _mm256_cmpgt_epi8,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );

    #[cfg(target_arch = "x86")]
    use arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    unsafe fn _mm256_cmplt_epi64(x: __m256i, y: __m256i) -> __m256i {
        let a = _mm256_cmpgt_epi64(x, y);
        let b = _mm256_cmpeq_epi64(x, y);
        let x = _mm256_or_si256(a, b); // ge
                                       // !(x) & 1:
        _mm256_andnot_si256(x, _mm256_set1_epi64x(-1_i64))
    }

    // `_mm256_cmplt_epi64` requires `AVX2`
    signed_256!(
        is_sorted_gt_i64,
        "avx2",
        i64,
        4,
        _mm256_cmplt_epi64,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );

    #[target_feature(enable = "avx2")]
    unsafe fn _mm256_cmplt_epi32(x: __m256i, y: __m256i) -> __m256i {
        let a = _mm256_cmpgt_epi32(x, y);
        let b = _mm256_cmpeq_epi32(x, y);
        let x = _mm256_or_si256(a, b); // ge
                                       // !(x) & 1:
        _mm256_andnot_si256(x, _mm256_set1_epi64x(-1_i64))
    }

    // `_mm256_cmplt_epi32` requires `AVX2`
    signed_256!(
        is_sorted_gt_i32,
        "avx2",
        i32,
        8,
        _mm256_cmplt_epi32,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );

    #[target_feature(enable = "avx2")]
    unsafe fn _mm256_cmplt_epi16(x: __m256i, y: __m256i) -> __m256i {
        let a = _mm256_cmpgt_epi16(x, y);
        let b = _mm256_cmpeq_epi16(x, y);
        let x = _mm256_or_si256(a, b); // ge
                                       // !(x) & 1:
        _mm256_andnot_si256(x, _mm256_set1_epi64x(-1_i64))
    }

    // `_mm256_cmplt_epi16` requires `AVX2`
    signed_256!(
        is_sorted_gt_i16,
        "avx2",
        i16,
        16,
        _mm256_cmplt_epi16,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );

    #[target_feature(enable = "avx2")]
    unsafe fn _mm256_cmplt_epi8(x: __m256i, y: __m256i) -> __m256i {
        let a = _mm256_cmpgt_epi8(x, y);
        let b = _mm256_cmpeq_epi8(x, y);
        let x = _mm256_or_si256(a, b); // ge
                                       // !(x) & 1:
        _mm256_andnot_si256(x, _mm256_set1_epi64x(-1_i64))
    }

    // `_mm256_cmplt_epi8` requires `AVX2`
    signed_256!(
        is_sorted_gt_i8,
        "avx2",
        i8,
        32,
        _mm256_cmplt_epi8,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );

}
