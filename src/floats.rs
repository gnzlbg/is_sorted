//! Algorithms for floats
#![allow(unused_attributes)]

/// 128-bit wide algorithm for slices of floating-point numbers
///
/// Note:
///
/// * `_mm_alignr_epi8` requires `SSSE3`
/// * `_mm_test_all_ones` requires `SSE4.1`
macro_rules! floats_128 {
    (
        $name:ident,
        $cpuid:tt,
        $id:ident,
        $nlanes:expr,
        $load:ident,
        $cmple:ident,
        $and:ident,
        $head:ident,
        $tail:ident
    ) => {
        #[inline]
        #[target_feature(enable = $cpuid)]
        pub unsafe fn $name(s: &[$id]) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            // The alignment requirements for 128-bit wide vectors is 16 bytes:
            const ALIGNMENT: usize = 16;
            let mut i = $head!(s, $id, ALIGNMENT);
            // ^^^^^^ i is the index of the first element aligned to an
            // ALIGNMENT boundary
            let n = s.len() as isize;
            let ap = |o| (s.as_ptr().offset(o)) as *const $id;

            // Unroll factor: #of 128-bit vectors processed per loop iteration
            const NVECS: isize = 4;
            // #lanes in each 128-bit vector
            const NLANES: isize = $nlanes;
            // Stride: number of elements processed in each loop iteration
            // unroll_factor * #lane per vector
            const STRIDE: isize = NLANES * NVECS;
            // Minimum number of elements required for explicit vectorization.
            // Since we need one extra vector to get the last element,
            // this is #lanes * (unroll + 1) == stride + #lanes
            const MIN_LEN: isize = NLANES * (NVECS + 1);
            // Width of the vector lanes in bytes
            const EWIDTH: i32 = 128 / 8 / NLANES as i32;
            if (n - i) >= MIN_LEN {
                let mut current = $load(ap(i + 0 * NLANES)); // [a0,..,a3]
                while i < n - STRIDE {
                    use mem::transmute;
                    let next0 = $load(ap(i + 1 * NLANES)); // [a4,..,a7]
                    let next1 = $load(ap(i + 2 * NLANES)); // [a8,..,a11]
                    let next2 = $load(ap(i + 3 * NLANES)); // [a12,..,a15]
                    let next3 = $load(ap(i + 4 * NLANES)); // [a16,..a19]

                    let compare0 = _mm_alignr_epi8(
                        transmute(next0),
                        transmute(current),
                        EWIDTH,
                    ); // [a1,..,a4]
                    let compare1 = _mm_alignr_epi8(
                        transmute(next1),
                        transmute(next0),
                        EWIDTH,
                    ); // [a5,..,a8]
                    let compare2 = _mm_alignr_epi8(
                        transmute(next2),
                        transmute(next1),
                        EWIDTH,
                    ); // [a9,..,a12]
                    let compare3 = _mm_alignr_epi8(
                        transmute(next3),
                        transmute(next2),
                        EWIDTH,
                    ); // [a13,..,a16]

                    // [a0 <= a1,..,a3 <= a4]
                    let mask0 = $cmple(current, transmute(compare0));
                    // [a4 <= a5,..,a7 <= a8]
                    let mask1 = $cmple(next0, transmute(compare1));
                    // [a8 <= a9,..,a11 <= a12]
                    let mask2 = $cmple(next1, transmute(compare2));
                    // [a12 <= a13,..,a15 <= a16]
                    let mask3 = $cmple(next2, transmute(compare3));

                    // mask = mask0 && mask1 && mask2 && mask3
                    let mask = $and($and(mask0, mask1), $and(mask2, mask3));

                    // if some le comparison was false, the mask will have some
                    // bits cleared and this will return 0:
                    if _mm_test_all_ones(transmute(mask)) == 0 {
                        return false;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            $tail!(s, n, i)
        }
    };
}

pub mod sse41 {
    // `_mm_load_ps` requires `SSE`
    // `_mm_cmple_ps` requires `SSE`
    // `_mm_and_ps` requires `SSE`
    floats_128!(
        is_sorted_lt_f32,
        "sse4.1",
        f32,
        4,
        _mm_load_ps,
        _mm_cmple_ps,
        _mm_and_ps,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );
    // `_mm_load_pd` requires `SSE2`
    // `_mm_cmple_pd` requires `SSE2`
    // `_mm_and_pd` requires `SSE2`
    floats_128!(
        is_sorted_lt_f64,
        "sse4.1",
        f64,
        2,
        _mm_load_pd,
        _mm_cmple_pd,
        _mm_and_pd,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );
    // `_mm_load_ps` requires `SSE`
    // `_mm_cmple_ps` requires `SSE`
    // `_mm_and_ps` requires `SSE`
    floats_128!(
        is_sorted_gt_f32,
        "sse4.1",
        f32,
        4,
        _mm_load_ps,
        _mm_cmpge_ps,
        _mm_and_ps,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );
    // `_mm_load_pd` requires `SSE2`
    // `_mm_cmple_pd` requires `SSE2`
    // `_mm_and_pd` requires `SSE2`
    floats_128!(
        is_sorted_gt_f64,
        "sse4.1",
        f64,
        2,
        _mm_load_pd,
        _mm_cmpge_pd,
        _mm_and_pd,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );
}

/// 256-bit wide algorithm for slices of floating-point numbers
macro_rules! floats_256 {
    ($name:ident, $cpuid:tt, $id:ident, $nlanes:expr, $load:ident, $loadu:ident,
     $cmp:ident, $cmp_t:ident, $and:ident, $testc:ident, $set1:ident, $ones:expr,
     $head:ident, $tail:ident) => {
        #[inline]
        #[target_feature(enable = $cpuid)]
        pub unsafe fn $name(s: &[$id]) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            // The alignment requirements for 256-bit wide vectors is 16 bytes:
            const ALIGNMENT: usize = 32;
            let mut i = $head!(s, $id, ALIGNMENT);
            // ^^^^^^ i is the index of the first element aligned to an ALIGNMENT boundary
            let n = s.len() as isize;
            let ap = |o| (s.as_ptr().offset(o)) as *const $id;

            // Unroll factor: #of 256-bit vectors processed per loop iteration
            const NVECS: isize = 4;
            // #lanes in each 256-bit vector
            const NLANES: isize = $nlanes;
            // Stride: number of elements processed in each loop iteration
            // unroll_factor * #lane per vector
            const STRIDE: isize = NLANES * NVECS;
            // Minimum number of elements required for explicit vectorization.
            // Since we need one extra vector to get the last element,
            // this is #lanes * (unroll + 1) == stride + #lanes
            const MIN_LEN: isize = NLANES * (NVECS + 1);
            if (n - i) >= MIN_LEN {
                while i < n - STRIDE {
                    use ::mem::transmute;
                    let current = $load(ap(i + 0 * NLANES)); // [a0,..,a3]
                    // == 16 | the last vector of current is the first of next
                    let next0 = $load(ap(i + 1 * NLANES)); // [a4,..,a7]
                    let next1 = $load(ap(i + 2 * NLANES)); // [a8,..,a11]
                    let next2 = $load(ap(i + 3 * NLANES)); // [a12,..,a15]

                    let compare0 = $loadu(ap(i + 0 * NLANES + 1)); // [a1,..,a8]
                    let compare1 = $loadu(ap(i + 1 * NLANES + 1)); // [a9,..,a16]
                    let compare2 = $loadu(ap(i + 2 * NLANES + 1)); // [a17,..,a23]
                    let compare3 = $loadu(ap(i + 3 * NLANES + 1)); // [a25,..,a32]

                    // [a0 <= a1,..,a3 <= a4]
                    let mask0 = $cmp(
                        current,
                        transmute(compare0),
                        $cmp_t,
                    );
                    // [a4 <= a5,..,a7 <= a8]
                    let mask1 = $cmp(
                        next0,
                        transmute(compare1),
                        $cmp_t,
                    );
                    // [a8 <= a9,..,a11 <= a12]
                    let mask2 = $cmp(
                        next1,
                        transmute(compare2),
                        $cmp_t,
                    );
                    // [a12 <= a13,..,a15 <= a16]
                    let mask3 = $cmp(
                        next2,
                        transmute(compare3),
                        $cmp_t,
                    );

                    // mask = mask0 | mask1 | mask2 | mask3
                    let mask = $and(
                        $and(mask0, mask1),
                        $and(mask2, mask3),
                    );

                    // if some le comparison was false the mask won't have all
                    // bits set and testc returns 0:
                    if $testc(
                        mask,
                        $set1(transmute($ones)),
                    ) == 0
                    {
                        return false;
                    }

                    i += STRIDE;
                }
            }

            $tail!(s, n, i)
        }
    }
}

pub mod avx {
    // `_mm256_load_ps` requires `AVX`
    // `_mm256_loadu_ps` requires `AVX`
    // `_mm256_cmp_ps` requires `AVX`
    // `_mm256_and_ps` requires `AVX`
    // `_mm256_testc_ps` requires `AVX`
    // `_mm256_set1_ps` requires `AVX`
    floats_256!(
        is_sorted_lt_f32,
        "avx",
        f32,
        8,
        _mm256_load_ps,
        _mm256_loadu_ps,
        _mm256_cmp_ps,
        _CMP_LE_OQ,
        _mm256_and_ps,
        _mm256_testc_ps,
        _mm256_set1_ps,
        -1_i32,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );
    // `_mm256_load_pd` requires `AVX`
    // `_mm256_loadu_pd` requires `AVX`
    // `_mm256_cmp_pd` requires `AVX`
    // `_mm256_and_pd` requires `AVX`
    // `_mm256_testc_pd` requires `AVX`
    // `_mm256_set1_pd` requires `AVX`
    floats_256!(
        is_sorted_lt_f64,
        "avx",
        f64,
        4,
        _mm256_load_pd,
        _mm256_loadu_pd,
        _mm256_cmp_pd,
        _CMP_LE_OQ,
        _mm256_and_pd,
        _mm256_testc_pd,
        _mm256_set1_pd,
        -1_i64,
        is_sorted_lt_until_alignment_boundary,
        is_sorted_lt_tail
    );
    // `_mm256_load_ps` requires `AVX`
    // `_mm256_loadu_ps` requires `AVX`
    // `_mm256_cmp_ps` requires `AVX`
    // `_mm256_and_ps` requires `AVX`
    // `_mm256_testc_ps` requires `AVX`
    // `_mm256_set1_ps` requires `AVX`
    floats_256!(
        is_sorted_gt_f32,
        "avx",
        f32,
        8,
        _mm256_load_ps,
        _mm256_loadu_ps,
        _mm256_cmp_ps,
        _CMP_GE_OQ,
        _mm256_and_ps,
        _mm256_testc_ps,
        _mm256_set1_ps,
        -1_i32,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );
    // `_mm256_load_pd` requires `AVX`
    // `_mm256_loadu_pd` requires `AVX`
    // `_mm256_cmp_pd` requires `AVX`
    // `_mm256_and_pd` requires `AVX`
    // `_mm256_testc_pd` requires `AVX`
    // `_mm256_set1_pd` requires `AVX`
    floats_256!(
        is_sorted_gt_f64,
        "avx",
        f64,
        4,
        _mm256_load_pd,
        _mm256_loadu_pd,
        _mm256_cmp_pd,
        _CMP_GE_OQ,
        _mm256_and_pd,
        _mm256_testc_pd,
        _mm256_set1_pd,
        -1_i64,
        is_sorted_gt_until_alignment_boundary,
        is_sorted_gt_tail
    );

}
