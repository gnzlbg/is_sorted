//! Extends `Iterator` with three algorithms, `is_sorted`, `is_sorted_by`, and
//! `is_sorted_by_key` that check whether the elements of an `Iterator` are
//! sorted in `O(N)` time and `O(1)` space.

// If the `use_std` feature is not enable, compile for `no_std`:
#![cfg_attr(not(feature = "use_std"), no_std)]
// If the `unstable` feature is enabled, enable nightly-only features:
#![cfg_attr(
    feature = "unstable",
    feature(
        specialization, fn_traits, unboxed_closures, stdsimd, align_offset,
        target_feature, cfg_target_feature
    )
)]

#[cfg(not(feature = "use_std"))]
use core as std;

use std::cmp;

#[cfg(feature = "unstable")]
use std::{arch, mem, slice};

use cmp::Ordering;

#[cfg(feature = "unstable")]
mod ord {
    use cmp::Ordering;

    /// Equivalent to `Ord::cmp(a, b)`
    pub struct Less();

    impl<'a, 'b, T: Ord> FnOnce<(&'a T, &'b T)> for Less {
        type Output = Ordering;
        extern "rust-call" fn call_once(
            self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.cmp(arg.1)
        }
    }

    impl<'a, 'b, T: Ord> FnMut<(&'a T, &'b T)> for Less {
        extern "rust-call" fn call_mut(
            &mut self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.cmp(arg.1)
        }
    }

    /// Equivalent to `Ord::cmp(a, b).reverse()`
    pub struct Greater();

    impl<'a, 'b, T: Ord> FnOnce<(&'a T, &'b T)> for Greater {
        type Output = Ordering;
        extern "rust-call" fn call_once(
            self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.cmp(arg.1).reverse()
        }
    }

    impl<'a, 'b, T: Ord> FnMut<(&'a T, &'b T)> for Greater {
        extern "rust-call" fn call_mut(
            &mut self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.cmp(arg.1).reverse()
        }
    }

    /// Equivalent to `PartialOrd::partial_cmp(a, b).unwrap()`
    pub struct PartialLessUnwrapped();

    impl<'a, 'b, T: PartialOrd> FnOnce<(&'a T, &'b T)> for PartialLessUnwrapped {
        type Output = Ordering;
        extern "rust-call" fn call_once(
            self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.partial_cmp(arg.1).unwrap()
        }
    }

    impl<'a, 'b, T: PartialOrd> FnMut<(&'a T, &'b T)> for PartialLessUnwrapped {
        extern "rust-call" fn call_mut(
            &mut self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.partial_cmp(arg.1).unwrap()
        }
    }

    /// Equivalent to `PartialOrd::partial_cmp(a, b).unwrap().reverse()`
    pub struct PartialGreaterUnwrapped();

    impl<'a, 'b, T: PartialOrd> FnOnce<(&'a T, &'b T)>
        for PartialGreaterUnwrapped
    {
        type Output = Ordering;
        extern "rust-call" fn call_once(
            self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.partial_cmp(arg.1).unwrap().reverse()
        }
    }

    impl<'a, 'b, T: PartialOrd> FnMut<(&'a T, &'b T)> for PartialGreaterUnwrapped {
        extern "rust-call" fn call_mut(
            &mut self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.partial_cmp(arg.1).unwrap().reverse()
        }
    }

}

/// Callable equivalent to `Ord::cmp(a, b)`.
#[cfg(feature = "unstable")]
#[allow(non_upper_case_globals)]
pub const Less: ord::Less = ord::Less();

/// Callable equivalent to `Ord::cmp(a, b).reverse()`.
#[cfg(feature = "unstable")]
#[allow(non_upper_case_globals)]
pub const Greater: ord::Greater = ord::Greater();

/// Callable equivalent to `PartialOrd::partial_cmp(a, b).unwrap()`.
#[cfg(feature = "unstable")]
#[allow(non_upper_case_globals)]
pub const PartialLessUnwrapped: ord::PartialLessUnwrapped =
    ord::PartialLessUnwrapped();

/// Callable equivalent to `PartialOrd::partial_cmp(a, b).unwrap().reverse()`.
#[cfg(feature = "unstable")]
#[allow(non_upper_case_globals)]
pub const PartialGreaterUnwrapped: ord::PartialGreaterUnwrapped =
    ord::PartialGreaterUnwrapped();

/// Extends `Iterator` with `is_sorted`, `is_sorted_by`, and
/// `is_sorted_by_key`.
pub trait IsSorted: Iterator {
    /// Returns `true` if the elements of the iterator are sorted in increasing
    /// order according to `<Self::Item as Ord>::cmp`.
    ///
    /// ```
    /// # use is_sorted::IsSorted;
    /// let v = vec![0, 1, 2 , 3];
    /// assert!(v.iter().is_sorted());
    ///
    /// let v = vec![0, 1, 2 , -1];
    /// assert!(!v.iter().is_sorted());
    /// ```
    #[inline]
    fn is_sorted(&mut self) -> bool
    where
        Self: Sized,
        Self::Item: Ord,
    {
        #[cfg(feature = "unstable")]
        {
            self.is_sorted_by(Less)
        }
        #[cfg(not(feature = "unstable"))]
        {
            self.is_sorted_by(<Self::Item as Ord>::cmp)
        }
    }

    /// Returns `true` if the elements of the iterator
    /// are sorted according to the `compare` function.
    ///
    /// ```
    /// # use is_sorted::IsSorted;
    /// # use std::cmp::Ordering;
    /// // Is an iterator sorted in decreasing order?
    /// fn decr<T: Ord>(a: &T, b: &T) -> Ordering {
    ///     a.cmp(b).reverse()
    /// }
    ///
    /// let v = vec![3, 2, 1 , 0];
    /// assert!(v.iter().is_sorted_by(decr));
    ///
    /// let v = vec![3, 2, 1 , 4];
    /// assert!(!v.iter().is_sorted_by(decr));
    /// ```
    #[inline]
    fn is_sorted_by<F>(&mut self, compare: F) -> bool
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Ordering,
    {
        is_sorted_by_impl(self, compare)
    }

    /// Returns `true` if the elements of the iterator
    /// are sorted according to the `key` extraction function.
    ///
    /// ```
    /// # use is_sorted::IsSorted;
    /// let v = vec![0_i32, -1, 2, -3];
    /// assert!(v.iter().is_sorted_by_key(|v| v.abs()));
    ///
    /// let v = vec![0_i32, -1, 2, 0];
    /// assert!(!v.iter().is_sorted_by_key(|v| v.abs()));
    /// ```
    #[inline]
    fn is_sorted_by_key<F, B>(&mut self, mut key: F) -> bool
    where
        Self: Sized,
        B: Ord,
        F: FnMut(&Self::Item) -> B,
    {
        self.map(|v| key(&v)).is_sorted()
    }
}

// Blanket implementation for all types that implement `Iterator`:
impl<I: Iterator> IsSorted for I {}

// This function dispatch to the appropriate `is_sorted_by` implementation.
#[inline]
fn is_sorted_by_impl<I, F>(iter: &mut I, compare: F) -> bool
where
    I: Iterator,
    F: FnMut(&I::Item, &I::Item) -> Ordering,
{
    <I as IsSortedBy<F>>::is_sorted_by(iter, compare)
}

// This trait is used to provide specialized implementations of `is_sorted_by`
// for different (Iterator,Cmp) pairs:
trait IsSortedBy<F>: Iterator
where
    F: FnMut(&Self::Item, &Self::Item) -> Ordering,
{
    fn is_sorted_by(&mut self, compare: F) -> bool;
}

// This blanket implementation acts as the fall-back, and just forwards to the
// scalar implementation of the algorithm.
impl<I, F> IsSortedBy<F> for I
where
    I: Iterator,
    F: FnMut(&I::Item, &I::Item) -> Ordering,
{
    #[inline]
    #[cfg(feature = "unstable")]
    default fn is_sorted_by(&mut self, compare: F) -> bool {
        is_sorted_by_scalar_impl(self, compare)
    }

    #[inline]
    #[cfg(not(feature = "unstable"))]
    fn is_sorted_by(&mut self, compare: F) -> bool {
        is_sorted_by_scalar_impl(self, compare)
    }
}

/// Scalar `is_sorted_by` implementation.
#[inline]
fn is_sorted_by_scalar_impl<I, F>(iter: &mut I, mut compare: F) -> bool
where
    I: Iterator,
    F: FnMut(&I::Item, &I::Item) -> Ordering,
{
    let first = iter.next();
    if let Some(mut first) = first {
        return iter.all(|second| {
            if compare(&first, &second) == Ordering::Greater {
                return false;
            }
            first = second;
            true
        });
    }
    true
}


/// Checks whether a slice is sorted until the first element aligned with a
/// $boundary (16 for 16-byte boundary). Returns a (i,n,s) tuple where `i` is
/// the index of the next element in the slice aligned to a 16-byte boundary,
/// `n` the slice length, and `s` the slice.
macro_rules! is_sorted_handle_unaligned_head_int {
    ($x:ident, $ty:ident, $boundary:expr) => {{
        let s = $x.as_slice();
        let n = s.len() as isize;
        // If the slice has zero or one elements, it is sorted:
        if n < 2 {
            return true;
        }

        let mut i: isize = 0;

        // The first element of the slice might not be aligned to a
        // 16-byte boundary. Handle the elements until the
        // first 16-byte boundary using the scalar algorithm
        {
            let mut a = s.as_ptr().align_offset($boundary) / mem::size_of::<$ty>();
            while a > 0 && i < n - 1 {
                if s.get_unchecked(i as usize)
                    > s.get_unchecked(i as usize + 1)
                {
                    return false;
                }
                i += 1;
                a -= 1;
            }
            debug_assert!(i == n - 1 || s.as_ptr().offset(i).align_offset($boundary) == 0);
        }

        (i, n, s)
    }}
}

/// Handles the tail of the `slice` `s` of length `n` starting at index `i`
/// using a scalar loop:
macro_rules! is_sorted_handle_tail {
    ($s:ident, $n:ident, $i:ident) => {{
        // Handle the tail of the slice using the scalar algoirithm:
        while $i < $n - 1 {
            if $s.get_unchecked($i as usize)
                > $s.get_unchecked($i as usize + 1)
            {
                return false;
            }
            $i += 1;
        }
        debug_assert!($i == $n - 1);
        true
    }}
}

/// Specialization for iterator over &[i32] and increasing order.
///
/// If `std` is available, always include this and perform run-time feature
/// detection inside it to select the `SSE4.1` algorithm when the CPU supports
/// it.
///
/// If `std` is not available, include this specialization only when the
/// target supports `SSE4.1` at compile-time.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(any(feature = "use_std", any(target_feature = "sse4.1", target_feature = "avx2")))]
impl<'a> IsSortedBy<ord::Less> for slice::Iter<'a, i32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::Less) -> bool {
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn sse41_i32_impl<'a>(x: &mut slice::Iter<'a, i32>) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            let (mut i, n, s) = is_sorted_handle_unaligned_head_int!(x, i32, 16);
            let ap = |o| (s.as_ptr().offset(o)) as *const __m128i;

            // `i` points to the first element of the slice at a 16-byte
            // boundary. Use the SSE4.1 algorithm from HeroicKatora
            // https://www.reddit.com/r/cpp/comments/8bkaj3/is_sorted_using_simd_instructions/dx7jj8u/
            // to handle the body of the slice.
            const LVECS: isize = 4; // #of vectors in the loop
            const NVECS: isize = 1 + LVECS; // #vectors in the loop + current
            const NLANES: isize = 4; // #lanes in each vector
            const STRIDE: isize = NLANES * LVECS; // #vectors in the loop * NLANES
            const MIN_LEN: isize = NLANES * NVECS; // minimum #elements required for vectorization
            const EWIDTH: i32 = 4; // width of the vector elements
            if (n - i) >= MIN_LEN {
                // 5 vectors of 4 elements = 20
                let mut current = _mm_load_si128(ap(i + 0 * NLANES)); // [a0,..,a3]
                while i < n - STRIDE {
                    // == 16 | the last vector of current is the first of next
                    let next0 = _mm_load_si128(ap(i + 1 * NLANES)); // [a4,..,a7]
                    let next1 = _mm_load_si128(ap(i + 2 * NLANES)); // [a8,..,a11]
                    let next2 = _mm_load_si128(ap(i + 3 * NLANES)); // [a12,..,a15]
                    let next3 = _mm_load_si128(ap(i + 4 * NLANES)); // [a16,..a19]

                    let compare0 = _mm_alignr_epi8(next0, current, EWIDTH); // [a1,..,a4]
                    let compare1 = _mm_alignr_epi8(next1, next0, EWIDTH); // [a5,..,a8]
                    let compare2 = _mm_alignr_epi8(next2, next1, EWIDTH); // [a9,..,a12]
                    let compare3 = _mm_alignr_epi8(next3, next2, EWIDTH); // [a13,..,a16]

                    // [a0 > a1,..,a3 > a4]
                    let mask0 = _mm_cmpgt_epi32(current, compare0);
                    // [a4 > a5,..,a7 > a8]
                    let mask1 = _mm_cmpgt_epi32(next0, compare1);
                    // [a8 > a9,..,a11 > a12]
                    let mask2 = _mm_cmpgt_epi32(next1, compare2);
                    // [a12 > a13,..,a15 > a16]
                    let mask3 = _mm_cmpgt_epi32(next2, compare3);

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
                        return false;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            is_sorted_handle_tail!(s, n, i)
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn avx2_i32_impl<'a>(x: &mut slice::Iter<'a, i32>) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            let (mut i, n, s) = is_sorted_handle_unaligned_head_int!(x, i32, 32);
            let ap = |o| (s.as_ptr().offset(o)) as *const __m256i;

            // `i` points to the first element of the slice at a 16-byte
            // boundary. Use the SSE4.1 algorithm from HeroicKatora
            // https://www.reddit.com/r/cpp/comments/8bkaj3/is_sorted_using_simd_instructions/dx7jj8u/
            // to handle the body of the slice.
            const LVECS: isize = 4; // #of vectors in the loop
            const NVECS: isize = 1 + LVECS; // #vectors in the loop + current
            const NLANES: isize = 8; // #lanes in each vector
            const STRIDE: isize = NLANES * LVECS; // #vectors in the loop * NLANES
            const MIN_LEN: isize = NLANES * NVECS; // minimum #elements required for vectorization
            const EWIDTH: i32 = 4; // width of the vector elements
            if (n - i) >= MIN_LEN {
                let mut current = _mm256_load_si256(ap(i + 0 * NLANES)); // [a0,..,a7]
                while i < n - STRIDE {
                    // == 16 | the last vector of current is the first of next
                    let next0 = _mm256_load_si256(ap(i + 1 * NLANES)); // [a8,..,a16]
                    let next1 = _mm256_load_si256(ap(i + 2 * NLANES)); // [a16,..,a23]
                    let next2 = _mm256_load_si256(ap(i + 3 * NLANES)); // [a24,..,a31]
                    let next3 = _mm256_load_si256(ap(i + 4 * NLANES)); // [a32,..a39]

                    let compare0 = _mm256_alignr_epi8(next0, current, EWIDTH); // [a1,..,a8]
                    let compare1 = _mm256_alignr_epi8(next1, next0, EWIDTH); // [a9,..,a16]
                    let compare2 = _mm256_alignr_epi8(next2, next1, EWIDTH); // [a17,..,a23]
                    let compare3 = _mm256_alignr_epi8(next3, next2, EWIDTH); // [a25,..,a32]

                    // [a0 > a1,..,a7 > a8]
                    let mask0 = _mm256_cmpgt_epi32(current, compare0);
                    // [a8 > a9,..,a15 > a16]
                    let mask1 = _mm256_cmpgt_epi32(next0, compare1);
                    // [a16 > a17,..,a23 > a24]
                    let mask2 = _mm256_cmpgt_epi32(next1, compare2);
                    // [a24 > a25,..,a31 > a32]
                    let mask3 = _mm256_cmpgt_epi32(next2, compare3);

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
                        return false;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            is_sorted_handle_tail!(s, n, i)
        }

        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))] {
                sse41_i32_impl(self)
            }
            #[cfg(target_feature = "avx2")] {
                avx2_i32_impl(self)
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { avx2_i32_impl(self) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { sse41_i32_impl(self) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[u32] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(any(feature = "use_std", target_feature = "sse4.1"))]
impl<'a> IsSortedBy<ord::Less> for slice::Iter<'a, u32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::Less) -> bool {
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn sse41_i32_impl<'a>(x: &mut slice::Iter<'a, u32>) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            let (mut i, n, s) = is_sorted_handle_unaligned_head_int!(x, u32, 16);
            let ap = |o| (s.as_ptr().offset(o)) as *const __m128i;

            // `i` points to the first element of the slice at a 16-byte
            // boundary.
            const LVECS: isize = 4; // #of vectors in the loop
            const NVECS: isize = 1 + LVECS; // #vectors in the loop + current
            const NLANES: isize = 4; // #lanes in each vector
            const STRIDE: isize = NLANES * LVECS; // #vectors in the loop
            const MIN_LEN: isize = NLANES * NVECS; // minimum #elements required for vectorization
            const EWIDTH: i32 = 4; // width of the vector elements
            if (n - i) >= MIN_LEN {
                // 5 vectors of 4 elements = 20
                // note: an alternative algorithm would be to add
                // i32::min_value() to each element of the vector and then use
                // the same algorithm as for signed integers. That approach
                // proved slower than this approach.
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

                    // a <= b <=> a == minu(a,b):
                    // [a0 <= a1,..,a3 <= a4]
                    let mask0 = _mm_cmpeq_epi32(
                        current,
                        _mm_min_epu32(current, compare0),
                    );
                    // [a4 <= a5,..,a7 <= a8]
                    let mask1 =
                        _mm_cmpeq_epi32(next0, _mm_min_epu32(next0, compare1));
                    // [a8 <= a9,..,a11 <= a12]
                    let mask2 =
                        _mm_cmpeq_epi32(next1, _mm_min_epu32(next1, compare2));
                    // [a12 <= a13,..,a15 <= a16]
                    let mask3 =
                        _mm_cmpeq_epi32(next2, _mm_min_epu32(next2, compare3));

                    // mask = mask0 && mask1 && mask2 && mask3
                    let mask = _mm_and_si128(
                        _mm_and_si128(mask0, mask1),
                        _mm_and_si128(mask2, mask3),
                    );

                    // If the resulting mask has all bits set it means that all
                    // the <= comparisons were succesfull:
                    if _mm_test_all_ones(mask) == 0 {
                        return false;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            is_sorted_handle_tail!(s, n, i)
        }

        #[cfg(not(feature = "use_std"))]
        unsafe {
            sse41_i32_impl(self)
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("sse4.1") {
                unsafe { sse41_i32_impl(self) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[f32] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(any(feature = "use_std", target_feature = "sse4.1"))]
impl<'a> IsSortedBy<ord::PartialLessUnwrapped> for slice::Iter<'a, f32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::PartialLessUnwrapped) -> bool {
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn sse41_f32_impl<'a>(x: &mut slice::Iter<'a, f32>) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            let s = x.as_slice();
            let n = s.len() as isize;
            // If the slice has zero or one elements, it is sorted:
            if n < 2 {
                return true;
            }

            let ap = |i| s.as_ptr().offset(i);

            let mut i: isize = 0;

            // The first element of the slice might not be aligned to a
            // 16-byte boundary. Handle the elements until the
            // first 16-byte boundary using the scalar algorithm
            {
                let mut a =
                    s.as_ptr().align_offset(16) / mem::size_of::<f32>();
                while a > 0 && i < n - 1 {
                    if s.get_unchecked(i as usize)
                        > s.get_unchecked(i as usize + 1)
                    {
                        return false;
                    }
                    i += 1;
                    a -= 1;
                }
                debug_assert!(i == n - 1 || ap(i).align_offset(16) == 0);
            }

            // `i` points to the first element of the slice at a 16-byte
            // boundary. 
            const LVECS: isize = 4; // #of vectors in the loop
            const NVECS: isize = 1 + LVECS; // #vectors in the loop + current
            const NLANES: isize = 4; // #lanes in each vector
            const STRIDE: isize = NLANES * LVECS; // #vectors in the loop * NLANES
            const MIN_LEN: isize = NLANES * NVECS; // minimum #elements required for vectorization
            const EWIDTH: i32 = 4; // width of the vector elements
            if (n - i) >= MIN_LEN {
                // 5 vectors of 4 elements = 20
                let mut current = _mm_load_ps(ap(i + 0 * NLANES)); // [a0,..,a3]
                while i < n - STRIDE {
                    // == 16 | the last vector of current is the first of next
                    let next0 = _mm_load_ps(ap(i + 1 * NLANES)); // [a4,..,a7]
                    let next1 = _mm_load_ps(ap(i + 2 * NLANES)); // [a8,..,a11]
                    let next2 = _mm_load_ps(ap(i + 3 * NLANES)); // [a12,..,a15]
                    let next3 = _mm_load_ps(ap(i + 4 * NLANES)); // [a16,..a19]

                    let compare0 = _mm_alignr_epi8(
                        mem::transmute(next0),
                        mem::transmute(current),
                        EWIDTH,
                    ); // [a1,..,a4]
                    let compare1 = _mm_alignr_epi8(
                        mem::transmute(next1),
                        mem::transmute(next0),
                        EWIDTH,
                    ); // [a5,..,a8]
                    let compare2 = _mm_alignr_epi8(
                        mem::transmute(next2),
                        mem::transmute(next1),
                        EWIDTH,
                    ); // [a9,..,a12]
                    let compare3 = _mm_alignr_epi8(
                        mem::transmute(next3),
                        mem::transmute(next2),
                        EWIDTH,
                    ); // [a13,..,a16]

                    // [a0 <= a1,..,a3 <= a4]
                    let mask0 =
                        _mm_cmple_ps(current, mem::transmute(compare0));
                    // [a4 <= a5,..,a7 <= a8]
                    let mask1 = _mm_cmple_ps(next0, mem::transmute(compare1));
                    // [a8 <= a9,..,a11 <= a12]
                    let mask2 = _mm_cmple_ps(next1, mem::transmute(compare2));
                    // [a12 <= a13,..,a15 <= a16]
                    let mask3 = _mm_cmple_ps(next2, mem::transmute(compare3));

                    // mask = mask0 | mask1 | mask2 | mask3
                    let mask = _mm_and_ps(
                        _mm_and_ps(mask0, mask1),
                        _mm_and_ps(mask2, mask3),
                    );

                    // mask & mask == 0: if some gt comparison was true, the
                    // mask will have some bits set. The result of bitwise & of
                    // the mask with itself is only zero if all of the bits of
                    // the mask are zero. Therefore, if some comparison
                    // succeeded, there will be some non-zero bit, and all
                    // zeros would return false (aka 0).
                    if _mm_test_all_ones(mem::transmute(mask)) == 0 {
                        return false;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            is_sorted_handle_tail!(s, n, i)
        }

        #[cfg(not(feature = "use_std"))]
        unsafe {
            sse41_f32_impl(self)
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("sse4.1") {
                unsafe { sse41_f32_impl(self) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[i16] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(any(feature = "use_std", any(target_feature = "sse4.1", target_feature = "avx2")))]
impl<'a> IsSortedBy<ord::Less> for slice::Iter<'a, i16> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::Less) -> bool {
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn sse41_i16_impl<'a>(x: &mut slice::Iter<'a, i16>) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            let (mut i, n, s) = is_sorted_handle_unaligned_head_int!(x, i16, 16);
            let ap = |o| (s.as_ptr().offset(o)) as *const __m128i;

            // `i` points to the first element of the slice at a 16-byte
            // boundary. 
            const LVECS: isize = 4; // #of vectors in the loop
            const NVECS: isize = 1 + LVECS; // #vectors in the loop + current
            const NLANES: isize = 8; // #lanes in each vector
            const STRIDE: isize = NLANES * LVECS; // #vectors in the loop * NLANES
            const MIN_LEN: isize = NLANES * NVECS; // minimum #elements required for vectorization
            const EWIDTH: i32 = 2; // width of the vector elements
            if (n - i) >= MIN_LEN {
                // == 40: no vectors * no lanes
                let mut current = _mm_load_si128(ap(i + 0 * NLANES)); // [a0..a7]
                while i < n - STRIDE {
                    // == 32 | the last vector of current is the first of next
                    let next0 = _mm_load_si128(ap(i + 1 * NLANES)); // [a8..a15]
                    let next1 = _mm_load_si128(ap(i + 2 * NLANES)); // [a16..a23]
                    let next2 = _mm_load_si128(ap(i + 3 * NLANES)); // [a24..a31]
                    let next3 = _mm_load_si128(ap(i + 4 * NLANES)); // [a32..a39]

                    let compare0 = _mm_alignr_epi8(next0, current, EWIDTH); // [a1..a8]
                    let compare1 = _mm_alignr_epi8(next1, next0, EWIDTH); // [a9..a16]
                    let compare2 = _mm_alignr_epi8(next2, next1, EWIDTH); // [a17..a24]
                    let compare3 = _mm_alignr_epi8(next3, next2, EWIDTH); // [a25..a32]

                    let mask0 = _mm_cmpgt_epi16(current, compare0); // [a0 > a1,..,a7 > a8]
                    let mask1 = _mm_cmpgt_epi16(next0, compare1); // [a8 > a9,..,a15 > a16]
                    let mask2 = _mm_cmpgt_epi16(next1, compare2); // [a16 > a17,..,a23 > a24]
                    let mask3 = _mm_cmpgt_epi16(next2, compare3); // [a24 > a25,..,a31 > a32]

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
                        return false;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            is_sorted_handle_tail!(s, n, i)
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn avx2_i16_impl<'a>(x: &mut slice::Iter<'a, i16>) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            let (mut i, n, s) = is_sorted_handle_unaligned_head_int!(x, i16, 32);
            let ap = |o| (s.as_ptr().offset(o)) as *const __m256i;

            // `i` points to the first element of the slice at a 16-byte
            // boundary. Use the SSE4.1 algorithm from HeroicKatora
            // https://www.reddit.com/r/cpp/comments/8bkaj3/is_sorted_using_simd_instructions/dx7jj8u/
            // to handle the body of the slice.
            const LVECS: isize = 4; // #of vectors in the loop
            const NVECS: isize = 1 + LVECS; // #vectors in the loop + current
            const NLANES: isize = 16; // #lanes in each vector
            const STRIDE: isize = NLANES * LVECS; // #vectors in the loop * NLANES
            const MIN_LEN: isize = NLANES * NVECS; // minimum #elements required for vectorization
            const EWIDTH: i32 = 2; // width of the vector elements
            if (n - i) >= MIN_LEN {
                let mut current = _mm256_load_si256(ap(i + 0 * NLANES)); // [a0,..,a15]
                while i < n - STRIDE {
                    // == 16 | the last vector of current is the first of next
                    let next0 = _mm256_load_si256(ap(i + 1 * NLANES)); // [a16,..,a31]
                    let next1 = _mm256_load_si256(ap(i + 2 * NLANES)); // [a32,..,a47]
                    let next2 = _mm256_load_si256(ap(i + 3 * NLANES)); // [a48,..,a63]
                    let next3 = _mm256_load_si256(ap(i + 4 * NLANES)); // [a64,..a79]

                    let compare0 = _mm256_alignr_epi16(next0, current, EWIDTH); // [a1,..,a16]
                    let compare1 = _mm256_alignr_epi16(next1, next0, EWIDTH); // [a17,..,a32]
                    let compare2 = _mm256_alignr_epi16(next2, next1, EWIDTH); // [a33,..,a48]
                    let compare3 = _mm256_alignr_epi16(next3, next2, EWIDTH); // [a49,..,a64]

                    // [a0 > a1,..,a15 > a16]
                    let mask0 = _mm256_cmpgt_epi32(current, compare0);
                    // [a16 > a17,..,a31 > a32]
                    let mask1 = _mm256_cmpgt_epi32(next0, compare1);
                    // [a32 > a33,..,a47 > a48]
                    let mask2 = _mm256_cmpgt_epi32(next1, compare2);
                    // [a48 > a49,..,a63 > a64]
                    let mask3 = _mm256_cmpgt_epi32(next2, compare3);

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
                        return false;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            is_sorted_handle_tail!(s, n, i)
        }

        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))] {
                sse41_i16_impl(self)
            }
            #[cfg(target_feature = "avx2")] {
                avx2_i16_impl(self)
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { avx2_i16_impl(self) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { sse41_i16_impl(self) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[u16] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(any(feature = "use_std", target_feature = "sse4.1"))]
impl<'a> IsSortedBy<ord::Less> for slice::Iter<'a, u16> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::Less) -> bool {
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn sse41_u16_impl<'a>(x: &mut slice::Iter<'a, u16>) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            let (mut i, n, s) = is_sorted_handle_unaligned_head_int!(x, u16, 16);
            let ap = |o| (s.as_ptr().offset(o)) as *const __m128i;

            // `i` points to the first element of the slice at a 16-byte
            // boundary. 
            const LVECS: isize = 4; // #of vectors in the loop
            const NVECS: isize = 1 + LVECS; // #vectors in the loop + current
            const NLANES: isize = 8; // #lanes in each vector
            const STRIDE: isize = NLANES * LVECS; // #vectors in the loop * NLANES
            const MIN_LEN: isize = NLANES * NVECS; // minimum #elements required for vectorization
            const EWIDTH: i32 = 2; // width of the vector elements
            if (n - i) >= MIN_LEN {
                // == 40: no vectors * no lanes
                let mut current = _mm_load_si128(ap(i + 0 * NLANES)); // [a0..a7]
                while i < n - STRIDE {
                    // == 32 | the last vector of current is the first of next
                    let next0 = _mm_load_si128(ap(i + 1 * NLANES)); // [a8..a15]
                    let next1 = _mm_load_si128(ap(i + 2 * NLANES)); // [a16..a23]
                    let next2 = _mm_load_si128(ap(i + 3 * NLANES)); // [a24..a31]
                    let next3 = _mm_load_si128(ap(i + 4 * NLANES)); // [a32..a39]

                    let compare0 = _mm_alignr_epi8(next0, current, EWIDTH); // [a1..a8]
                    let compare1 = _mm_alignr_epi8(next1, next0, EWIDTH); // [a9..a16]
                    let compare2 = _mm_alignr_epi8(next2, next1, EWIDTH); // [a17..a24]
                    let compare3 = _mm_alignr_epi8(next3, next2, EWIDTH); // [a25..a32]

                    // a <= b <=> a == minu(a,b):
                    // [a0 <= a1,..,a7 <= a8]
                    let mask0 = _mm_cmpeq_epi16(
                        current,
                        _mm_min_epu16(current, compare0),
                    );
                    // [a8 <= a9,..,a15 <= a16]
                    let mask1 =
                        _mm_cmpeq_epi16(next0, _mm_min_epu16(next0, compare1));
                    // [a16 <= a17,.., a23 <= a24]
                    let mask2 =
                        _mm_cmpeq_epi16(next1, _mm_min_epu16(next1, compare2));
                    // [a24 <= a25,..,a31 <= a32]
                    let mask3 =
                        _mm_cmpeq_epi16(next2, _mm_min_epu16(next2, compare3));

                    // mask = mask0 && mask1 && mask2 && mask3
                    let mask = _mm_and_si128(
                        _mm_and_si128(mask0, mask1),
                        _mm_and_si128(mask2, mask3),
                    );

                    // If the resulting mask has all bits set it means that all
                    // the <= comparisons were succesfull:
                    if _mm_test_all_ones(mask) == 0 {
                        return false;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            is_sorted_handle_tail!(s, n, i)
        }

        #[cfg(not(feature = "use_std"))]
        unsafe {
            sse41_u16_impl(self)
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("sse4.1") {
                unsafe { sse41_u16_impl(self) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[i8] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(any(feature = "use_std", any(target_feature = "sse4.1", target_feature = "avx2")))]
impl<'a> IsSortedBy<ord::Less> for slice::Iter<'a, i8> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::Less) -> bool {
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn sse41_i8_impl<'a>(x: &mut slice::Iter<'a, i8>) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            let (mut i, n, s) = is_sorted_handle_unaligned_head_int!(x, i8, 16);
            let ap = |o| (s.as_ptr().offset(o)) as *const __m128i;

            // `i` points to the first element of the slice at a 16-byte
            // boundary. 
            const LVECS: isize = 4; // #of vectors in the loop
            const NVECS: isize = 1 + LVECS; // #vectors in the loop + current
            const NLANES: isize = 16; // #lanes in each vector
            const STRIDE: isize = NLANES * LVECS; // #vectors in the loop * NLANES
            const MIN_LEN: isize = NLANES * NVECS; // minimum #elements required for vectorization
            const EWIDTH: i32 = 1; // width of the vector elements
            if (n - i) >= MIN_LEN {
                // == 40: no vectors * no lanes
                let mut current = _mm_load_si128(ap(i + 0 * NLANES)); // [a0..a7]
                while i < n - STRIDE {
                    // == 32 | the last vector of current is the first of next
                    let next0 = _mm_load_si128(ap(i + 1 * NLANES)); // [a8..a15]
                    let next1 = _mm_load_si128(ap(i + 2 * NLANES)); // [a16..a23]
                    let next2 = _mm_load_si128(ap(i + 3 * NLANES)); // [a24..a31]
                    let next3 = _mm_load_si128(ap(i + 4 * NLANES)); // [a32..a39]

                    let compare0 = _mm_alignr_epi8(next0, current, EWIDTH); // [a1..a8]
                    let compare1 = _mm_alignr_epi8(next1, next0, EWIDTH); // [a9..a16]
                    let compare2 = _mm_alignr_epi8(next2, next1, EWIDTH); // [a17..a24]
                    let compare3 = _mm_alignr_epi8(next3, next2, EWIDTH); // [a25..a32]

                    let mask0 = _mm_cmpgt_epi8(current, compare0); // [a0 > a1,..,a7 > a8]
                    let mask1 = _mm_cmpgt_epi8(next0, compare1); // [a8 > a9,..,a15 > a16]
                    let mask2 = _mm_cmpgt_epi8(next1, compare2); // [a16 > a17,..,a23 > a24]
                    let mask3 = _mm_cmpgt_epi8(next2, compare3); // [a24 > a25,..,a31 > a32]

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
                        return false;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            is_sorted_handle_tail!(s, n, i)
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn avx2_i8_impl<'a>(x: &mut slice::Iter<'a, i8>) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            let (mut i, n, s) = is_sorted_handle_unaligned_head_int!(x, i8, 32);
            let ap = |o| (s.as_ptr().offset(o)) as *const __m256i;

            // `i` points to the first element of the slice at a 16-byte
            // boundary. Use the SSE4.1 algorithm from HeroicKatora
            // https://www.reddit.com/r/cpp/comments/8bkaj3/is_sorted_using_simd_instructions/dx7jj8u/
            // to handle the body of the slice.
            const LVECS: isize = 4; // #of vectors in the loop
            const NVECS: isize = 1 + LVECS; // #vectors in the loop + current
            const NLANES: isize = 32; // #lanes in each vector
            const STRIDE: isize = NLANES * LVECS; // #vectors in the loop * NLANES
            const MIN_LEN: isize = NLANES * NVECS; // minimum #elements required for vectorization
            const EWIDTH: i32 = 1; // width of the vector elements
            if (n - i) >= MIN_LEN {
                let mut current = _mm256_load_si256(ap(i + 0 * NLANES)); // [a0,..,a31]
                while i < n - STRIDE {
                    // == 16 | the last vector of current is the first of next
                    let next0 = _mm256_load_si256(ap(i + 1 * NLANES)); // [a32,..,a63]
                    let next1 = _mm256_load_si256(ap(i + 2 * NLANES)); // [a64,..,a95]
                    let next2 = _mm256_load_si256(ap(i + 3 * NLANES)); // [a96,..,a127]
                    let next3 = _mm256_load_si256(ap(i + 4 * NLANES)); // [a128,..a159]

                    let compare0 = _mm256_alignr_epi8(next0, current, EWIDTH); // [a1,..,a32]
                    let compare1 = _mm256_alignr_epi8(next1, next0, EWIDTH); // [a33,..,a64]
                    let compare2 = _mm256_alignr_epi8(next2, next1, EWIDTH); // [a65,..,a96]
                    let compare3 = _mm256_alignr_epi8(next3, next2, EWIDTH); // [a97,..,a128]

                    // [a0 > a1,..,a31 > a32]
                    let mask0 = _mm256_cmpgt_epi32(current, compare0);
                    // [a32 > a33,..,a63 > a64]
                    let mask1 = _mm256_cmpgt_epi32(next0, compare1);
                    // [a64 > a65,..,a95 > a96]
                    let mask2 = _mm256_cmpgt_epi32(next1, compare2);
                    // [a96 > a97,..,a127 > a128]
                    let mask3 = _mm256_cmpgt_epi32(next2, compare3);

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
                        return false;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            is_sorted_handle_tail!(s, n, i)
        }


        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))] {
                sse41_i8_impl(self)
            }
            #[cfg(target_feature = "avx2")] {
                avx2_i8_impl(self)
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { avx2_i8_impl(self) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { sse41_i8_impl(self) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[u8] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(any(feature = "use_std", target_feature = "sse4.1"))]
impl<'a> IsSortedBy<ord::Less> for slice::Iter<'a, u8> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::Less) -> bool {
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn sse41_u8_impl<'a>(x: &mut slice::Iter<'a, u8>) -> bool {
            #[cfg(target_arch = "x86")]
            use arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use arch::x86_64::*;

            let (mut i, n, s) = is_sorted_handle_unaligned_head_int!(x, i8, 16);
            let ap = |o| (s.as_ptr().offset(o)) as *const __m128i;

            // `i` points to the first element of the slice at a 16-byte
            // boundary. 
            const LVECS: isize = 4; // #of vectors in the loop
            const NVECS: isize = 1 + LVECS; // #vectors in the loop + current
            const NLANES: isize = 16; // #lanes in each vector
            const STRIDE: isize = NLANES * LVECS; // #vectors in the loop * NLANES
            const MIN_LEN: isize = NLANES * NVECS; // minimum #elements required for vectorization
            const EWIDTH: i32 = 1; // width of the vector elements
            if (n - i) >= MIN_LEN {
                // == 40: no vectors * no lanes
                let mut current = _mm_load_si128(ap(i + 0 * NLANES)); // [a0..a7]
                while i < n - STRIDE {
                    // == 32 | the last vector of current is the first of next
                    let next0 = _mm_load_si128(ap(i + 1 * NLANES)); // [a8..a15]
                    let next1 = _mm_load_si128(ap(i + 2 * NLANES)); // [a16..a23]
                    let next2 = _mm_load_si128(ap(i + 3 * NLANES)); // [a24..a31]
                    let next3 = _mm_load_si128(ap(i + 4 * NLANES)); // [a32..a39]

                    let compare0 = _mm_alignr_epi8(next0, current, EWIDTH); // [a1..a8]
                    let compare1 = _mm_alignr_epi8(next1, next0, EWIDTH); // [a9..a16]
                    let compare2 = _mm_alignr_epi8(next2, next1, EWIDTH); // [a17..a24]
                    let compare3 = _mm_alignr_epi8(next3, next2, EWIDTH); // [a25..a32]

                    // a <= b <=> a == minu(a,b):
                    // [a0 <= a1,..,a7 <= a8]
                    let mask0 = _mm_cmpeq_epi8(
                        current,
                        _mm_min_epu8(current, compare0),
                    );
                    // [a8 <= a9,..,a15 <= a16]
                    let mask1 =
                        _mm_cmpeq_epi8(next0, _mm_min_epu8(next0, compare1));
                    // [a16 <= a17,.., a23 <= a24]
                    let mask2 =
                        _mm_cmpeq_epi8(next1, _mm_min_epu8(next1, compare2));
                    // [a24 <= a25,..,a31 <= a32]
                    let mask3 =
                        _mm_cmpeq_epi8(next2, _mm_min_epu8(next2, compare3));

                    // mask = mask0 && mask1 && mask2 && mask3
                    let mask = _mm_and_si128(
                        _mm_and_si128(mask0, mask1),
                        _mm_and_si128(mask2, mask3),
                    );

                    // If the resulting mask has all bits set it means that all
                    // the <= comparisons were succesfull:
                    if _mm_test_all_ones(mask) == 0 {
                        return false;
                    }

                    current = next3;

                    i += STRIDE;
                }
            }

            is_sorted_handle_tail!(s, n, i)
        }

        #[cfg(not(feature = "use_std"))]
        unsafe {
            sse41_u8_impl(self)
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("sse4.1") {
                unsafe { sse41_u8_impl(self) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use IsSorted;
    extern crate rand;
    extern crate std;
    use self::rand::{thread_rng, Rng};

    use self::std::vec::Vec;

    macro_rules! test_float {
        ($name:ident, $id:ident) => {
            #[test]
            fn $name() {
                #[cfg(feature = "unstable")]
                macro_rules! cmp {
                    () => {
                        ::PartialLessUnwrapped
                    };
                }

                #[cfg(not(feature = "unstable"))]
                macro_rules! cmp {
                    () => {
                        |a, b| a.partial_cmp(b).unwrap()
                    };
                }

                fn random_vec(x: usize) -> Vec<$id> {
                    let mut vec = Vec::with_capacity(x);
                    let mut rng = thread_rng();
                    for _ in 0..x {
                        let val: $id = rng.gen_range(0. as $id, 1. as $id);
                        vec.push(val);
                    }
                    vec
                }

                let x = [1., 2., 3., 4.];
                assert!(x.iter().is_sorted_by(cmp!()));

                let x: [$id; 0] = [];
                assert!(x.iter().is_sorted_by(cmp!()));

                let x = [0 as $id];
                assert!(x.iter().is_sorted_by(cmp!()));

                let max = ::std::$id::INFINITY;
                let min = -max;

                let x = [min, max];
                assert!(x.iter().is_sorted_by(cmp!()));

                let x = [1 as $id, 2., 3., 4.];
                assert!(x.iter().is_sorted_by(cmp!()));

                let x = [1 as $id, 3., 2., 4.];
                assert!(!x.iter().is_sorted_by(cmp!()));

                let x = [4 as $id, 3., 2., 1.];
                assert!(!x.iter().is_sorted_by(cmp!()));

                let x = [4 as $id, 4., 4., 4.];
                assert!(x.iter().is_sorted_by(cmp!()));

                let mut v = Vec::new();
                for _ in 0..2 {
                    v.push(min);
                }
                for _ in 0..2 {
                    v.push(max);
                }
                assert!(v.as_slice().iter().is_sorted_by(cmp!()));

                let mut v = Vec::new();
                for _ in 0..4 {
                    v.push(min);
                }
                for _ in 0..5 {
                    v.push(max);
                }
                assert!(v.as_slice().iter().is_sorted_by(cmp!()));

                for i in 0..1_000 {
                    let mut vec: Vec<$id> = random_vec(i);
                    vec.sort_by(cmp!());
                    assert!(
                        vec.as_slice().iter().is_sorted_by(cmp!()),
                        "is_sorted0: {:?}",
                        vec
                    );
                    if i > 4 {
                        vec.push(min);
                        assert!(
                            !vec.as_slice().iter().is_sorted_by(cmp!()),
                            "!is_sorted1: {:?}",
                            vec
                        );
                        vec.insert(i / 3 * 2, min);
                        assert!(
                            !vec.as_slice().iter().is_sorted_by(cmp!()),
                            "!is_sorted2: {:?}",
                            vec
                        );
                        vec.insert(0, max);
                        assert!(
                            !vec.as_slice().iter().is_sorted_by(cmp!()),
                            "!is_sorted3: {:?}",
                            vec
                        );
                    }
                }
            }
        };
    }

    test_float!(test_f32, f32);
    test_float!(test_f64, f64);

    macro_rules! small {
        ($name:ident, $id:ident) => {
            #[test]
            fn $name() {
                fn random_vec(x: usize) -> Vec<$id> {
                    let mut vec = Vec::with_capacity(x);
                    let mut rng = thread_rng();
                    for _ in 0..x {
                        let val: $id =
                            rng.gen_range($id::min_value(), $id::max_value());
                        vec.push(val);
                    }
                    vec
                }

                let x: [$id; 0] = [];
                assert!(x.iter().is_sorted());

                let x = [0 as $id];
                assert!(x.iter().is_sorted());

                let x = [$id::min_value(), $id::max_value()];
                assert!(x.iter().is_sorted());

                let x = [1 as $id, 2, 3, 4];
                assert!(x.iter().is_sorted());

                let x = [1 as $id, 3, 2, 4];
                assert!(!x.iter().is_sorted());

                let x = [4 as $id, 3, 2, 1];
                assert!(!x.iter().is_sorted());

                let x = [4 as $id, 4, 4, 4];
                assert!(x.iter().is_sorted());

                let min = $id::min_value();
                let max = $id::max_value();

                let mut v = Vec::new();
                for _ in 0..2 {
                    v.push(min);
                }
                for _ in 0..2 {
                    v.push(max);
                }
                assert!(v.as_slice().iter().is_sorted());

                let mut v = Vec::new();
                for _ in 0..4 {
                    v.push(min);
                }
                for _ in 0..5 {
                    v.push(max);
                }
                assert!(v.as_slice().iter().is_sorted());

                for i in 0..1_000 {
                    let mut vec: Vec<$id> = random_vec(i);
                    vec.sort();
                    assert!(
                        vec.as_slice().iter().is_sorted(),
                        "is_sorted0: {:?}",
                        vec
                    );
                    if i > 4 {
                        vec.push($id::min_value());
                        assert!(
                            !vec.as_slice().iter().is_sorted(),
                            "!is_sorted1: {:?}",
                            vec
                        );
                        vec.insert(i / 3 * 2, $id::min_value());
                        assert!(
                            !vec.as_slice().iter().is_sorted(),
                            "!is_sorted2: {:?}",
                            vec
                        );
                        vec.insert(0, $id::max_value());
                        assert!(
                            !vec.as_slice().iter().is_sorted(),
                            "!is_sorted3: {:?}",
                            vec
                        );
                    }
                }
            }
        };
    }
    small!(small_i8, i8);
    small!(small_u8, u8);
    small!(small_i16, i16);
    small!(small_u16, u16);
    small!(small_u32, u32);
    small!(small_i32, i32);

    macro_rules! large {
        ($name:ident, $id:ident) => {
            #[test]
            fn $name() {
                let n = 1_000;
                let mut v = Vec::new();
                for i in 0..n {
                    v.push(i as $id);
                }
                v.sort();
                {
                    let s: &[$id] = v.as_slice();
                    assert!(s.iter().is_sorted());
                }

                v.push(0);
                {
                    let s: &[$id] = v.as_slice();
                    assert!(!s.iter().is_sorted());
                }
                for i in &mut v {
                    *i = 42;
                }
                {
                    let s: &[$id] = v.as_slice();
                    assert!(s.iter().is_sorted());
                }
                let min = $id::min_value();
                let max = $id::max_value();

                for i in &mut v {
                    *i = $id::max_value();
                }
                {
                    let s: &[$id] = v.as_slice();
                    assert!(s.iter().is_sorted());
                }
                v.push(min);
                {
                    let s: &[$id] = v.as_slice();
                    assert!(!s.iter().is_sorted());
                }
                for i in &mut v {
                    *i = $id::min_value();
                }
                {
                    let s: &[$id] = v.as_slice();
                    assert!(s.iter().is_sorted());
                }
                let mut v = Vec::new();
                for _ in 0..n {
                    v.push(min);
                }
                for _ in 0..n {
                    v.push(max);
                }
                assert!(v.as_slice().iter().is_sorted());
            }
        };
    }
    large!(i8_large, i8);
    large!(u8_large, u8);
    large!(i16_large, i16);
    large!(u16_large, u16);
    large!(u32_large, u32);
    large!(i32_large, i32);
}
