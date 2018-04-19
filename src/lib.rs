//! Extends `Iterator` with three algorithms, `is_sorted`, `is_sorted_by`, and
//! `is_sorted_by_key` that check whether the elements of an `Iterator` are
//! sorted in `O(N)` time and `O(1)` space.
//!
//! To enable explicitly-vectorized implementations enable the `unstable`
//! nightly-only feature and use the typed comparators: `Increasing` and
//! `Decreasing`.

// If the `use_std` feature is not enable, compile for `no_std`:
#![cfg_attr(not(feature = "use_std"), no_std)]
// If the `unstable` feature is enabled, enable nightly-only features:
#![cfg_attr(
    feature = "unstable",
    feature(specialization, fn_traits, unboxed_closures, stdsimd, align_offset)
)]

#[allow(unused_imports, unused_macros)]
#[cfg(not(feature = "use_std"))]
use core as std;

use std::cmp;

#[cfg(feature = "unstable")]
use std::{arch, mem, slice};

use cmp::Ordering;

#[cfg(feature = "unstable")]
mod ord;
#[cfg(feature = "unstable")]
pub use self::ord::*;

#[cfg(feature = "unstable")]
#[macro_use]
mod macros;

#[cfg(feature = "unstable")]
mod signed;

#[cfg(feature = "unstable")]
mod unsigned;

#[cfg(feature = "unstable")]
mod floats;

/// Extends `Iterator` with `is_sorted`, `is_sorted_by`, and
/// `is_sorted_by_key`.
pub trait IsSorted: Iterator {
    /// Returns `true` if the elements of the iterator are sorted in increasing
    /// order according to `<Self::Item as PartialOrd>::partial_cmp`.
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
        Self::Item: PartialOrd,
    {
        #[cfg(feature = "unstable")]
        {
            self.is_sorted_by(Increasing)
        }
        #[cfg(not(feature = "unstable"))]
        {
            self.is_sorted_by(<Self::Item as PartialOrd>::partial_cmp)
        }
    }

    /// Returns `true` if the elements of the iterator
    /// are sorted according to the `compare` function.
    ///
    /// ```
    /// # use is_sorted::IsSorted;
    /// # use std::cmp::Ordering;
    /// // Is an iterator sorted in decreasing order?
    /// fn decr<T: PartialOrd>(a: &T, b: &T) -> Option<Ordering> {
    ///     a.partial_cmp(b).map(|v| v.reverse())
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
        F: FnMut(&Self::Item, &Self::Item) -> Option<Ordering>,
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
        B: PartialOrd,
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
    F: FnMut(&I::Item, &I::Item) -> Option<Ordering>,
{
    <I as IsSortedBy<F>>::is_sorted_by(iter, compare)
}

// This trait is used to provide specialized implementations of `is_sorted_by`
// for different (Iterator,Cmp) pairs:
trait IsSortedBy<F>: Iterator
where
    F: FnMut(&Self::Item, &Self::Item) -> Option<Ordering>,
{
    fn is_sorted_by(&mut self, compare: F) -> bool;
}

// This blanket implementation acts as the fall-back, and just forwards to the
// scalar implementation of the algorithm.
impl<I, F> IsSortedBy<F> for I
where
    I: Iterator,
    F: FnMut(&I::Item, &I::Item) -> Option<Ordering>,
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
///
/// It just forwards to `Iterator::all`.
#[inline]
fn is_sorted_by_scalar_impl<I, F>(iter: &mut I, mut compare: F) -> bool
where
    I: Iterator,
    F: FnMut(&I::Item, &I::Item) -> Option<Ordering>,
{
    let first = iter.next();
    if let Some(mut first) = first {
        return iter.all(|second| {
            if let Some(ord) = compare(&first, &second) {
                if ord != Ordering::Greater {
                    first = second;
                    return true;
                }
            }
            false
        });
    }
    true
}

/// Specialization for iterator over &[i64] and increasing order.
///
/// On nightly:
///
/// * if `std` is available, always include this and select the best
/// implementation using run-time feature detection.
///
/// * otherwise, include this specialization only when we can statically
/// determine that the target supports one of the implementations available
/// (using compile-time feature detection).
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.2", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Increasing> for slice::Iter<'a, i64> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Increasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::signed::sse42::is_sorted_lt_i64(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::signed::avx2::is_sorted_lt_i64(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::signed::avx2::is_sorted_lt_i64(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.2") {
                unsafe { ::signed::sse42::is_sorted_lt_i64(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[i64] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.2", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Decreasing> for slice::Iter<'a, i64> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Decreasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::signed::sse42::is_sorted_gt_i64(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::signed::avx2::is_sorted_gt_i64(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::signed::avx2::is_sorted_gt_i64(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.2") {
                unsafe { ::signed::sse42::is_sorted_gt_i64(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[f64] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx")
    )
)]
impl<'a> IsSortedBy<ord::types::Increasing> for slice::Iter<'a, f64> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Increasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx"))]
            {
                unsafe { ::floats::sse41::is_sorted_lt_f64(self.as_slice()) }
            }
            #[cfg(target_feature = "avx")]
            {
                unsafe { ::floats::avx::is_sorted_lt_f64(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe { ::floats::avx::is_sorted_lt_f64(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::floats::sse41::is_sorted_lt_f64(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[f64] and decreasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx")
    )
)]
impl<'a> IsSortedBy<ord::types::Decreasing> for slice::Iter<'a, f64> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Decreasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx"))]
            {
                unsafe { ::floats::sse41::is_sorted_gt_f64(self.as_slice()) }
            }
            #[cfg(target_feature = "avx")]
            {
                unsafe { ::floats::avx::is_sorted_gt_f64(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe { ::floats::avx::is_sorted_gt_f64(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::floats::sse41::is_sorted_gt_f64(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[i32] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Increasing> for slice::Iter<'a, i32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Increasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::signed::sse41::is_sorted_lt_i32(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::signed::avx2::is_sorted_lt_i32(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::signed::avx2::is_sorted_lt_i32(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::signed::sse41::is_sorted_lt_i32(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[i32] and decreasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Decreasing> for slice::Iter<'a, i32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Decreasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::signed::sse41::is_sorted_gt_i32(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::signed::avx2::is_sorted_gt_i32(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::signed::avx2::is_sorted_gt_i32(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::signed::sse41::is_sorted_gt_i32(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[u32] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Increasing> for slice::Iter<'a, u32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Increasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::unsigned::sse41::is_sorted_lt_u32(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::unsigned::avx2::is_sorted_lt_u32(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::unsigned::avx2::is_sorted_lt_u32(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::unsigned::sse41::is_sorted_lt_u32(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[u32] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Decreasing> for slice::Iter<'a, u32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Decreasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::unsigned::sse41::is_sorted_gt_u32(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::unsigned::avx2::is_sorted_gt_u32(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::unsigned::avx2::is_sorted_gt_u32(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::unsigned::sse41::is_sorted_gt_u32(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[f32] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx")
    )
)]
impl<'a> IsSortedBy<ord::types::Increasing> for slice::Iter<'a, f32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Increasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx"))]
            {
                unsafe { ::floats::sse41::is_sorted_lt_f32(self.as_slice()) }
            }
            #[cfg(target_feature = "avx")]
            {
                unsafe { ::floats::avx::is_sorted_lt_f32(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe { ::floats::avx::is_sorted_lt_f32(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::floats::sse41::is_sorted_lt_f32(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[f32] and decreasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx")
    )
)]
impl<'a> IsSortedBy<ord::types::Decreasing> for slice::Iter<'a, f32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Decreasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx"))]
            {
                unsafe { ::floats::sse41::is_sorted_gt_f32(self.as_slice()) }
            }
            #[cfg(target_feature = "avx")]
            {
                unsafe { ::floats::avx::is_sorted_gt_f32(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe { ::floats::avx::is_sorted_gt_f32(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::floats::sse41::is_sorted_gt_f32(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[i16] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Increasing> for slice::Iter<'a, i16> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Increasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::signed::sse41::is_sorted_lt_i16(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::signed::avx2::is_sorted_lt_i16(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::signed::avx2::is_sorted_lt_i16(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::signed::sse41::is_sorted_lt_i16(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[i16] and decreasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Decreasing> for slice::Iter<'a, i16> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Decreasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::signed::sse41::is_sorted_gt_i16(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::signed::avx2::is_sorted_gt_i16(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::signed::avx2::is_sorted_gt_i16(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::signed::sse41::is_sorted_gt_i16(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[u16] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Increasing> for slice::Iter<'a, u16> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Increasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::unsigned::sse41::is_sorted_lt_u16(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::unsigned::avx2::is_sorted_lt_u16(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::unsigned::avx2::is_sorted_lt_u16(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::unsigned::sse41::is_sorted_lt_u16(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[u16] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Decreasing> for slice::Iter<'a, u16> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Decreasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::unsigned::sse41::is_sorted_gt_u16(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::unsigned::avx2::is_sorted_gt_u16(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::unsigned::avx2::is_sorted_gt_u16(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::unsigned::sse41::is_sorted_gt_u16(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[i8] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Increasing> for slice::Iter<'a, i8> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Increasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::signed::sse41::is_sorted_lt_i8(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::signed::avx2::is_sorted_lt_i8(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::signed::avx2::is_sorted_lt_i8(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::signed::sse41::is_sorted_lt_i8(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[i8] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Decreasing> for slice::Iter<'a, i8> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Decreasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::signed::sse41::is_sorted_gt_i8(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::signed::avx2::is_sorted_gt_i8(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::signed::avx2::is_sorted_gt_i8(self.as_slice()) }
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { ::signed::sse41::is_sorted_gt_i8(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[u8] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Increasing> for slice::Iter<'a, u8> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Increasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::unsigned::sse41::is_sorted_lt_u8(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::unsigned::avx2::is_sorted_lt_u8(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::unsigned::avx2::is_sorted_lt_u8(self.as_slice()) }
            } else if is_x86_feature_detected!("sse2") {
                unsafe { ::unsigned::sse41::is_sorted_lt_u8(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}

/// Specialization for iterator over &[u8] and increasing order.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Decreasing> for slice::Iter<'a, u8> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Decreasing) -> bool {
        #[cfg(not(feature = "use_std"))]
        unsafe {
            #[cfg(not(target_feature = "avx2"))]
            {
                unsafe { ::unsigned::sse41::is_sorted_gt_u8(self.as_slice()) }
            }
            #[cfg(target_feature = "avx2")]
            {
                unsafe { ::unsigned::avx2::is_sorted_gt_u8(self.as_slice()) }
            }
        }

        #[cfg(feature = "use_std")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { ::unsigned::avx2::is_sorted_gt_u8(self.as_slice()) }
            } else if is_x86_feature_detected!("sse2") {
                unsafe { ::unsigned::sse41::is_sorted_gt_u8(self.as_slice()) }
            } else {
                is_sorted_by_scalar_impl(self, compare)
            }
        }
    }
}
