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

/// Specialization for iterator over &[i64] and increasing order.
///
/// If `std` is available, always include this and perform run-time feature
/// detection inside it to select the `SSE4.1` algorithm when the CPU supports
/// it.
///
/// If `std` is not available, include this specialization only when the
/// target supports `SSE4.1` at compile-time.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.2", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Less> for slice::Iter<'a, i64> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Less) -> bool {
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
///
/// If `std` is available, always include this and perform run-time feature
/// detection inside it to select the `SSE4.1` algorithm when the CPU supports
/// it.
///
/// If `std` is not available, include this specialization only when the
/// target supports `SSE4.1` at compile-time.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.2", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Greater> for slice::Iter<'a, i64> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Greater) -> bool {
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
impl<'a> IsSortedBy<ord::types::PartialLessUnwrapped>
    for slice::Iter<'a, f64>
{
    #[inline]
    fn is_sorted_by(
        &mut self, compare: ord::types::PartialLessUnwrapped,
    ) -> bool {
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
impl<'a> IsSortedBy<ord::types::PartialGreaterUnwrapped>
    for slice::Iter<'a, f64>
{
    #[inline]
    fn is_sorted_by(
        &mut self, compare: ord::types::PartialGreaterUnwrapped,
    ) -> bool {
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
///
/// If `std` is available, always include this and perform run-time feature
/// detection inside it to select the `SSE4.1` algorithm when the CPU supports
/// it.
///
/// If `std` is not available, include this specialization only when the
/// target supports `SSE4.1` at compile-time.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Less> for slice::Iter<'a, i32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Less) -> bool {
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
///
/// If `std` is available, always include this and perform run-time feature
/// detection inside it to select the `SSE4.1` algorithm when the CPU supports
/// it.
///
/// If `std` is not available, include this specialization only when the
/// target supports `SSE4.1` at compile-time.
#[cfg(feature = "unstable")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(
    any(
        feature = "use_std",
        any(target_feature = "sse4.1", target_feature = "avx2")
    )
)]
impl<'a> IsSortedBy<ord::types::Greater> for slice::Iter<'a, i32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Greater) -> bool {
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
impl<'a> IsSortedBy<ord::types::Less> for slice::Iter<'a, u32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Less) -> bool {
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
impl<'a> IsSortedBy<ord::types::Greater> for slice::Iter<'a, u32> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Greater) -> bool {
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
impl<'a> IsSortedBy<ord::types::PartialLessUnwrapped>
    for slice::Iter<'a, f32>
{
    #[inline]
    fn is_sorted_by(
        &mut self, compare: ord::types::PartialLessUnwrapped,
    ) -> bool {
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
impl<'a> IsSortedBy<ord::types::PartialGreaterUnwrapped>
    for slice::Iter<'a, f32>
{
    #[inline]
    fn is_sorted_by(
        &mut self, compare: ord::types::PartialGreaterUnwrapped,
    ) -> bool {
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
impl<'a> IsSortedBy<ord::types::Less> for slice::Iter<'a, i16> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Less) -> bool {
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
impl<'a> IsSortedBy<ord::types::Greater> for slice::Iter<'a, i16> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Greater) -> bool {
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
impl<'a> IsSortedBy<ord::types::Less> for slice::Iter<'a, u16> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Less) -> bool {
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
impl<'a> IsSortedBy<ord::types::Greater> for slice::Iter<'a, u16> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Greater) -> bool {
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
impl<'a> IsSortedBy<ord::types::Less> for slice::Iter<'a, i8> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Less) -> bool {
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
impl<'a> IsSortedBy<ord::types::Greater> for slice::Iter<'a, i8> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Greater) -> bool {
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
impl<'a> IsSortedBy<ord::types::Less> for slice::Iter<'a, u8> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Less) -> bool {
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
impl<'a> IsSortedBy<ord::types::Greater> for slice::Iter<'a, u8> {
    #[inline]
    fn is_sorted_by(&mut self, compare: ord::types::Greater) -> bool {
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

#[cfg(test)]
mod tests {
    use IsSorted;
    extern crate rand;

    #[cfg(not(feature = "use_std"))]
    #[macro_use]
    extern crate std;

    #[cfg(feature = "use_std")]
    use super::std;

    use self::rand::{thread_rng, Rng};

    use self::std::vec::Vec;

    macro_rules! test_float {
        ($name:ident, $id:ident, $cmp_t:ident) => {
            #[test]
            #[allow(unused_mut, unused_macros, dead_code)]
            fn $name() {
                #[cfg(feature = "unstable")]
                macro_rules! cmp_lt {
                    () => {
                        ::PartialLessUnwrapped
                    };
                }

                #[cfg(not(feature = "unstable"))]
                macro_rules! cmp_lt {
                    () => {
                        |a, b| a.partial_cmp(b).unwrap()
                    };
                }

                #[cfg(feature = "unstable")]
                macro_rules! cmp_gt {
                    () => {
                        ::PartialGreaterUnwrapped
                    };
                }

                #[cfg(not(feature = "unstable"))]
                macro_rules! cmp_gt {
                    () => {
                        |a, b| a.partial_cmp(b).unwrap().reverse()
                    };
                }

                macro_rules! rev {
                    (cmp_gt,$i: ident) => {
                        $i.reverse();
                    };
                    ($_o: ident,$i: ident) => {};
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

                let mut x = [1., 2., 3., 4.];
                rev!($cmp_t, x);
                assert!(x.iter().is_sorted_by($cmp_t!()));

                let mut x: [$id; 0] = [];
                rev!($cmp_t, x);
                assert!(x.iter().is_sorted_by($cmp_t!()));

                let mut x = [0 as $id];
                rev!($cmp_t, x);
                assert!(x.iter().is_sorted_by($cmp_t!()));

                let max = ::std::$id::INFINITY;
                let min = -max;

                let mut x = [min, max];
                rev!($cmp_t, x);
                assert!(x.iter().is_sorted_by($cmp_t!()));

                let mut x = [1 as $id, 2., 3., 4.];
                rev!($cmp_t, x);
                assert!(x.iter().is_sorted_by($cmp_t!()));

                let mut x = [1 as $id, 3., 2., 4.];
                rev!($cmp_t, x);
                assert!(!x.iter().is_sorted_by($cmp_t!()));

                let mut x = [4 as $id, 3., 2., 1.];
                rev!($cmp_t, x);
                assert!(!x.iter().is_sorted_by($cmp_t!()));

                let mut x = [4 as $id, 4., 4., 4.];
                rev!($cmp_t, x);
                assert!(x.iter().is_sorted_by($cmp_t!()));

                let mut v = Vec::new();
                for _ in 0..2 {
                    v.push(min);
                }
                for _ in 0..2 {
                    v.push(max);
                }
                rev!($cmp_t, v);
                assert!(v.as_slice().iter().is_sorted_by($cmp_t!()));

                let mut v = Vec::new();
                for _ in 0..4 {
                    v.push(min);
                }
                for _ in 0..5 {
                    v.push(max);
                }
                rev!($cmp_t, v);
                assert!(v.as_slice().iter().is_sorted_by($cmp_t!()));

                macro_rules! min_max {
                    (cmp_lt,$min_: ident,$max_: ident) => {{
                        ($min_, $max_)
                    }};
                    (cmp_gt,$min_: ident,$max_: ident) => {{
                        ($max_, $min_)
                    }};
                }

                let (min, max) = min_max!($cmp_t, min, max);

                for i in 0..1_000 {
                    let mut vec: Vec<$id> = random_vec(i);
                    vec.sort_by($cmp_t!());
                    assert!(
                        vec.as_slice().iter().is_sorted_by($cmp_t!()),
                        "is_sorted0: {:?}",
                        vec
                    );
                    if i > 4 {
                        vec.push(min);
                        assert!(
                            !vec.as_slice().iter().is_sorted_by($cmp_t!()),
                            "!is_sorted1: {:?}",
                            vec
                        );
                        vec.insert(i / 3 * 2, min);
                        assert!(
                            !vec.as_slice().iter().is_sorted_by($cmp_t!()),
                            "!is_sorted2: {:?}",
                            vec
                        );
                        vec.insert(0, max);
                        assert!(
                            !vec.as_slice().iter().is_sorted_by($cmp_t!()),
                            "!is_sorted3: {:?}",
                            vec
                        );
                    }
                }
            }
        };
    }

    test_float!(test_lt_f32, f32, cmp_lt);
    test_float!(test_lt_f64, f64, cmp_lt);
    test_float!(test_gt_f32, f32, cmp_gt);
    test_float!(test_gt_f64, f64, cmp_gt);

    macro_rules! ints {
        ($name:ident, $id:ident, $cmp_t:ident) => {
            #[test]
            #[allow(unused_mut, unused_macros, dead_code)]
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

                #[cfg(feature = "unstable")]
                macro_rules! cmp_lt {
                    () => {
                        ::Less
                    };
                }

                #[cfg(not(feature = "unstable"))]
                macro_rules! cmp_lt {
                    () => {
                        |a, b| a.cmp(b)
                    };
                }

                #[cfg(feature = "unstable")]
                macro_rules! cmp_gt {
                    () => {
                        ::Greater
                    };
                }

                #[cfg(not(feature = "unstable"))]
                macro_rules! cmp_gt {
                    () => {
                        |a, b| a.partial_cmp(b).unwrap().reverse()
                    };
                }

                macro_rules! rev {
                    (cmp_gt,$i: ident) => {
                        $i.reverse();
                    };
                    ($_o: ident,$i: ident) => {};
                }

                let mut x: [$id; 0] = [];
                rev!($cmp_t, x);
                assert!(x.iter().is_sorted_by($cmp_t!()));

                let mut x = [0 as $id];
                rev!($cmp_t, x);
                assert!(x.iter().is_sorted_by($cmp_t!()));

                let mut x = [$id::min_value(), $id::max_value()];
                rev!($cmp_t, x);
                assert!(x.iter().is_sorted_by($cmp_t!()));

                let mut x = [1 as $id, 2, 3, 4];
                rev!($cmp_t, x);
                assert!(x.iter().is_sorted_by($cmp_t!()));

                let mut x = [1 as $id, 3, 2, 4];
                rev!($cmp_t, x);
                assert!(!x.iter().is_sorted_by($cmp_t!()));

                let mut x = [4 as $id, 3, 2, 1];
                rev!($cmp_t, x);
                assert!(!x.iter().is_sorted_by($cmp_t!()));

                let mut x = [4 as $id, 4, 4, 4];
                rev!($cmp_t, x);
                assert!(x.iter().is_sorted_by($cmp_t!()));

                let min = $id::min_value();
                let max = $id::max_value();

                let mut v = Vec::new();
                for _ in 0..2 {
                    v.push(min);
                }
                for _ in 0..2 {
                    v.push(max);
                }
                rev!($cmp_t, v);
                assert!(v.as_slice().iter().is_sorted_by($cmp_t!()));

                let mut v = Vec::new();
                for _ in 0..4 {
                    v.push(min);
                }
                for _ in 0..5 {
                    v.push(max);
                }
                rev!($cmp_t, v);
                assert!(v.as_slice().iter().is_sorted_by($cmp_t!()));

                macro_rules! min_max {
                    (cmp_lt,$min_: ident,$max_: ident) => {{
                        ($min_, $max_)
                    }};
                    (cmp_gt,$min_: ident,$max_: ident) => {{
                        ($max_, $min_)
                    }};
                }

                let (min, max) = min_max!($cmp_t, min, max);

                for i in 0..1_000 {
                    let mut vec: Vec<$id> = random_vec(i);
                    vec.sort();
                    rev!($cmp_t, vec);
                    assert!(
                        vec.as_slice().iter().is_sorted_by($cmp_t!()),
                        "is_sorted0: {:?}",
                        vec
                    );
                    if i > 4 {
                        vec.push(min);
                        assert!(
                            !vec.as_slice().iter().is_sorted_by($cmp_t!()),
                            "!is_sorted1: {:?}",
                            vec
                        );
                        vec.insert(i / 3 * 2, min);
                        assert!(
                            !vec.as_slice().iter().is_sorted_by($cmp_t!()),
                            "!is_sorted2: {:?}",
                            vec
                        );
                        vec.insert(0, max);
                        assert!(
                            !vec.as_slice().iter().is_sorted_by($cmp_t!()),
                            "!is_sorted3: {:?}",
                            vec
                        );
                    }
                }

                {
                    let n = 1_000;
                    let mut v = Vec::new();
                    for i in 0..n {
                        v.push(i as $id);
                    }
                    v.sort();
                    rev!($cmp_t, v);
                    {
                        let s: &[$id] = v.as_slice();
                        assert!(s.iter().is_sorted_by($cmp_t!()));
                    }

                    v.push(min);
                    {
                        let s: &[$id] = v.as_slice();
                        assert!(!s.iter().is_sorted_by($cmp_t!()));
                    }
                    for i in &mut v {
                        *i = 42;
                    }
                    {
                        let s: &[$id] = v.as_slice();
                        assert!(s.iter().is_sorted_by($cmp_t!()));
                    }
                    for i in &mut v {
                        *i = max;
                    }
                    {
                        let s: &[$id] = v.as_slice();
                        assert!(s.iter().is_sorted_by($cmp_t!()));
                    }
                    v.push(min);
                    {
                        let s: &[$id] = v.as_slice();
                        assert!(!s.iter().is_sorted_by($cmp_t!()));
                    }
                    for i in &mut v {
                        *i = min;
                    }
                    {
                        let s: &[$id] = v.as_slice();
                        assert!(s.iter().is_sorted_by($cmp_t!()));
                    }
                    let mut v = Vec::new();
                    for _ in 0..n {
                        v.push(min);
                    }
                    for _ in 0..n {
                        v.push(max);
                    }
                    assert!(v.as_slice().iter().is_sorted_by($cmp_t!()));
                }
            }
        };
    }
    ints!(ints_lt_i8, i8, cmp_lt);
    ints!(ints_lt_u8, u8, cmp_lt);
    ints!(ints_lt_i16, i16, cmp_lt);
    ints!(ints_lt_u16, u16, cmp_lt);
    ints!(ints_lt_u32, u32, cmp_lt);
    ints!(ints_lt_i32, i32, cmp_lt);
    ints!(ints_lt_u64, u64, cmp_lt);
    ints!(ints_lt_i64, i64, cmp_lt);

    ints!(ints_gt_i8, i8, cmp_gt);
    ints!(ints_gt_u8, u8, cmp_gt);
    ints!(ints_gt_i16, i16, cmp_gt);
    ints!(ints_gt_u16, u16, cmp_gt);
    ints!(ints_gt_u32, u32, cmp_gt);
    ints!(ints_gt_i32, i32, cmp_gt);
    ints!(ints_gt_u64, u64, cmp_gt);
    ints!(ints_gt_i64, i64, cmp_gt);

    #[test]
    fn x86_failures() {
        #[cfg_attr(rustfmt, skip)]
        let v: Vec<i16> = vec![
            -32587, -31811, -31810, -31622, -30761, -29579, -28607, -27995,
            -27980, -27403, -26662, -26316, -25664, -25650, -25585, -23815,
            -22096, -21967, -21411, -20551, -20407, -20313, -19771, -19229,
            -18646, -17645, -16922, -16563, -16206, -15835, -14874, -14356,
            -13805, -13365, -12367, -12120, -11968, -11306, -10933, -10483,
            -9675, -9461, -9085, -8820, -8335, -7610, -6900, -6816, -5990,
            -5968, -5437, -4304, -3563, -3066, -2585, -1965, -1743, -1635,
            -1547, -1509, -1080, -452, 150, 1735, 1958, 3050, 3185, 3308,
            3668, 3937, 3991, 5067, 5140, 5167, 5309, 5464, 7062, 7063, 8366,
            9067, 9330, 9966, 10253, 10407, 12210, 12309, 12322, 12744, 12789,
            12847, 13542, 14028, 14548, 14818, 15699, 16127, 16297, 16493,
            16618, 16629, 17196, 17726, 18188, 18321, 19237, 19691, 20367,
            20633, 20843, 20919, 22205, 22219, 24090, 25047, 25976, 27148,
            27280, 27976, 28195, 28496, 29367, 29714, 29741, 30975, 31389,
            31621, 31641, 31730, 31732,
        ];
        assert!(v.iter().is_sorted());

        #[cfg_attr(rustfmt, skip)]
        let v: Vec<i32> = vec![
            -2127396028,
            -2082815528,
            -2038895088,
            -2019079871,
            -1978373996,
            -1835721329,
            -1831387531,
            -1779937646,
            -1739829077,
            -1587879517,
            -1512361690,
            -1360053313,
            -1320500302,
            -1312546330,
            -1233666039,
            -1227337358,
            -1199207574,
            -1174355055,
            -1085592280,
            -997390415,
            -889799053,
            -835634996,
            -830313699,
            -686077565,
            -653162121,
            -600377558,
            -555885531,
            -420404737,
            -413324460,
            -300193793,
            -297974875,
            -290727125,
            -273972354,
            -188203173,
            -164412618,
            -100667379,
            -52404093,
            29881861,
            90172874,
            225566667,
            238100506,
            240707584,
            250544067,
            327778371,
            371256113,
            687979273,
            704065256,
            804811282,
            811146835,
            837098934,
            920358630,
            979089785,
            1125388001,
            1204033686,
            1321135512,
            1352639888,
            1556346641,
            1632068112,
            1655184247,
            1679920790,
            1806456281,
            1848685160,
            1896103285,
            1919676348,
            1953567150,
        ];
        assert!(v.iter().is_sorted());

        #[cfg_attr(rustfmt, skip)]
        let v: Vec<i8> = vec![
            -128, -126, -126, -125, -125, -123, -122, -120, -119, -117, -115,
            -114, -113, -112, -110, -110, -105, -105, -102, -102, -101, -101,
            -98, -97, -95, -92, -91, -90, -89, -89, -88, -88, -87, -87, -87,
            -86, -85, -85, -82, -82, -80, -78, -76, -74, -68, -67, -64, -64,
            -62, -61, -57, -57, -56, -56, -53, -51, -48, -45, -44, -43, -39,
            -38, -36, -35, -34, -33, -33, -29, -29, -27, -26, -24, -20, -20,
            -17, -16, -15, -14, -12, -12, -11, -10, -10, -8, -7, -1, 0, 0, 0,
            1, 1, 1, 3, 5, 6, 9, 11, 11, 12, 13, 14, 15, 20, 22, 23, 23, 25,
            26, 28, 30, 31, 32, 32, 35, 37, 40, 41, 42, 43, 44, 48, 48, 49,
            51, 51, 56, 62, 64, 70, 72, 76, 77, 78, 80, 81, 81, 82, 84, 87,
            88, 90, 91, 93, 93, 95, 97, 98, 99, 102, 102, 104, 106, 106, 109,
            112, 113, 115, 115, 117, 124, 125,
        ];
        assert!(v.iter().is_sorted());
    }
}
