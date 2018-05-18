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
    feature(
        specialization, fn_traits, unboxed_closures, stdsimd, align_offset
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

    /// Returns the first unsorted pair of items in the iterator and its tail.
    ///
    /// ```
    /// # use is_sorted::IsSorted;
    /// let v: &[i32] = &[0, 1, 2, 3, 4, 1, 2, 3];
    /// let (first, tail) = v.iter().is_sorted_until_by(|a,b| {
    ///     a.partial_cmp(b)
    /// });
    /// assert_eq!(first, Some((&4,&1)));
    /// assert_eq!(tail.as_slice(), &[2, 3]);
    /// ```
    #[inline]
    fn is_sorted_until_by<F>(
        self, compare: F,
    ) -> (Option<(Self::Item, Self::Item)>, Self)
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Option<Ordering>,
    {
        is_sorted_until_by_impl(self, compare)
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

/// Scalar `is_sorted_by` implementation for slices.
///
/// Avoids bound checks.
#[cfg(feature = "unstable")]
#[inline]
fn is_sorted_by_scalar_slice_impl<'a, T, F>(
    iter: &mut slice::Iter<'a, T>, mut compare: F,
) -> bool
where
    T: PartialOrd,
    F: FnMut(&&'a T, &&'a T) -> Option<Ordering>,
{
    let s = iter.as_slice();
    if s.len() < 2 {
        return true;
    }
    let mut first = unsafe { s.get_unchecked(0) };
    for i in 1..s.len() {
        let second = unsafe { s.get_unchecked(i) };
        if let Some(ord) = compare(&first, &second) {
            if ord != Ordering::Greater {
                first = second;
                continue;
            }
        }
        return false;
    }
    true
}

// This function dispatch to the appropriate `is_sorted_until_by`
// implementation.
#[inline]
fn is_sorted_until_by_impl<I, F>(
    iter: I, compare: F,
) -> (Option<(I::Item, I::Item)>, I)
where
    I: Iterator,
    F: FnMut(&I::Item, &I::Item) -> Option<Ordering>,
{
    <I as IsSortedUntilBy<F>>::is_sorted_until_by(iter, compare)
}

// This trait is used to provide specialized implementations of
// `is_sorted_until_by` for different (Iterator,Cmp) pairs:
trait IsSortedUntilBy<F>: Iterator
where
    F: FnMut(&Self::Item, &Self::Item) -> Option<Ordering>,
{
    fn is_sorted_until_by(
        self, compare: F,
    ) -> (Option<(Self::Item, Self::Item)>, Self);
}

// This blanket implementation acts as the fall-back, and just forwards to the
// scalar implementation of the algorithm.
impl<I, F> IsSortedUntilBy<F> for I
where
    I: Iterator,
    F: FnMut(&I::Item, &I::Item) -> Option<Ordering>,
{
    #[inline]
    #[cfg(feature = "unstable")]
    default fn is_sorted_until_by(
        self, compare: F,
    ) -> (Option<(Self::Item, Self::Item)>, Self) {
        is_sorted_until_by_scalar_impl(self, compare)
    }

    #[inline]
    #[cfg(not(feature = "unstable"))]
    fn is_sorted_until_by(
        self, compare: F,
    ) -> (Option<(Self::Item, Self::Item)>, Self) {
        is_sorted_until_by_scalar_impl(self, compare)
    }
}

/// Scalar `is_sorted_until_by` implementation.
#[inline]
fn is_sorted_until_by_scalar_impl<I, F>(
    mut iter: I, mut compare: F,
) -> (Option<(I::Item, I::Item)>, I)
where
    I: Iterator,
    F: FnMut(&I::Item, &I::Item) -> Option<Ordering>,
{
    let first = iter.next();
    if let Some(mut first) = first {
        loop {
            let next = iter.next();
            if let Some(next) = next {
                if let Some(ord) = compare(&first, &next) {
                    if ord != Ordering::Greater {
                        first = next;
                        continue;
                    }
                }
                return (Some((first, next)), iter);
            }
            return (None, iter);
        }
    }
    (None, iter)
}

/// Scalar `is_sorted_until_by` implementation for slices.
///
/// Avoids bounds check.
#[cfg(feature = "unstable")]
#[inline]
fn is_sorted_until_by_scalar_slice_impl<'a, T, F>(
    iter: slice::Iter<'a, T>, mut compare: F,
) -> (Option<(&'a T, &'a T)>, slice::Iter<'a, T>)
where
    T: PartialOrd,
    F: FnMut(&&'a T, &&'a T) -> Option<Ordering>,
{
    let s = iter.as_slice();
    if s.len() < 2 {
        return (None, unsafe { s.get_unchecked(s.len()..).iter() });
    }
    let mut first = unsafe { s.get_unchecked(0) };
    for i in 0..s.len() {
        let second = unsafe { s.get_unchecked(i) };
        if let Some(ord) = compare(&first, &second) {
            if ord != Ordering::Greater {
                first = second;
                continue;
            }
        }
        return (Some((first, second)), unsafe {
            s.get_unchecked((i + 1)..).iter()
        });
    }
    return (None, unsafe { s.get_unchecked(s.len()..).iter() });
}

pub fn is_sorted_until_by<T, F>(slice: &[T], f: F) -> usize
where
    for<'r, 's> F: FnMut(&'r &T, &'s &T) -> Option<Ordering>,
{
    let (boundary, tail) = IsSorted::is_sorted_until_by(slice.iter(), f);
    match boundary {
        Some(_) => {
            debug_assert!(tail.as_slice().len() < slice.len());
            slice.len() - tail.as_slice().len() - 1
        }
        None => {
            debug_assert!(tail.as_slice().is_empty());
            slice.len()
        }
    }
}

#[cfg(feature = "unstable")]
impl<'a, T, F> IsSortedBy<F> for slice::Iter<'a, T>
where
    T: PartialOrd,
    F: FnMut(&&'a T, &&'a T) -> Option<Ordering>,
{
    #[inline]
    default fn is_sorted_by(&mut self, compare: F) -> bool {
        is_sorted_by_scalar_slice_impl(self, compare)
    }
}

#[cfg(feature = "unstable")]
impl<'a, T, F> IsSortedUntilBy<F> for slice::Iter<'a, T>
where
    T: PartialOrd,
    F: FnMut(&&'a T, &&'a T) -> Option<Ordering>,
{
    #[inline]
    default fn is_sorted_until_by(
        self, compare: F,
    ) -> (Option<(&'a T, &'a T)>, Self) {
        return is_sorted_until_by_scalar_slice_impl(self, compare);
    }
}

/// Adds a specialization of the IsSortedBy trait for a slice iterator.
///
/// The (feature,function) pairs must be listed in order of decreasing
/// preference, that is, first pair will be preferred over second pair if its
/// feature is enabled.
macro_rules! is_sorted_by_slice_iter_x86 {
    ($id:ident, $cmp:path : $([$feature:tt, $function:path]),*) => {
        #[cfg(feature = "unstable")]
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(
            any(
                // Either we have run-time feature detection:
                feature = "use_std",
                // Or the features are enabled at compile-time
                any($(target_feature = $feature),*)
            )
        )]
        impl<'a> IsSortedBy<$cmp> for slice::Iter<'a, $id> {
            #[inline]
            fn is_sorted_by(&mut self, compare: $cmp) -> bool {
                // If we don't have run-time feature detection, we use
                // compile-time detection. This specialization only exists if
                // at least one of the features is actually enabled, so we don't
                // need a fallback here.
                #[cfg(not(feature = "use_std"))]
                unsafe {
                    $(
                        #[cfg(target_feature = $feature)]
                        {
                            return $function(self.as_slice()) == self.as_slice().len();
                        }
                    )*
                }

                #[cfg(feature = "use_std")]
                {
                    $(
                        if is_x86_feature_detected!($feature) {
                            return unsafe { $function(self.as_slice())  == self.as_slice().len() };
                        }
                    )*;
                    // If feature detection fails use scalar code:
                    return is_sorted_by_scalar_slice_impl(self, compare);
                }
            }
        }

        #[cfg(feature = "unstable")]
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(
            any(
                // Either we have run-time feature detection:
                feature = "use_std",
                // Or the features are enabled at compile-time
                any($(target_feature = $feature),*)
            )
        )]
        impl<'a> IsSortedUntilBy<$cmp> for slice::Iter<'a, $id> {
            #[inline]
            fn is_sorted_until_by(self, compare: $cmp)
                                  -> (Option<(Self::Item,Self::Item)>, Self) {
                // If we don't have run-time feature detection, we use
                // compile-time detection. This specialization only exists if
                // at least one of the features is actually enabled, so we don't
                // need a fallback here.
                #[cfg(not(feature = "use_std"))]
                unsafe {
                    $(
                        #[cfg(target_feature = $feature)]
                        {
                            // The slice values until the index i returned by
                            // the impl $function are sorted. The slice &[0..j]
                            // for j > i might be sorted as well, so we just
                            // proceed to call the scalar implementation on it:
                            let s = self.as_slice();
                            let i = $function(s);
                            return is_sorted_until_by_scalar_slice_impl(s[i..].iter(), compare);
                        }
                    )*
                }

                #[cfg(feature = "use_std")]
                {
                    $(
                        if is_x86_feature_detected!($feature) {
                            let s = self.as_slice();
                            let i =  unsafe { $function(s) };
                            return is_sorted_until_by_scalar_slice_impl(s[i..].iter(), compare);
                        }
                    )*;
                    // If feature detection fails use scalar code:
                    return is_sorted_until_by_scalar_slice_impl(self, compare);
                }
            }
        }
    }
}

is_sorted_by_slice_iter_x86!(
    i64, ord::types::Increasing :
    ["avx2", ::signed::avx2::is_sorted_lt_i64],
    ["sse4.2", ::signed::sse42::is_sorted_lt_i64]
);

is_sorted_by_slice_iter_x86!(
    i64, ord::types::Decreasing :
    ["avx2", ::signed::avx2::is_sorted_gt_i64],
    ["sse4.2", ::signed::sse42::is_sorted_gt_i64]
);

is_sorted_by_slice_iter_x86!(
    f64, ord::types::Increasing :
    ["avx", ::floats::avx::is_sorted_lt_f64],
    ["sse4.1", ::floats::sse41::is_sorted_lt_f64]
);

is_sorted_by_slice_iter_x86!(
    f64, ord::types::Decreasing :
    ["avx", ::floats::avx::is_sorted_gt_f64],
    ["sse4.1", ::floats::sse41::is_sorted_gt_f64]
);

is_sorted_by_slice_iter_x86!(
    i32, ord::types::Increasing :
    ["avx2", ::signed::avx2::is_sorted_lt_i32],
    ["sse4.1", ::signed::sse41::is_sorted_lt_i32]
);

is_sorted_by_slice_iter_x86!(
    i32, ord::types::Decreasing :
    ["avx2", ::signed::avx2::is_sorted_gt_i32],
    ["sse4.1", ::signed::sse41::is_sorted_gt_i32]
);

is_sorted_by_slice_iter_x86!(
    u32, ord::types::Increasing :
    ["avx2", ::unsigned::avx2::is_sorted_lt_u32],
    ["sse4.1", ::unsigned::sse41::is_sorted_lt_u32]
);

is_sorted_by_slice_iter_x86!(
    u32, ord::types::Decreasing :
    ["avx2", ::unsigned::avx2::is_sorted_gt_u32],
    ["sse4.1", ::unsigned::sse41::is_sorted_gt_u32]
);

is_sorted_by_slice_iter_x86!(
    f32, ord::types::Increasing :
    ["avx", ::floats::avx::is_sorted_lt_f32],
    ["sse4.1", ::floats::sse41::is_sorted_lt_f32]
);

is_sorted_by_slice_iter_x86!(
    f32, ord::types::Decreasing :
    ["avx", ::floats::avx::is_sorted_gt_f32],
    ["sse4.1", ::floats::sse41::is_sorted_gt_f32]
);

is_sorted_by_slice_iter_x86!(
    i16, ord::types::Increasing :
    ["avx2", ::signed::avx2::is_sorted_lt_i16],
    ["sse4.1", ::signed::sse41::is_sorted_lt_i16]
);

is_sorted_by_slice_iter_x86!(
    i16, ord::types::Decreasing :
    ["avx2", ::signed::avx2::is_sorted_gt_i16],
    ["sse4.1", ::signed::sse41::is_sorted_gt_i16]
);

is_sorted_by_slice_iter_x86!(
    u16, ord::types::Increasing :
    ["avx2", ::unsigned::avx2::is_sorted_lt_u16],
    ["sse4.1", ::unsigned::sse41::is_sorted_lt_u16]
);

is_sorted_by_slice_iter_x86!(
    u16, ord::types::Decreasing :
    ["avx2", ::unsigned::avx2::is_sorted_gt_u16],
    ["sse4.1", ::unsigned::sse41::is_sorted_gt_u16]
);

is_sorted_by_slice_iter_x86!(
    i8, ord::types::Increasing :
    ["avx2", ::signed::avx2::is_sorted_lt_i8],
    ["sse4.1", ::signed::sse41::is_sorted_lt_i8]
);

is_sorted_by_slice_iter_x86!(
    i8, ord::types::Decreasing :
    ["avx2", ::signed::avx2::is_sorted_gt_i8],
    ["sse4.1", ::signed::sse41::is_sorted_gt_i8]
);

is_sorted_by_slice_iter_x86!(
    u8, ord::types::Increasing :
    ["avx2", ::unsigned::avx2::is_sorted_lt_u8],
    ["sse4.1", ::unsigned::sse41::is_sorted_lt_u8]
);

is_sorted_by_slice_iter_x86!(
    u8, ord::types::Decreasing :
    ["avx2", ::unsigned::avx2::is_sorted_gt_u8],
    ["sse4.1", ::unsigned::sse41::is_sorted_gt_u8]
);
