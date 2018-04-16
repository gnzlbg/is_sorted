//! Extends `Iterator` with three algorithms, `is_sorted`, `is_sorted_by`, and
//! `is_sorted_by_key` that check whether the elements of an `Iterator` are
//! sorted in `O(N)` time and `O(1)` space.
#![no_std]

#![feature(specialization, fn_traits, unboxed_closures, stdsimd, align_offset)]

use core::{mem, slice, cmp::Ordering};

mod ord {
    use ::core::cmp::Ordering;

    /// Equivalent to `Ord::cmp(a, b)`
    pub struct Less();

    impl<'a, 'b, T: Ord> FnOnce<(&'a T, &'b T)> for Less {
        type Output = Ordering;
        extern "rust-call" fn call_once(self, arg: (&'a T, &'b T)) -> Self::Output {
            arg.0.cmp(arg.1)
        }
    }

    impl<'a, 'b, T: Ord> FnMut<(&'a T, &'b T)> for Less {
        extern "rust-call" fn call_mut(&mut self, arg: (&'a T, &'b T)) -> Self::Output {
            arg.0.cmp(arg.1)
        }
    }

    /// Equivalent to `Ord::cmp(a, b).invert()`
    pub struct Greater();

    impl<'a, 'b, T: Ord> FnOnce<(&'a T, &'b T)> for Greater {
        type Output = Ordering;
        extern "rust-call" fn call_once(self, arg: (&'a T, &'b T)) -> Self::Output {
            arg.0.cmp(arg.1).reverse()
        }
    }

    impl<'a, 'b, T: Ord> FnMut<(&'a T, &'b T)> for Greater {
        extern "rust-call" fn call_mut(&mut self, arg: (&'a T, &'b T)) -> Self::Output {
            arg.0.cmp(arg.1).reverse()
        }
    }
}

/// Function object equivalent to `Ord::cmp(a, b)`.
#[allow(non_upper_case_globals)]
pub const Less: ord::Less = ord::Less();

/// Function object equivalent to `Ord::cmp(a, b).reverse()`.
#[allow(non_upper_case_globals)]
pub const Greater: ord::Greater = ord::Greater();

/// Extends `Iterator` with `is_sorted`, `is_sorted_by`, and `is_sorted_by_key`.
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
        where Self: Sized,
              Self::Item: Ord,
    {
        self.is_sorted_by(<Self::Item as Ord>::cmp)
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
        where Self: Sized,
              F: FnMut(&Self::Item, &Self::Item) -> Ordering
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
        where Self: Sized,
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
fn is_sorted_by_impl<I,F>(iter: &mut I, compare: F) -> bool
    where I: Iterator,
          F: FnMut(&I::Item, &I::Item) -> Ordering,
{
    <I as IsSortedBy<F>>::is_sorted_by(iter, compare)
}


// This trait is used to provide specialized implementations of `is_sorted_by`
// for different (Iterator,Cmp) pairs:
trait IsSortedBy<F>: Iterator
    where F: FnMut(&Self::Item, &Self::Item) -> Ordering
{
    fn is_sorted_by(&mut self, compare: F) -> bool;
}

// This blanket implementation acts as the fall-back, and just forwards to the
// scalar implementation of the algorithm.
impl<I, F> IsSortedBy<F> for I
    where I: Iterator,
          F: FnMut(&I::Item, &I::Item) -> Ordering {
    #[inline]
    default fn is_sorted_by(&mut self, compare: F) -> bool {
        is_sorted_by_scalar_impl(self, compare)
    }
}

/// Scalar `is_sorted_by` implementation.
#[inline]
fn is_sorted_by_scalar_impl<I,F>(iter: &mut I, mut compare: F) -> bool
    where I: Iterator,
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

/// Specialization for iterator over &[u32] and increasing order.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl<'a> IsSortedBy<ord::Less> for slice::Iter<'a, u32> {
    #[inline]
    fn is_sorted_by(&mut self, _compare: ord::Less) -> bool {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use core::arch::x86::*;

            let s = self.as_slice();
            let n = s.len() as isize;

            // If the slice has zero or one elements, it is sorted:
            if n < 2 {
                return true;
            }

            let ap = |i| {
                (s.as_ptr().offset(i)) as *const __m128i
            };

            let mut i: isize = 0;

            // The first element of the slice might not be aligned to a 16-byte boundary.
            // Handle the elements until the first 16-byte boundary using the scalar algorithm
            {
                let mut a = s.as_ptr().align_offset(16) / mem::size_of::<u32>();
                while a > 0 && i < n {
                    if s.get_unchecked(i as usize) > s.get_unchecked(i as usize + 1) {
                        return false;
                    }
                    i += 1;
                    a -= 1;
                }
                debug_assert!(ap(i).align_offset(16) == 0);
            }

            // `i` points to the first element of the slice at a 16-byte boundary.
            // Use the SSE2 algorithm from HeroicKatora
            // https://www.reddit.com/r/cpp/comments/8bkaj3/is_sorted_using_simd_instructions/dx7jj8u/
            // to handle the body of the slice.
            if (n - i) >= 20 {
		            let mut curr = _mm_load_si128(ap(i));
                while i < n - 16 {
                    let next0 = _mm_load_si128(ap(i + 4));
                    let next1 = _mm_load_si128(ap(i + 8));
                    let next2 = _mm_load_si128(ap(i + 12));
                    let next3 = _mm_load_si128(ap(i + 16));

			              let compare0 = _mm_alignr_epi8(next0, curr, 4);
			              let compare1 = _mm_alignr_epi8(next1, next0, 4);
			              let compare2 = _mm_alignr_epi8(next2, next1, 4);
			              let compare3 = _mm_alignr_epi8(next3, next2, 4);

                    let mask0 = _mm_cmpgt_epi32(curr, compare0);
                    let mask1 = _mm_cmpgt_epi32(next0, compare1);
                    let mask2 = _mm_cmpgt_epi32(next1, compare2);
                    let mask3 = _mm_cmpgt_epi32(next2, compare3);

			              let mergedmask = _mm_or_si128(
				                _mm_or_si128(mask0, mask1),
				                _mm_or_si128(mask2, mask3)
                    );
                    if _mm_test_all_zeros(mergedmask, mergedmask) == 0 {
                        return false;
                    }

			              curr = next3;

                    i += 16;
                }
            }

            // Handle the tail of the slice using the scalar algoirithm:
            while i < n - 1 {
                if s.get_unchecked(i as usize) > s.get_unchecked(i as usize + 1) {
                    return false;
                }
                i += 1;
            }

            debug_assert!(i == n - 1);
            true
        }
    }
}


#[cfg(test)]
mod tests {
    use ::{IsSorted, Less};
    extern crate std;
    use self::std::vec::Vec;

    #[test]
    fn floats() {
        let x = [1., 2., 3., 4.];
        assert!(x.iter().is_sorted_by(|a,b| a.partial_cmp(b).unwrap()));
    }

    #[test]
    fn u32_small() {
        let x: [u32; 0] = [];
        assert!(x.iter().is_sorted_by(Less));

        let x = [0_u32];
        assert!(x.iter().is_sorted_by(Less));

        let x = [1_u32, 2, 3, 4];
        assert!(x.iter().is_sorted_by(Less));

        let x = [4_u32, 3, 2, 1];
        assert!(!x.iter().is_sorted_by(Less));
    }

    #[test]
    fn u32_large() {
        let mut v = Vec::new();
        for i in 0..1000 {
            v.push(i as u32);
        }
        {
            let s: &[u32] = v.as_slice();
            assert!(s.iter().is_sorted_by(Less));
        }

        v.push(0);
        {
            let s: &[u32] = v.as_slice();
            assert!(!s.iter().is_sorted_by(Less));
        }
    }
}
