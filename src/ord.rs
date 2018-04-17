//! Implementation of comparison callables.
#![allow(non_upper_case_globals)]

pub mod types {
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
pub const Less: types::Less = types::Less();

/// Callable equivalent to `Ord::cmp(a, b).reverse()`.
pub const Greater: types::Greater = types::Greater();

/// Callable equivalent to `PartialOrd::partial_cmp(a, b).unwrap()`.
pub const PartialLessUnwrapped: types::PartialLessUnwrapped =
    types::PartialLessUnwrapped();

/// Callable equivalent to `PartialOrd::partial_cmp(a, b).unwrap().reverse()`.
pub const PartialGreaterUnwrapped: types::PartialGreaterUnwrapped =
    types::PartialGreaterUnwrapped();
