//! Implementation of comparison callables.
#![allow(non_upper_case_globals)]

pub mod types {
    use cmp::Ordering;

    /// Equivalent to `Ord::partial_cmp(a, b)`
    pub struct Increasing();

    impl<'a, 'b, T: PartialOrd> FnOnce<(&'a T, &'b T)> for Increasing {
        type Output = Option<Ordering>;
        extern "rust-call" fn call_once(
            self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.partial_cmp(arg.1)
        }
    }

    impl<'a, 'b, T: PartialOrd> FnMut<(&'a T, &'b T)> for Increasing {
        extern "rust-call" fn call_mut(
            &mut self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.partial_cmp(arg.1)
        }
    }

    /// Equivalent to `Ord::partial_cmp(a, b).reverse()`
    pub struct Decreasing();

    impl<'a, 'b, T: PartialOrd> FnOnce<(&'a T, &'b T)> for Decreasing {
        type Output = Option<Ordering>;
        extern "rust-call" fn call_once(
            self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.partial_cmp(arg.1).map(|v| v.reverse())
        }
    }

    impl<'a, 'b, T: PartialOrd> FnMut<(&'a T, &'b T)> for Decreasing {
        extern "rust-call" fn call_mut(
            &mut self, arg: (&'a T, &'b T),
        ) -> Self::Output {
            arg.0.partial_cmp(arg.1).map(|v| v.reverse())
        }
    }
}

/// Increasing ordering: callable equivalent to `a.partial_cmp(b)`.
pub const Increasing: types::Increasing = types::Increasing();

/// Decreasing ordering: callable equivalent to `a.partial_cmp(b).reverse()`.
pub const Decreasing: types::Decreasing = types::Decreasing();
