#![feature(test)]
#![allow(non_camel_case_types)]

extern crate is_sorted;
use is_sorted::IsSorted;

extern crate test;
use test::black_box;
use test::Bencher;

extern crate rand;
use self::rand::{thread_rng, Rng};

trait Rnd {
    fn rnd() -> Self;
}

macro_rules! from_usize_prim {
    ($id:ident) => {
        impl Rnd for $id {
            fn rnd() -> Self {
                let mut rng = thread_rng();
                rng.gen_range(Self::min_value(), Self::max_value())
            }
        }
    };
}
macro_rules! from_usize_prim_f {
    ($id:ident) => {
        impl Rnd for $id {
            fn rnd() -> Self {
                let mut rng = thread_rng();
                rng.gen_range(0. as $id, 1. as $id)
            }
        }
    };
}

from_usize_prim!(i8);
from_usize_prim!(u8);
from_usize_prim!(i16);
from_usize_prim!(u16);
from_usize_prim!(i32);
from_usize_prim!(u32);
from_usize_prim_f!(f32);
from_usize_prim!(i64);
from_usize_prim!(u64);
from_usize_prim_f!(f64);

macro_rules! wrapper {
    ($id:ident, $inner:ident) => {
        #[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
        struct $id($inner);
        impl Rnd for $id {
            fn rnd() -> Self {
                $id(<$inner as Rnd>::rnd())
            }
        }
    };
}
macro_rules! wrapper_f {
    ($id:ident, $inner:ident) => {
        #[derive(Copy, Clone, PartialEq, PartialOrd)]
        struct $id($inner);
        impl Rnd for $id {
            fn rnd() -> Self {
                $id(<$inner as Rnd>::rnd())
            }
        }
    };
}

wrapper!(wu8, u8);
wrapper!(wi8, i8);
wrapper!(wu16, u16);
wrapper!(wi16, i16);
wrapper!(wu32, u32);
wrapper!(wi32, i32);
wrapper_f!(wf32, f32);
wrapper!(wu64, u64);
wrapper!(wi64, i64);
wrapper_f!(wf64, f64);

macro_rules! sorted_lt {
    ($name:ident, $ty:ident, $size:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let n = $size;
            let mut v: Vec<$ty> = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(<$ty as Rnd>::rnd());
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            black_box(&mut v);
            b.iter(|| {
                let s: &[$ty] = v.as_slice();
                black_box(s.as_ptr());
                #[cfg(feature = "unstable")]
                {
                    use is_sorted::Increasing;
                    assert!(
                        black_box(s.iter().is_sorted_until_by(Increasing))
                            .0
                            .is_none()
                    );
                }
                #[cfg(not(feature = "unstable"))]
                {
                    assert!(
                        black_box(
                            s.iter()
                                .is_sorted_until_by(|a, b| a.partial_cmp(b))
                        ).0
                            .is_none()
                    );
                }
            });
        }
    };
}

macro_rules! sorted_gt {
    ($name:ident, $ty:ident, $size:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let n = $size;
            let mut v: Vec<$ty> = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(<$ty as Rnd>::rnd());
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.reverse();
            black_box(&mut v);
            b.iter(|| {
                let s: &[$ty] = v.as_slice();
                black_box(s.as_ptr());
                #[cfg(feature = "unstable")]
                {
                    use is_sorted::Decreasing;
                    assert!(
                        black_box(s.iter().is_sorted_until_by(Decreasing))
                            .0
                            .is_none()
                    );
                }
                #[cfg(not(feature = "unstable"))]
                {
                    assert!(
                        black_box(s.iter().is_sorted_until_by(|a, b| {
                            a.partial_cmp(b).map(|v| v.reverse())
                        })).0
                            .is_none()
                    );
                }
            });
        }
    };
}

const N8: usize = 4_000_000;
const N16: usize = 2_000_000;
const N32: usize = 2_000_000;
const N64: usize = 2_000_000;

sorted_lt!(lt_8i_scalar, wi8, N8);
sorted_lt!(lt_8u_scalar, wu8, N8);
sorted_lt!(lt_8i_lib, i8, N8);
sorted_lt!(lt_8u_lib, u8, N8);

sorted_lt!(lt_16i_scalar, wi16, N16);
sorted_lt!(lt_16u_scalar, wu16, N16);
sorted_lt!(lt_16i_lib, i16, N16);
sorted_lt!(lt_16u_lib, u16, N16);

sorted_lt!(lt_32i_scalar, wi32, N32);
sorted_lt!(lt_32u_scalar, wu32, N32);
sorted_lt!(lt_32f_scalar, wf32, N32);
sorted_lt!(lt_32i_lib, i32, N32);
sorted_lt!(lt_32u_lib, u32, N32);
sorted_lt!(lt_32f_lib, f32, N32);

sorted_lt!(lt_64i_scalar, wi64, N64);
sorted_lt!(lt_64u_scalar, wu64, N64);
sorted_lt!(lt_64f_scalar, wf64, N64);
sorted_lt!(lt_64i_lib, i64, N64);
sorted_lt!(lt_64u_lib, u64, N64);
sorted_lt!(lt_64f_lib, f64, N64);

sorted_gt!(gt_8i_scalar, wi8, N8);
sorted_gt!(gt_8u_scalar, wu8, N8);
sorted_gt!(gt_8i_lib, i8, N8);
sorted_gt!(gt_8u_lib, u8, N8);

sorted_gt!(gt_16i_scalar, wi16, N16);
sorted_gt!(gt_16u_scalar, wu16, N16);
sorted_gt!(gt_16i_lib, i16, N16);
sorted_gt!(gt_16u_lib, u16, N16);

sorted_gt!(gt_32i_scalar, wi32, N32);
sorted_gt!(gt_32u_scalar, wu32, N32);
sorted_gt!(gt_32f_scalar, wf32, N32);
sorted_gt!(gt_32i_lib, i32, N32);
sorted_gt!(gt_32u_lib, u32, N32);
sorted_gt!(gt_32f_lib, f32, N32);

sorted_gt!(gt_64i_scalar, wi64, N64);
sorted_gt!(gt_64u_scalar, wu64, N64);
sorted_gt!(gt_64f_scalar, wf64, N64);
sorted_gt!(gt_64i_lib, i64, N64);
sorted_gt!(gt_64u_lib, u64, N64);
sorted_gt!(gt_64f_lib, f64, N64);
