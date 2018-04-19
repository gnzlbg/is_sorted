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
                assert!(black_box(s.iter().is_sorted()));
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
                    assert!(black_box(s.iter().is_sorted_by(Decreasing)));
                }
                #[cfg(not(feature = "unstable"))]
                {
                    assert!(black_box(s.iter().is_sorted_by(|a, b| {
                        a.partial_cmp(b).map(|v| v.reverse())
                    }),));
                }
            });
        }
    };
}

const N8: usize = 4_000_000;
const N16: usize = 2_000_000;
const N32: usize = 2_000_000;
const N64: usize = 2_000_000;

sorted_lt!(run_lt_8i_baseline, wi8, N8);
sorted_lt!(run_lt_8u_baseline, wu8, N8);
sorted_lt!(run_lt_8i_is_sorted, i8, N8);
sorted_lt!(run_lt_8u_is_sorted, u8, N8);

sorted_lt!(run_lt_16i_baseline, wi16, N16);
sorted_lt!(run_lt_16u_baseline, wu16, N16);
sorted_lt!(run_lt_16i_is_sorted, i16, N16);
sorted_lt!(run_lt_16u_is_sorted, u16, N16);

sorted_lt!(run_lt_32i_baseline, wi32, N32);
sorted_lt!(run_lt_32u_baseline, wu32, N32);
sorted_lt!(run_lt_32f_baseline, wf32, N32);
sorted_lt!(run_lt_32i_is_sorted, i32, N32);
sorted_lt!(run_lt_32u_is_sorted, u32, N32);
sorted_lt!(run_lt_32f_is_sorted, f32, N32);

sorted_lt!(run_lt_64i_baseline, wi64, N64);
sorted_lt!(run_lt_64u_baseline, wu64, N64);
sorted_lt!(run_lt_64f_baseline, wf64, N64);
sorted_lt!(run_lt_64i_is_sorted, i64, N64);
sorted_lt!(run_lt_64u_is_sorted, u64, N64);
sorted_lt!(run_lt_64f_is_sorted, f64, N64);

sorted_gt!(run_gt_8i_baseline, wi8, N8);
sorted_gt!(run_gt_8u_baseline, wu8, N8);
sorted_gt!(run_gt_8i_is_sorted, i8, N8);
sorted_gt!(run_gt_8u_is_sorted, u8, N8);

sorted_gt!(run_gt_16i_baseline, wi16, N16);
sorted_gt!(run_gt_16u_baseline, wu16, N16);
sorted_gt!(run_gt_16i_is_sorted, i16, N16);
sorted_gt!(run_gt_16u_is_sorted, u16, N16);

sorted_gt!(run_gt_32i_baseline, wi32, N32);
sorted_gt!(run_gt_32u_baseline, wu32, N32);
sorted_gt!(run_gt_32f_baseline, wf32, N32);
sorted_gt!(run_gt_32i_is_sorted, i32, N32);
sorted_gt!(run_gt_32u_is_sorted, u32, N32);
sorted_gt!(run_gt_32f_is_sorted, f32, N32);

sorted_gt!(run_gt_64i_baseline, wi64, N64);
sorted_gt!(run_gt_64u_baseline, wu64, N64);
sorted_gt!(run_gt_64f_baseline, wf64, N64);
sorted_gt!(run_gt_64i_is_sorted, i64, N64);
sorted_gt!(run_gt_64u_is_sorted, u64, N64);
sorted_gt!(run_gt_64f_is_sorted, f64, N64);
