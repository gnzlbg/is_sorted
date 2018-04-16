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

macro_rules! sorted {
    ($name:ident, $ty:ident, $size:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let n = $size;
            let mut v: Vec<$ty> = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(<$ty as Rnd>::rnd());
            }
            v.sort();
            black_box(&mut v);
            b.iter(|| {
                let s: &[$ty] = v.as_slice();
                black_box(s.as_ptr());
                black_box(s.iter().is_sorted());
            });
        }
    };
}

macro_rules! sorted_f {
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
                    use is_sorted::PartialLessUnwrapped;
                    black_box(s.iter().is_sorted_by(PartialLessUnwrapped));
                }
                #[cfg(not(feature = "unstable"))]
                {
                    black_box(
                        s.iter()
                            .is_sorted_by(|a, b| a.partial_cmp(b).unwrap()),
                    );
                }
            });
        }
    };
}

const N8: usize = 4_000_000;
const N16: usize = 2_000_000;
const N32: usize = 2_000_000;

sorted!(run_8i_baseline, wi8, N8);
sorted!(run_8u_baseline, wu8, N8);
sorted!(run_8i_is_sorted, i8, N8);
sorted!(run_8u_is_sorted, u8, N8);

sorted!(run_16i_baseline, wi16, N16);
sorted!(run_16u_baseline, wu16, N16);
sorted!(run_16i_is_sorted, i16, N16);
sorted!(run_16u_is_sorted, u16, N16);

sorted!(run_32i_baseline, wi32, N32);
sorted!(run_32u_baseline, wu32, N32);
sorted_f!(run_32f_baseline, wf32, N32);
sorted!(run_32i_is_sorted, i32, N32);
sorted!(run_32u_is_sorted, u32, N32);
sorted_f!(run_32f_is_sorted, f32, N32);
