#![feature(test)]

extern crate is_sorted;
use is_sorted::{IsSorted};

extern crate test;
use test::black_box;
use test::Bencher;

trait FromUsize {
    fn from_usize(x: usize) -> Self;
}
impl FromUsize for i32 { fn from_usize(x: usize) -> Self { x as Self } }
impl FromUsize for u32 { fn from_usize(x: usize) -> Self { x as Self } }

macro_rules! sorted {
    ($name:ident, $ty:ident, $size:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let n = $size;
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push($ty::from_usize(i));
            }
            black_box(&mut v);
            b.iter(|| {
                let s: &[$ty] = v.as_slice();
                black_box(s.as_ptr());
                black_box(s.iter().is_sorted());
            });
        }
    }
}

sorted!(large_i32, i32, 1_000_000);
sorted!(large_u32, u32, 1_000_000);
