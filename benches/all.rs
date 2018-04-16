#![feature(test)]

extern crate is_sorted;
use is_sorted::{IsSorted, Less};

extern crate test;
use test::black_box;
use test::Bencher;


#[bench]
fn large_i32(b: &mut Bencher) {
    let n = 1_000_000;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(i as i32);
    }
    black_box(&mut v);
    b.iter(|| {
        let s: &[i32] = v.as_slice();
        black_box(s.as_ptr());
        black_box(s.iter().is_sorted_by(Less));
    });
}

#[bench]
fn large_u32(b: &mut Bencher) {
    let n = 1_000_000;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(i as u32);
    }
    black_box(&mut v);
    b.iter(|| {
        let s: &[u32] = v.as_slice();
        black_box(s.as_ptr());
        black_box(s.iter().is_sorted_by(Less));
    });
}
