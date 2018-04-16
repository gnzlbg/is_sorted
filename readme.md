# `is_sorted`: is an `Iterator` sorted

This crate extends the `Iterator` trait with the `is_sorted`, `is_sorted_by`,
and `is_sorted_by_key` methods that check whether the iterator elements are
sorted according to some order in `O(N)` time and `O(1)` space. 

The algorithm to do this is obviously pretty trivial, but this allows this crate
to showcases a couple of intermediate-level techniques that are useful when
writing Rust components. This crates shows how to:

* extend `Iterator` with your own algorithms
* use specialization to provide more efficient implementations for
  certain type-comparison pairs
* use `stdsimd`, `target_feature`, and `cfg_target_feature` to
  explicitly-vectorize some of the specializations using both compile-time (for
  `#![no_std]` users) and run-time (for `std` users) feature detection
* implement callables that can be specialized on using `fn_traits` and
  `unboxed_closures`
* support stable users even though the crate uses a lot of nightly-only features

The crate also adds the following callables that enable specialization based on
the comparison operation being used:

* `is_sorted::Less`: equivalent to `a.cmp(b)`
* `is_sorted::Greater`: equivalent to `a.cmp(b).reverse()`

When compiled with `--features unstable` the crate makes use of the following
nightly-only features:

* `fn_traits`, `unboxed_closures`: to implement the comparison callables
* `specialization`: to specialize the algorithms for pairs of types and callables
* `stdsimd`: `std::arch` is used by the specialization to explicitly vectorize the algorithms
* `target_feature` and `cfg_target_feature`: use to enable the appropriate
  target-features of the specialized algorithms
* `align_offset`: is used to handle misaligned inputs in the vectorized algorithms

Explicit vectorization delivers a 2x throughput increase over the
auto-vectorized versions produced by `rustc`, just run:

> cargo bench
> cargo bench --features unstable

and see for yourself.

## License

This project is licensed under either of

* Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in SliceDeque by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
