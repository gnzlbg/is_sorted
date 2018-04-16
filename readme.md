# `is_sorted`: is an `Iterator` sorted

[![crates.io version][crate-shield]][crate] [![Travis build status][travis-shield]][travis] [![Appveyor build status][appveyor-shield]][appveyor] [![Docs][docs-shield]][docs] [![License][license-shield]][license]


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
* `is_sorted::PartialLessUnwrapped`: equivalent to `a.partial_cmp(b).unwrap()`
* `is_sorted::PartialGreaterUnwrapped`: equivalent to `a.partial_cmp(b).unwrap().reverse()`

When compiled with `--features unstable` the crate makes use of the following
nightly-only features:

* `fn_traits`, `unboxed_closures`: to implement the comparison callables
* `specialization`: to specialize the algorithms for pairs of types and callables
* `stdsimd`: `std::arch` is used by the specialization to explicitly vectorize the algorithms
* `target_feature` and `cfg_target_feature`: use to enable the appropriate
  target-features of the specialized algorithms
* `align_offset`: is used to handle misaligned inputs in the vectorized algorithms

Explicit vectorization delivers between 1.6x and 6.5x speed-ups over the code
produced by `rustc`, just run:

>$ cargo bench --features unstable

and see for yourself. On my machine:

```shell
test run_16i_baseline  ... bench:   1,030,343 ns/iter (+/- 128,024)
test run_16i_is_sorted ... bench:     291,504 ns/iter (+/- 36,711) # 3.5x
test run_16u_baseline  ... bench:   1,030,282 ns/iter (+/- 133,204) 
test run_16u_is_sorted ... bench:     339,998 ns/iter (+/- 72,954) # 3.0x
test run_32f_baseline  ... bench:   3,186,189 ns/iter (+/- 1,879,466)
test run_32f_is_sorted ... bench:     666,493 ns/iter (+/- 164,823) # 4.8x
test run_32i_baseline  ... bench:   1,061,721 ns/iter (+/- 144,198) 
test run_32i_is_sorted ... bench:     603,093 ns/iter (+/- 104,585) # 1.7x
test run_32u_baseline  ... bench:   1,064,075 ns/iter (+/- 158,429)
test run_32u_is_sorted ... bench:     666,817 ns/iter (+/- 135,030) # 1.6x
test run_8i_baseline   ... bench:   1,837,582 ns/iter (+/- 529,630)
test run_8i_is_sorted  ... bench:     284,903 ns/iter (+/- 82,538) # 6.5x
test run_8u_baseline   ... bench:   1,972,757 ns/iter (+/- 347,044)
test run_8u_is_sorted  ... bench:     316,199 ns/iter (+/- 42,681) # 6.2x
```

## License

This project is licensed under either of

* Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in SliceDeque by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

[travis-shield]: https://img.shields.io/travis/gnzlbg/is_sorted.svg?style=flat-square
[travis]: https://travis-ci.org/gnzlbg/is_sorted
[appveyor-shield]: https://img.shields.io/appveyor/ci/gnzlbg/is-sorted.svg?style=flat-square
[appveyor]: https://ci.appveyor.com/project/gnzlbg/is-sorted/branch/master
[docs-shield]: https://img.shields.io/badge/docs-online-blue.svg?style=flat-square
[docs]: https://docs.rs/crate/is-sorted/
[license-shield]: https://img.shields.io/badge/License-MIT%2FApache2.0-green.svg?style=flat-square
[license]: https://github.com/gnzlbg/is_sorted/blob/master/license.md
[crate-shield]: https://img.shields.io/crates/v/is_sorted.svg?style=flat-square
[crate]: https://crates.io/crates/is_sorted
