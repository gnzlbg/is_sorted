# `is_sorted`: is an `Iterator` sorted

[![crates.io version][crate-shield]][crate] [![Travis build status][travis-shield]][travis] [![Appveyor build status][appveyor-shield]][appveyor] [![Docs][docs-shield]][docs] [![License][license-shield]][license]

> **Note:** This is now exported by Rust's `core` library, e.g., see [`Iterator`](https://doc.rust-lang.org/nightly/core/iter/trait.Iterator.html?search=#method.is_sorted). The implementations in
this crate remain up to 10x faster than libcore because this crate can make use of features that
libcore cannot use yet (we'll fix that eventually). 

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

* `is_sorted::Increasing`: equivalent to `a.partial_cmp(b)`
* `is_sorted::Decreasing`: equivalent to `a.partial_cmp(b).map(|v| v.reverse())`

When compiled with `--features unstable` the crate makes use of the following
nightly-only features:

* `fn_traits`, `unboxed_closures`: to implement the comparison callables
* `specialization`: to specialize the algorithms for pairs of types and callables
* `stdsimd`: `std::arch` is used by the specialization to explicitly vectorize the algorithms
* `align_offset`: is used to handle misaligned inputs in the vectorized algorithms

Explicit vectorization delivers between 1.5x and 10x speed-ups over the code
produced by `rustc` for some `Iterator`s. To see it for yourself, just run:

>$ cargo bench --features unstable

On my laptop (2012 Intel Core i5, AVX, no AVX2) for slices:

```shell
test run_gt_16i_baseline  ... bench:   1,056,679 ns/iter (+/- 143,130)
test run_gt_16i_is_sorted ... bench:     297,737 ns/iter (+/- 73,536)
test run_gt_16u_baseline  ... bench:   1,048,473 ns/iter (+/- 151,361)
test run_gt_16u_is_sorted ... bench:     330,464 ns/iter (+/- 57,180)
test run_gt_32f_baseline  ... bench:   5,859,745 ns/iter (+/- 643,377)
test run_gt_32f_is_sorted ... bench:     665,421 ns/iter (+/- 116,844)
test run_gt_32i_baseline  ... bench:   1,089,292 ns/iter (+/- 124,127)
test run_gt_32i_is_sorted ... bench:     627,826 ns/iter (+/- 85,923)
test run_gt_32u_baseline  ... bench:   1,108,998 ns/iter (+/- 163,938)
test run_gt_32u_is_sorted ... bench:     702,923 ns/iter (+/- 112,332)
test run_gt_64f_baseline  ... bench:   3,824,177 ns/iter (+/- 492,995)
test run_gt_64f_is_sorted ... bench:   1,364,920 ns/iter (+/- 197,156)
test run_gt_64i_baseline  ... bench:   1,310,525 ns/iter (+/- 245,387)
test run_gt_64i_is_sorted ... bench:   1,301,780 ns/iter (+/- 367,669)
test run_gt_64u_baseline  ... bench:   1,313,168 ns/iter (+/- 169,762)
test run_gt_64u_is_sorted ... bench:   1,300,316 ns/iter (+/- 209,528)
test run_gt_8i_baseline   ... bench:   2,010,967 ns/iter (+/- 175,342)
test run_gt_8i_is_sorted  ... bench:     303,082 ns/iter (+/- 68,407)
test run_gt_8u_baseline   ... bench:   2,029,840 ns/iter (+/- 300,009)
test run_gt_8u_is_sorted  ... bench:     337,869 ns/iter (+/- 114,609)
test run_lt_16i_baseline  ... bench:   1,037,078 ns/iter (+/- 104,067)
test run_lt_16i_is_sorted ... bench:     301,939 ns/iter (+/- 89,317)
test run_lt_16u_baseline  ... bench:   1,031,273 ns/iter (+/- 115,523)
test run_lt_16u_is_sorted ... bench:     334,392 ns/iter (+/- 107,146)
test run_lt_32f_baseline  ... bench:   3,201,105 ns/iter (+/- 257,790)
test run_lt_32f_is_sorted ... bench:     681,012 ns/iter (+/- 264,631)
test run_lt_32i_baseline  ... bench:   1,140,243 ns/iter (+/- 587,907)
test run_lt_32i_is_sorted ... bench:     789,255 ns/iter (+/- 890,137)
test run_lt_32u_baseline  ... bench:   1,192,369 ns/iter (+/- 615,291)
test run_lt_32u_is_sorted ... bench:     747,146 ns/iter (+/- 233,472)
test run_lt_64f_baseline  ... bench:   3,635,677 ns/iter (+/- 3,049,427)
test run_lt_64f_is_sorted ... bench:   1,565,629 ns/iter (+/- 521,948)
test run_lt_64i_baseline  ... bench:   1,321,831 ns/iter (+/- 265,014)
test run_lt_64i_is_sorted ... bench:   1,478,014 ns/iter (+/- 412,410)
test run_lt_64u_baseline  ... bench:   1,428,323 ns/iter (+/- 413,818)
test run_lt_64u_is_sorted ... bench:   1,313,273 ns/iter (+/- 223,336)
test run_lt_8i_baseline   ... bench:   2,000,606 ns/iter (+/- 253,341)
test run_lt_8i_is_sorted  ... bench:     299,591 ns/iter (+/- 45,478)
test run_lt_8u_baseline   ... bench:   1,989,555 ns/iter (+/- 247,432)
test run_lt_8u_is_sorted  ... bench:     327,004 ns/iter (+/- 74,697)
```

On an Xeon(R) CPU E5-2695 v3 @ 2.30GHz:

```shell
test run_gt_16i_baseline  ... bench:     530,724 ns/iter (+/- 9,966)
test run_gt_16i_is_sorted ... bench:     164,685 ns/iter (+/- 2,791)
test run_gt_16u_baseline  ... bench:     530,724 ns/iter (+/- 11,917)
test run_gt_16u_is_sorted ... bench:     166,817 ns/iter (+/- 2,649)
test run_gt_32f_baseline  ... bench:   3,870,192 ns/iter (+/- 58,284)
test run_gt_32f_is_sorted ... bench:     270,014 ns/iter (+/- 3,011)
test run_gt_32i_baseline  ... bench:     530,887 ns/iter (+/- 2,091)
test run_gt_32i_is_sorted ... bench:     329,381 ns/iter (+/- 5,795)
test run_gt_32u_baseline  ... bench:     530,927 ns/iter (+/- 19,677)
test run_gt_32u_is_sorted ... bench:     333,764 ns/iter (+/- 6,847)
test run_gt_64f_baseline  ... bench:   2,180,730 ns/iter (+/- 52,635)
test run_gt_64f_is_sorted ... bench:     538,037 ns/iter (+/- 8,761)
test run_gt_64i_baseline  ... bench:     691,102 ns/iter (+/- 14,588)
test run_gt_64i_is_sorted ... bench:     628,104 ns/iter (+/- 15,375)
test run_gt_64u_baseline  ... bench:     690,482 ns/iter (+/- 16,620)
test run_gt_64u_is_sorted ... bench:     690,421 ns/iter (+/- 15,802)
test run_gt_8i_baseline   ... bench:     909,540 ns/iter (+/- 16,632)
test run_gt_8i_is_sorted  ... bench:     136,164 ns/iter (+/- 1,880)
test run_gt_8u_baseline   ... bench:     909,536 ns/iter (+/- 14,981)
test run_gt_8u_is_sorted  ... bench:     140,761 ns/iter (+/- 2,443)
test run_lt_16i_baseline  ... bench:     530,707 ns/iter (+/- 2,364)
test run_lt_16i_is_sorted ... bench:     160,733 ns/iter (+/- 1,780)
test run_lt_16u_baseline  ... bench:     530,713 ns/iter (+/- 9,249)
test run_lt_16u_is_sorted ... bench:     168,386 ns/iter (+/- 2,356)
test run_lt_32f_baseline  ... bench:   1,529,191 ns/iter (+/- 11,369)
test run_lt_32f_is_sorted ... bench:     269,269 ns/iter (+/- 3,844)
test run_lt_32i_baseline  ... bench:     530,920 ns/iter (+/- 18,349)
test run_lt_32i_is_sorted ... bench:     327,124 ns/iter (+/- 5,909)
test run_lt_32u_baseline  ... bench:     530,893 ns/iter (+/- 749)
test run_lt_32u_is_sorted ... bench:     338,442 ns/iter (+/- 3,649)
test run_lt_64f_baseline  ... bench:   1,537,085 ns/iter (+/- 22,255)
test run_lt_64f_is_sorted ... bench:     537,958 ns/iter (+/- 15,249)
test run_lt_64i_baseline  ... bench:     690,423 ns/iter (+/- 21,300)
test run_lt_64i_is_sorted ... bench:     635,785 ns/iter (+/- 17,912)
test run_lt_64u_baseline  ... bench:     690,270 ns/iter (+/- 15,633)
test run_lt_64u_is_sorted ... bench:     690,380 ns/iter (+/- 13,587)
test run_lt_8i_baseline   ... bench:     909,537 ns/iter (+/- 13,486)
test run_lt_8i_is_sorted  ... bench:     131,389 ns/iter (+/- 1,350)
test run_lt_8u_baseline   ... bench:     909,535 ns/iter (+/- 20,730)
test run_lt_8u_is_sorted  ... bench:     140,781 ns/iter (+/- 1,980)
```

## License

This project is licensed under either of

* Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `is_sorted` by you, as defined in the Apache-2.0 license, shall
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
