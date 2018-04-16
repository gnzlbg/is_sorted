# `is_sorted`: is an iterator sorted

This crate extends the `Iterator` trait with the `is_sorted`, `is_sorted_by`,
and `is_sorted_by_key` methods that check whether the iterator elements are
sorted according to some order. 

It also adds the following callables that enable specialization based on the
comparison operation being used:

* `is_sorted::Less`: equivalent to `a.cmp(b)`
* `is_sorted::Greater`: equivalent to `a.cmp(b).reverse()`
* `is_sorted::PartialLess`: equivalent to `a.partial_cmp(b)`
* `is_sorted::PartialGreater`: equivalent to `a.partial_cmp(b).reverse()`

The crate requires nightly and uses the following nightly features:

* `fn_traits`, `unboxed_closures`: to implement the comparison callables
* `specialization`: to specialize the algorithms for pairs of types and callables
* `stdsimd`: `std::arch` is used by the specialization to explicitly vectorize the algorithms
* `align_offset`: is used to handle misaligned inputs in the vectorized algorithms

## License

This project is licensed under either of

* Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in SliceDeque by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
