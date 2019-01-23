#![feature(trace_macros)]

extern crate is_sorted;

use is_sorted::IsSorted;

extern crate rand;

use rand::{thread_rng, Rng};

macro_rules! test_float {
    ($name:ident, $id:ident, $cmp_t:ident) => {
        #[test]
        #[allow(unused_mut, unused_macros, dead_code)]
        fn $name() {
            #[cfg(feature = "unstable")]
            macro_rules! cmp_lt {
                () => {
                    ::is_sorted::Increasing
                };
            }

            #[cfg(not(feature = "unstable"))]
            macro_rules! cmp_lt {
                () => {
                    |a, b| a.partial_cmp(b)
                };
            }

            #[cfg(feature = "unstable")]
            macro_rules! cmp_gt {
                () => {
                    ::is_sorted::Decreasing
                };
            }

            #[cfg(not(feature = "unstable"))]
            macro_rules! cmp_gt {
                () => {
                    |a, b| a.partial_cmp(b).map(|v| v.reverse())
                };
            }

            macro_rules! rev {
                (cmp_gt,$i: ident) => {
                    $i.reverse();
                };
                ($_o: ident,$i: ident) => {};
            }

            fn random_vec(x: usize) -> Vec<$id> {
                let mut vec = Vec::with_capacity(x);
                let mut rng = thread_rng();
                for _ in 0..x {
                    let val: $id = rng.gen_range(0. as $id, 1. as $id);
                    vec.push(val);
                }
                vec
            }

            let mut x = [1., 2., 3., 4.];
            rev!($cmp_t, x);
            assert!(IsSorted::is_sorted_by(&mut x.iter(), $cmp_t!()));

            let mut x: [$id; 0] = [];
            rev!($cmp_t, x);
            assert!(IsSorted::is_sorted_by(&mut x.iter(), $cmp_t!()));

            let mut x = [0 as $id];
            rev!($cmp_t, x);
            assert!(IsSorted::is_sorted_by(&mut x.iter(), $cmp_t!()));

            let max = ::std::$id::INFINITY;
            let min = -max;

            let mut x = [min, max];
            rev!($cmp_t, x);
            assert!(IsSorted::is_sorted_by(&mut x.iter(), $cmp_t!()));

            let mut x = [1 as $id, 2., 3., 4.];
            rev!($cmp_t, x);
            assert!(IsSorted::is_sorted_by(&mut x.iter(), $cmp_t!()));

            let mut x = [1 as $id, 3., 2., 4.];
            rev!($cmp_t, x);
            assert!(!IsSorted::is_sorted_by(&mut x.iter(), $cmp_t!()));

            let mut x = [4 as $id, 3., 2., 1.];
            rev!($cmp_t, x);
            assert!(!IsSorted::is_sorted_by(&mut x.iter(), $cmp_t!()));

            let mut x = [4 as $id, 4., 4., 4.];
            rev!($cmp_t, x);
            assert!(IsSorted::is_sorted_by(&mut x.iter(), $cmp_t!()));

            let mut v = Vec::new();
            for _ in 0..2 {
                v.push(min);
            }
            for _ in 0..2 {
                v.push(max);
            }
            rev!($cmp_t, v);
            assert!(IsSorted::is_sorted_by(
                &mut v.as_slice().iter(),
                $cmp_t!()
            ));

            let mut v = Vec::new();
            for _ in 0..4 {
                v.push(min);
            }
            for _ in 0..5 {
                v.push(max);
            }
            rev!($cmp_t, v);
            assert!(IsSorted::is_sorted_by(
                &mut v.as_slice().iter(),
                $cmp_t!()
            ));

            macro_rules! min_max {
                (cmp_lt,$min_: ident,$max_: ident) => {{
                    ($min_, $max_)
                }};
                (cmp_gt,$min_: ident,$max_: ident) => {{
                    ($max_, $min_)
                }};
            }
            /*
                    let (min, max) = min_max!($cmp_t, min, max);

                    for i in 0..1_000 {
                        let mut vec: Vec<$id> = random_vec(i);
                        vec.sort_by($cmp_t!());
                        assert!(
                            vec.as_slice().iter().is_sorted_by($cmp_t!()),
                            "is_sorted0: {:?}",
                            vec
                        );
                        if i > 4 {
                            vec.push(min);
                            assert!(
                                !vec.as_slice().iter().is_sorted_by($cmp_t!()),
                                "!is_sorted1: {:?}",
                                vec
                            );
                            vec.insert(i / 3 * 2, min);
                            assert!(
                                !vec.as_slice().iter().is_sorted_by($cmp_t!()),
                                "!is_sorted2: {:?}",
                                vec
                            );
                            vec.insert(0, max);
                            assert!(
                                !vec.as_slice().iter().is_sorted_by($cmp_t!()),
                                "!is_sorted3: {:?}",
                                vec
                            );
                        }
                    }
                    */
        }
    };
}

test_float!(test_lt_f32, f32, cmp_lt);
test_float!(test_lt_f64, f64, cmp_lt);
test_float!(test_gt_f32, f32, cmp_gt);
test_float!(test_gt_f64, f64, cmp_gt);

#[test]
fn exphp_tests() {
    macro_rules! check {
        ($vec:expr, $cmp_lt:expr, $cmp_gt:expr) => {
            let mut v = $vec;
            // Test the small vector
            assert!(
                !IsSorted::is_sorted_by(&mut v.iter(), $cmp_lt),
                "{:?}",
                v
            );
            v.reverse();
            assert!(
                !IsSorted::is_sorted_by(&mut v.iter(), $cmp_gt),
                "{:?}",
                v
            );
            v.reverse();
            // Test a large vector with the pattern in the middle
            let mut o = Vec::new();
            for _ in 0..100 {
                o.push(0.0);
            }
            o.append(&mut v);
            for _ in 0..100 {
                o.push(10.0);
            }
            assert!(!IsSorted::is_sorted_by(&mut o.iter(), $cmp_lt));
            o.reverse();
            assert!(!IsSorted::is_sorted_by(&mut o.iter(), $cmp_gt));
        };
    }

    macro_rules! exphp {
        ($id:ident, $cmp_lt:expr, $cmp_gt:expr) => {
            let nan = ::std::$id::NAN;
            check!(vec![nan, nan, nan], $cmp_lt, $cmp_gt);
            check!(vec![1.0, nan, 2.0], $cmp_lt, $cmp_gt);
            check!(vec![2.0, nan, 1.0], $cmp_lt, $cmp_gt);
            check!(vec![2.0, nan, 1.0, 7.0], $cmp_lt, $cmp_gt);
            check!(vec![2.0, nan, 1.0, 0.0], $cmp_lt, $cmp_gt);

            check!(vec![-nan, -1.0, 0.0, 1.0, nan], $cmp_lt, $cmp_gt);
            check!(vec![nan, -nan, -1.0, 0.0, 1.0], $cmp_lt, $cmp_gt);
            check!(vec![1.0, nan, -nan, -1.0, 0.0], $cmp_lt, $cmp_gt);
            check!(vec![0.0, 1.0, nan, -nan, -1.0], $cmp_lt, $cmp_gt);
            check!(vec![-1.0, 0.0, 1.0, nan, -nan], $cmp_lt, $cmp_gt);
        };
    }
    exphp!(f32, |a, b| a.partial_cmp(b), |a, b| a
        .partial_cmp(b)
        .map(|v| v.reverse()));
    exphp!(f64, |a, b| a.partial_cmp(b), |a, b| a
        .partial_cmp(b)
        .map(|v| v.reverse()));

    #[cfg(feature = "unstable")]
    {
        exphp!(f32, is_sorted::Increasing, is_sorted::Decreasing);
        exphp!(f64, is_sorted::Increasing, is_sorted::Decreasing);
    }
}
