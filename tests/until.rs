//! is_sorted_until_by tests

extern crate is_sorted;

use is_sorted::IsSorted;

extern crate rand;

use rand::{thread_rng, Rng};

macro_rules! integers {
    ($test_name:ident, $id:ident) => {
        #[test]
        fn $test_name() {
            use std::cmp::Ordering;
            // Random vector of size `x` sorted until element `i` according to
            // `f`
            fn random_vec<F>(x: usize, mut f: F) -> Vec<$id>
            where
                for<'r, 's> F: FnMut(&'r $id, &'s $id) -> Option<Ordering>,
            {
                let mut vec = Vec::with_capacity(x);
                let mut rng = thread_rng();
                for _ in 0..x {
                    let val: $id =
                        rng.gen_range($id::min_value(), $id::max_value());
                    vec.push(val);
                }

                vec.sort_by(|a, b| f(a, b).unwrap());
                vec
            }

            fn unsort<F>(v: &mut Vec<$id>, idx: usize, mut f: F)
            where
                for<'r, 's> F:
                    FnMut(&'r $id, &'s $id) -> Option<Ordering> + Copy,
            {
                assert!(IsSorted::is_sorted_by(&mut v.iter(), |a, b| f(a, b)));
                v.insert(idx, $id::min_value());
                while IsSorted::is_sorted_by(&mut v.iter(), |a, b| f(a, b)) {
                    let mut nv = random_vec(v.len() - 1, f);
                    nv.insert(idx, $id::min_value());
                    *v = nv;
                }
            }

            fn check_unsorted<F>(v: &[$id], idx: usize, mut f: F)
            where
                for<'r, 's> F:
                    FnMut(&'r $id, &'s $id) -> Option<Ordering> + Copy,
            {
                let len = v.len();
                let (p, s) = v.iter().is_sorted_until_by(|a, b| f(a, b));
                assert!(p.is_some());
                let (_, &m) = p.unwrap();
                let min = $id::min_value();
                assert_eq!(m, min);
                assert_eq!(s.len(), len - 1 - idx);

                use is_sorted::is_sorted_until_by;
                assert_eq!(
                    is_sorted_until_by(v, |a, b| f(a, b)),
                    idx,
                    "{:?}",
                    v
                );
                check_sorted(&v[0..idx], |a, b| f(a, b));
            }

            fn check_sorted<F>(v: &[$id], mut f: F)
            where
                for<'r, 's> F: FnMut(&'r $id, &'s $id) -> Option<Ordering>,
            {
                let (p, s) = v.iter().is_sorted_until_by(|a, b| f(a, b));
                assert!(p.is_none());
                assert!(
                    s.as_slice().is_empty(),
                    "v: {:?}, s: {:?}",
                    v,
                    s.as_slice()
                );

                use is_sorted::is_sorted_until_by;
                assert_eq!(
                    is_sorted_until_by(v, |a, b| f(a, b)),
                    v.len(),
                    "{:?}",
                    v
                );
            }

            let cmp = <$id as PartialOrd>::partial_cmp;
            for len in 4..127 {
                for idx in 3..len + 1 {
                    let mut v = random_vec(len, cmp);
                    unsort(&mut v, idx, |a, b| a.partial_cmp(b));
                    check_unsorted(&v, idx, |a, b| a.partial_cmp(b));
                }
            }

            let v = vec![];
            check_sorted(&v, |a, b| a.partial_cmp(b));
            let v = vec![1 as $id];
            check_sorted(&v, |a, b| a.partial_cmp(b));
        }
    };
}

integers!(test_i8, i8);
integers!(test_i16, i16);
integers!(test_i32, i32);
integers!(test_i64, i64);

integers!(test_u8, u8);
integers!(test_u16, u16);
integers!(test_u32, u32);
integers!(test_u64, u64);
