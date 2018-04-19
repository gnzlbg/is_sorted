extern crate is_sorted;

use is_sorted::IsSorted;

extern crate rand;

use rand::{thread_rng, Rng};

macro_rules! ints {
    ($name:ident, $id:ident, $cmp_t:ident) => {
        #[test]
        #[allow(unused_mut, unused_macros, dead_code)]
        fn $name() {
            fn random_vec(x: usize) -> Vec<$id> {
                let mut vec = Vec::with_capacity(x);
                let mut rng = thread_rng();
                for _ in 0..x {
                    let val: $id =
                        rng.gen_range($id::min_value(), $id::max_value());
                    vec.push(val);
                }
                vec
            }

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

            let mut x: [$id; 0] = [];
            rev!($cmp_t, x);
            assert!(x.iter().is_sorted_by($cmp_t!()));

            let mut x = [0 as $id];
            rev!($cmp_t, x);
            assert!(x.iter().is_sorted_by($cmp_t!()));

            let mut x = [$id::min_value(), $id::max_value()];
            rev!($cmp_t, x);
            assert!(x.iter().is_sorted_by($cmp_t!()));

            let mut x = [1 as $id, 2, 3, 4];
            rev!($cmp_t, x);
            assert!(x.iter().is_sorted_by($cmp_t!()));

            let mut x = [1 as $id, 3, 2, 4];
            rev!($cmp_t, x);
            assert!(!x.iter().is_sorted_by($cmp_t!()));

            let mut x = [4 as $id, 3, 2, 1];
            rev!($cmp_t, x);
            assert!(!x.iter().is_sorted_by($cmp_t!()));

            let mut x = [4 as $id, 4, 4, 4];
            rev!($cmp_t, x);
            assert!(x.iter().is_sorted_by($cmp_t!()));

            let min = $id::min_value();
            let max = $id::max_value();

            let mut v = Vec::new();
            for _ in 0..2 {
                v.push(min);
            }
            for _ in 0..2 {
                v.push(max);
            }
            rev!($cmp_t, v);
            assert!(v.as_slice().iter().is_sorted_by($cmp_t!()));

            let mut v = Vec::new();
            for _ in 0..4 {
                v.push(min);
            }
            for _ in 0..5 {
                v.push(max);
            }
            rev!($cmp_t, v);
            assert!(v.as_slice().iter().is_sorted_by($cmp_t!()));

            macro_rules! min_max {
                (cmp_lt,$min_: ident,$max_: ident) => {{
                    ($min_, $max_)
                }};
                (cmp_gt,$min_: ident,$max_: ident) => {{
                    ($max_, $min_)
                }};
            }

            let (min, max) = min_max!($cmp_t, min, max);

            for i in 0..1_000 {
                let mut vec: Vec<$id> = random_vec(i);
                vec.sort();
                rev!($cmp_t, vec);
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

            {
                let n = 1_000;
                let mut v = Vec::new();
                for i in 0..n {
                    v.push(i as $id);
                }
                v.sort();
                rev!($cmp_t, v);
                {
                    let s: &[$id] = v.as_slice();
                    assert!(s.iter().is_sorted_by($cmp_t!()));
                }

                v.push(min);
                {
                    let s: &[$id] = v.as_slice();
                    assert!(!s.iter().is_sorted_by($cmp_t!()));
                }
                for i in &mut v {
                    *i = 42;
                }
                {
                    let s: &[$id] = v.as_slice();
                    assert!(s.iter().is_sorted_by($cmp_t!()));
                }
                for i in &mut v {
                    *i = max;
                }
                {
                    let s: &[$id] = v.as_slice();
                    assert!(s.iter().is_sorted_by($cmp_t!()));
                }
                v.push(min);
                {
                    let s: &[$id] = v.as_slice();
                    assert!(!s.iter().is_sorted_by($cmp_t!()));
                }
                for i in &mut v {
                    *i = min;
                }
                {
                    let s: &[$id] = v.as_slice();
                    assert!(s.iter().is_sorted_by($cmp_t!()));
                }
                let mut v = Vec::new();
                for _ in 0..n {
                    v.push(min);
                }
                for _ in 0..n {
                    v.push(max);
                }
                assert!(v.as_slice().iter().is_sorted_by($cmp_t!()));
            }
        }
    };
}
ints!(ints_lt_i8, i8, cmp_lt);
ints!(ints_lt_u8, u8, cmp_lt);
ints!(ints_lt_i16, i16, cmp_lt);
ints!(ints_lt_u16, u16, cmp_lt);
ints!(ints_lt_u32, u32, cmp_lt);
ints!(ints_lt_i32, i32, cmp_lt);
ints!(ints_lt_u64, u64, cmp_lt);
ints!(ints_lt_i64, i64, cmp_lt);

ints!(ints_gt_i8, i8, cmp_gt);
ints!(ints_gt_u8, u8, cmp_gt);
ints!(ints_gt_i16, i16, cmp_gt);
ints!(ints_gt_u16, u16, cmp_gt);
ints!(ints_gt_u32, u32, cmp_gt);
ints!(ints_gt_i32, i32, cmp_gt);
ints!(ints_gt_u64, u64, cmp_gt);
ints!(ints_gt_i64, i64, cmp_gt);

#[test]
fn x86_failures() {
    #[cfg_attr(rustfmt, skip)]
    let v: Vec<i16> = vec![
        -32587, -31811, -31810, -31622, -30761, -29579, -28607, -27995,
        -27980, -27403, -26662, -26316, -25664, -25650, -25585, -23815,
        -22096, -21967, -21411, -20551, -20407, -20313, -19771, -19229,
        -18646, -17645, -16922, -16563, -16206, -15835, -14874, -14356,
        -13805, -13365, -12367, -12120, -11968, -11306, -10933, -10483, -9675,
        -9461, -9085, -8820, -8335, -7610, -6900, -6816, -5990, -5968, -5437,
        -4304, -3563, -3066, -2585, -1965, -1743, -1635, -1547, -1509, -1080,
        -452, 150, 1735, 1958, 3050, 3185, 3308, 3668, 3937, 3991, 5067, 5140,
        5167, 5309, 5464, 7062, 7063, 8366, 9067, 9330, 9966, 10253, 10407,
        12210, 12309, 12322, 12744, 12789, 12847, 13542, 14028, 14548, 14818,
        15699, 16127, 16297, 16493, 16618, 16629, 17196, 17726, 18188, 18321,
        19237, 19691, 20367, 20633, 20843, 20919, 22205, 22219, 24090, 25047,
        25976, 27148, 27280, 27976, 28195, 28496, 29367, 29714, 29741, 30975,
        31389, 31621, 31641, 31730, 31732,
    ];
    assert!(v.iter().is_sorted());

    #[cfg_attr(rustfmt, skip)]
    let v: Vec<i32> = vec![
        -2127396028,
        -2082815528,
        -2038895088,
        -2019079871,
        -1978373996,
        -1835721329,
        -1831387531,
        -1779937646,
        -1739829077,
        -1587879517,
        -1512361690,
        -1360053313,
        -1320500302,
        -1312546330,
        -1233666039,
        -1227337358,
        -1199207574,
        -1174355055,
        -1085592280,
        -997390415,
        -889799053,
        -835634996,
        -830313699,
        -686077565,
        -653162121,
        -600377558,
        -555885531,
        -420404737,
        -413324460,
        -300193793,
        -297974875,
        -290727125,
        -273972354,
        -188203173,
        -164412618,
        -100667379,
        -52404093,
        29881861,
        90172874,
        225566667,
        238100506,
        240707584,
        250544067,
        327778371,
        371256113,
        687979273,
        704065256,
        804811282,
        811146835,
        837098934,
        920358630,
        979089785,
        1125388001,
        1204033686,
        1321135512,
        1352639888,
        1556346641,
        1632068112,
        1655184247,
        1679920790,
        1806456281,
        1848685160,
        1896103285,
        1919676348,
        1953567150,
    ];
    assert!(v.iter().is_sorted());

    #[cfg_attr(rustfmt, skip)]
    let v: Vec<i8> = vec![
        -128, -126, -126, -125, -125, -123, -122, -120, -119, -117, -115,
        -114, -113, -112, -110, -110, -105, -105, -102, -102, -101, -101, -98,
        -97, -95, -92, -91, -90, -89, -89, -88, -88, -87, -87, -87, -86, -85,
        -85, -82, -82, -80, -78, -76, -74, -68, -67, -64, -64, -62, -61, -57,
        -57, -56, -56, -53, -51, -48, -45, -44, -43, -39, -38, -36, -35, -34,
        -33, -33, -29, -29, -27, -26, -24, -20, -20, -17, -16, -15, -14, -12,
        -12, -11, -10, -10, -8, -7, -1, 0, 0, 0, 1, 1, 1, 3, 5, 6, 9, 11, 11,
        12, 13, 14, 15, 20, 22, 23, 23, 25, 26, 28, 30, 31, 32, 32, 35, 37,
        40, 41, 42, 43, 44, 48, 48, 49, 51, 51, 56, 62, 64, 70, 72, 76, 77,
        78, 80, 81, 81, 82, 84, 87, 88, 90, 91, 93, 93, 95, 97, 98, 99, 102,
        102, 104, 106, 106, 109, 112, 113, 115, 115, 117, 124, 125,
    ];
    assert!(v.iter().is_sorted());
}
