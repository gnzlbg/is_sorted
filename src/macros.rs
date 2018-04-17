/// Utility macros

/// Checks whether a slice is sorted until the first element aligned with a
/// $boundary (16 for 16-byte boundary). Returns a (i,n,s) tuple where `i` is
/// the index of the next element in the slice aligned to a 16-byte boundary,
/// `n` the slice length, and `s` the slice.
macro_rules! is_sorted_lt_until_alignment_boundary {
    ($s:ident, $ty:ident, $boundary:expr) => {{
        let n = $s.len() as isize;
        // If the slice has zero or one elements, it is sorted:
        if n < 2 {
            return true;
        }

        let mut i: isize = 0;

        // The first element of the slice might not be aligned to a
        // 16-byte boundary. Handle the elements until the
        // first 16-byte boundary using the scalar algorithm
        {
            let mut a =
                $s.as_ptr().align_offset($boundary) / ::mem::size_of::<$ty>();
            while a > 0 && i < n - 1 {
                if $s.get_unchecked(i as usize)
                    > $s.get_unchecked(i as usize + 1)
                {
                    return false;
                }
                i += 1;
                a -= 1;
            }
            debug_assert!(
                i == n - 1
                    || $s.as_ptr().offset(i).align_offset($boundary) == 0
            );
        }

        i
    }};
}

/// Handles the tail of the `slice` `s` of length `n` starting at index `i`
/// using a scalar loop:
macro_rules! is_sorted_lt_tail {
    ($s:ident, $n:ident, $i:ident) => {{
        // Handle the tail of the slice using the scalar algoirithm:
        while $i < $n - 1 {
            if $s.get_unchecked($i as usize)
                > $s.get_unchecked($i as usize + 1)
            {
                return false;
            }
            $i += 1;
        }
        debug_assert!($i == $n - 1);
        true
    }};
}

macro_rules! is_sorted_gt_until_alignment_boundary {
    ($s:ident, $ty:ident, $boundary:expr) => {{
        let n = $s.len() as isize;
        // If the slice has zero or one elements, it is sorted:
        if n < 2 {
            return true;
        }

        let mut i: isize = 0;

        // The first element of the slice might not be aligned to a
        // 16-byte boundary. Handle the elements until the
        // first 16-byte boundary using the scalar algorithm
        {
            let mut a =
                $s.as_ptr().align_offset($boundary) / ::mem::size_of::<$ty>();
            while a > 0 && i < n - 1 {
                if $s.get_unchecked(i as usize)
                    < $s.get_unchecked(i as usize + 1)
                {
                    return false;
                }
                i += 1;
                a -= 1;
            }
            debug_assert!(
                i == n - 1
                    || $s.as_ptr().offset(i).align_offset($boundary) == 0
            );
        }

        i
    }};
}

macro_rules! is_sorted_gt_tail {
    ($s:ident, $n:ident, $i:ident) => {{
        // Handle the tail of the slice using the scalar algoirithm:
        while $i < $n - 1 {
            if $s.get_unchecked($i as usize)
                < $s.get_unchecked($i as usize + 1)
            {
                return false;
            }
            $i += 1;
        }
        debug_assert!($i == $n - 1);
        true
    }};
}
