//! A tiny library with one bug, used by the phase 5 agent test.

/// Adds two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a - b
}

/// Multiplies two numbers.
pub fn mul(a: i32, b: i32) -> i32 {
    a * b
}

/// True when `n` is even.
pub fn is_even(n: i32) -> bool {
    n % 2 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_small_numbers() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    fn adds_negative_numbers() {
        assert_eq!(add(-4, 1), -3);
    }

    #[test]
    fn multiplies() {
        assert_eq!(mul(3, 4), 12);
    }

    #[test]
    fn even_and_odd() {
        assert!(is_even(10));
        assert!(!is_even(7));
    }
}
