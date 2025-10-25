//! Gadget decomposition G^{-1}_{b,n} over R_q (Sec. 2.1), and helpers
//! Used for \hat{w}=G^{-1}_{b1,r}(w) and \hat{t_i}=G^{-1}_{b1,n}(t_i) (Fig. 1, Eq. (3)),
//! and Commit/Open (Eq. (4)).  :contentReference[oaicite:2]{index=2}

use greyhound_ring::{ModQ, Poly, D};

/// Compute δ = ceil(log_b(q)) without floating point.
#[inline]
pub fn digits_for(q: &ModQ, base: u32) -> usize {
    assert!(base >= 2, "base must be >= 2");
    let mut pow: u128 = 1;
    let b = base as u128;
    let qv = q.q as u128;
    let mut delta = 0usize;
    while pow < qv {
        pow *= b;
        delta += 1;
    }
    delta
}

/// Convert a residue in [0,q) to its signed representative in [-(q-1)/2, (q-1)/2].
#[inline]
fn signed_rep(x: u32, q: &ModQ) -> i64 {
    let q64 = q.q as i64;
    let xi  = x as i64;
    if xi > q64 / 2 { xi - q64 } else { xi }
}

/// Map a signed integer back to [0,q) (canonical residue).
#[inline]
fn canon_mod_q(x: i64, q: &ModQ) -> u32 {
    let q64 = q.q as i64;
    let mut y = x % q64;
    if y < 0 { y += q64; }
    y as u32
}

/// Multiply a polynomial by a small scalar (u32) modulo q.
#[inline]
fn poly_scale_small(p: &Poly, k: u32, q: &ModQ) -> Poly {
    let mut out = [0u32; D];
    for i in 0..D {
        let t = ((p.c[i] as u64) * (k as u64)) % (q.q as u64);
        out[i] = t as u32;
    }
    Poly { c: out }
}

/// Decompose a single coefficient x ∈ [0,q) into δ balanced base-b digits.
/// Returns digits d_0,...,d_{δ-1} as residues in [0,q), each representing a signed
/// value in [ -⌊b/2⌋ , ⌊b/2⌋ ].
#[inline]
fn decompose_coeff_balanced(x: u32, base: u32, delta: usize, q: &ModQ) -> [u32; 32] {
    assert!(delta <= 32, "delta too large for fixed buffer; bump if needed");
    let mut out = [0u32; 32];

    let mut y = signed_rep(x, q) as i64;     // work with a small signed integer
    let b = base as i64;
    let half = (base / 2) as i64;            // floor(b/2)

    for i in 0..delta {
        // Euclidean remainder in 0..b-1
        let r = ((y % b) + b) % b;
        // Balance into [-floor(b/2), floor(b/2)]
        let di = if r > half { r - b } else { r };
        out[i] = canon_mod_q(di as i64, q);
        // Update quotient
        y = (y - di) / b;
    }
    // With δ=ceil(log_b q) and |x|<=q/2, y should be 0.
    debug_assert!(y == 0, "non-zero carry after balanced decomposition");
    out
}

/// Recompose digits (length δ) back to a residue in [0,q).
#[inline]
fn recompose_coeff(digits: &[u32], base: u32, q: &ModQ) -> u32 {
    let mut acc: i128 = 0;
    let mut pow: i128 = 1;                  // b^0
    let b = base as i128;
    for &d in digits {
        let ds = signed_rep(d, q) as i128;  // interpret balanced digit
        acc += ds * pow;
        pow *= b;
    }
    canon_mod_q(acc as i64, q)
}

/// Decompose a polynomial into δ polynomials of balanced digits.
pub fn decompose_poly_balanced(p: &Poly, base: u32, delta: usize, q: &ModQ) -> Vec<Poly> {
    let mut out = vec![Poly::zero(); delta];
    for j in 0..D {
        let digs = decompose_coeff_balanced(p.c[j], base, delta, q);
        for i in 0..delta {
            out[i].c[j] = digs[i];
        }
    }
    out
}

/// Recompose δ digit-polynomials back to a polynomial.
pub fn recompose_poly(digits: &[Poly], base: u32, q: &ModQ) -> Poly {
    let delta = digits.len();
    // Precompute b^i mod q
    let mut powers = vec![1u32; delta];
    for i in 1..delta {
        powers[i] = ((powers[i-1] as u64 * base as u64) % (q.q as u64)) as u32;
    }
    // acc = sum_i digits[i] * b^i
    let mut acc = Poly::zero();
    for i in 0..delta {
        let term = poly_scale_small(&digits[i], powers[i], q);
        acc = acc.add(&term, q);
    }
    acc
}

/// G^{-1}_{b,n} on a vector in R_q^n: concatenate δ digits for each coordinate.
pub fn g_inv_vec(vec: &[Poly], base: u32, q: &ModQ) -> Vec<Poly> {
    let delta = digits_for(q, base);
    let mut out = Vec::with_capacity(vec.len() * delta);
    for p in vec {
        let ds = decompose_poly_balanced(p, base, delta, q);
        out.extend(ds);
    }
    out
}

/// G_{b,n} on δn digits: recomposes to R_q^n.
/// Expects input arranged as [digits(coord0) || digits(coord1) || ...].
pub fn g_fwd_vec(digits: &[Poly], n: usize, base: u32, q: &ModQ) -> Vec<Poly> {
    let delta = digits.len() / n;
    let mut res = Vec::with_capacity(n);
    for i in 0..n {
        let chunk = &digits[i*delta .. (i+1)*delta];
        res.push(recompose_poly(chunk, base, q));
    }
    res
}

// --------------------- Tests ---------------------
#[cfg(test)]
mod tests {
    use super::*;
    use greyhound_ring::{ModQ, Poly};

    #[test]
    fn coeff_roundtrip_balanced() {
        let q = ModQ::new(229); // 229 ≡ 5 (mod 8)
        let base = 6;
        let delta = digits_for(&q, base);
        for x in [0, 1, 2, 3, 114, 228] {
            let digs = decompose_coeff_balanced(x, base, delta, &q);
            let x2 = recompose_coeff(&digs[..delta], base, &q);
            assert_eq!(x, x2);
            // Bound check: |digit| <= floor(b/2)
            let half = (base / 2) as i64;
            for i in 0..delta {
                let si = signed_rep(digs[i], &q);
                assert!(si.abs() <= half);
            }
        }
    }

    #[test]
    fn poly_roundtrip_balanced() {
        let q = ModQ::new(229);
        let base = 7;
        let delta = digits_for(&q, base);

        let mut p = Poly::zero();
        for i in 0..D { p.c[i] = ((i*17 + 5) as u32) % q.q; }

        let ds = decompose_poly_balanced(&p, base, delta, &q);
        let p2 = recompose_poly(&ds, base, &q);
        assert_eq!(p, p2);

        // Norm bound for digits
        let half = (base / 2) as i64;
        for d in ds {
            for &coeff in &d.c {
                let s = signed_rep(coeff, &q).abs();
                assert!(s <= half);
            }
        }
    }

    #[test]
    fn vec_roundtrip() {
        let q = ModQ::new(229);
        let base = 6;
        let n = 3;

        // Make a small vector in R_q^n
        let mut v = Vec::new();
        for j in 0..n {
            let mut p = Poly::zero();
            for i in 0..D { p.c[i] = ((i as u32 + j as u32 * 9) % q.q); }
            v.push(p);
        }
        let digits = g_inv_vec(&v, base, &q);
        let rec = g_fwd_vec(&digits, n, base, &q);
        assert_eq!(v, rec);
    }
}
