//! Minimal ring R_q = Z_q[X]/(X^d + 1) with d=64, as used by Greyhound
//! (notation & operations per §§2.1 and 4.1).  :contentReference[oaicite:2]{index=2}

pub const D: usize = 64;

/// Modulus wrapper (caller supplies a 32-bit prime; later we’ll pick q ≡ 5 (mod 8)).
#[derive(Clone, Copy, Debug)]
pub struct ModQ {
    pub q: u32,
}

impl ModQ {
    #[inline] pub fn new(q: u32) -> Self { Self { q } }
    #[inline] pub fn add(&self, a: u32, b: u32) -> u32 {
        let mut x = a as u64 + b as u64;
        if x >= self.q as u64 { x -= self.q as u64; }
        x as u32
    }
    #[inline] pub fn sub(&self, a: u32, b: u32) -> u32 {
        // Return a - b mod q in [0, q)
        if a >= b { a - b } else { (a as u64 + self.q as u64 - b as u64) as u32 }
    }
    #[inline] pub fn neg(&self, a: u32) -> u32 {
        if a == 0 { 0 } else { (self.q as u64 - a as u64) as u32 }
    }
    #[inline] pub fn mul(&self, a: u32, b: u32) -> u32 {
        // 64-bit intermediate is fine for 32-bit q.
        let x = (a as u64) * (b as u64) % (self.q as u64);
        x as u32
    }
}

/// Dense polynomial with coefficients in [0, q), degree < D.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Poly {
    pub c: [u32; D],
}

impl Poly {
    pub fn zero() -> Self { Self { c: [0; D] } }
    pub fn from_coeffs(coeffs: [u32; D], q: &ModQ) -> Self {
        // Normalize into [0,q)
        let mut out = [0u32; D];
        for (i, &v) in coeffs.iter().enumerate() {
            out[i] = v % q.q;
        }
        Self { c: out }
    }
    pub fn monomial(k: usize, a: u32, q: &ModQ) -> Self {
        assert!(k < D);
        let mut p = [0u32; D];
        p[k] = a % q.q;
        Self { c: p }
    }
    #[inline] pub fn ct(&self) -> u32 { self.c[0] } // constant term

    pub fn add(&self, other: &Self, q: &ModQ) -> Self {
        let mut r = [0u32; D];
        for i in 0..D { r[i] = q.add(self.c[i], other.c[i]); }
        Self { c: r }
    }

    pub fn sub(&self, other: &Self, q: &ModQ) -> Self {
        let mut r = [0u32; D];
        for i in 0..D { r[i] = q.sub(self.c[i], other.c[i]); }
        Self { c: r }
    }

    pub fn neg(&self, q: &ModQ) -> Self {
        let mut r = [0u32; D];
        for i in 0..D { r[i] = q.neg(self.c[i]); }
        Self { c: r }
    }

    /// Multiply in R_q = Z_q[X]/(X^D + 1).
    /// Schoolbook O(D^2), with wrap-and-negate for terms of degree ≥ D.
    pub fn mul(&self, other: &Self, q: &ModQ) -> Self {
        let mut acc = [0i128; D]; // signed accumulator for wrap/neg
        for i in 0..D {
            let ai = self.c[i] as i128;
            for j in 0..D {
                let prod = ai * (other.c[j] as i128);
                let k = i + j;
                if k < D {
                    acc[k] += prod;
                } else {
                    // X^{i+j} = X^{k-D} * X^D ≡ -X^{k-D}
                    acc[k - D] -= prod;
                }
            }
        }
        // Reduce mod q into [0, q)
        let qq = q.q as i128;
        let mut out = [0u32; D];
        for i in 0..D {
            // ((acc % q) + q) % q to canonicalize
            let mut v = acc[i] % qq;
            if v < 0 { v += qq; }
            out[i] = v as u32;
        }
        Self { c: out }
    }

    /// σ^{-1}: X ↦ X^{-1} in R_q (see §4.1). For a = ∑ a_i X^i:
    /// a(X^{-1}) ≡ a_0 + ∑_{i=1}^{D-1} (-a_i) X^{D-i} (mod X^D+1).
    pub fn sigma_inv(&self, q: &ModQ) -> Self {
        let mut b = [0u32; D];
        b[0] = self.c[0];
        for i in 1..D {
            // coefficient at X^{D - i} is -a_i
            let pos = D - i;
            b[pos] = q.neg(self.c[i]);
        }
        Self { c: b }
    }
}

// ---- simple tests (run `cargo test -p greyhound-ring`) ----
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sub_roundtrip() {
        let q = ModQ::new(229); // toy prime (229 ≡ 5 mod 8)
        let mut a = Poly::zero();
        a.c[0] = 5; a.c[1] = 7; a.c[63] = 200;
        let mut b = Poly::zero();
        b.c[0] = 228; b.c[1] = 10; b.c[10] = 3;

        let s = a.add(&b, &q);
        let t = s.sub(&b, &q);
        assert_eq!(t, a);
        assert_eq!(s.c[0], 4); // (5 + 228) mod 229 = 4
    }

    #[test]
    fn mul_wrap_and_negate() {
        let q = ModQ::new(229);
        // (X^{63}) * X ≡ -1  (since X^{64} ≡ -1)
        let x63 = Poly::monomial(63, 1, &q);
        let x = Poly::monomial(1, 1, &q);
        let prod = x63.mul(&x, &q);
        let one = Poly::monomial(0, 1, &q);
        assert_eq!(prod, one.neg(&q));
    }

    #[test]
    fn sigma_inv_is_involution() {
        let q = ModQ::new(229);
        let mut a = Poly::zero();
        for i in 0..D { a.c[i] = (i as u32 * 3 + 7) % q.q; }
        let b = a.sigma_inv(&q);
        let c = b.sigma_inv(&q);
        assert_eq!(a, c);
        // constant term preserved
        assert_eq!(a.ct(), b.ct());
    }
}
