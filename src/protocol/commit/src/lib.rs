//! Inner+outer commitments per Sec. 2.5 (Eq. (4)) and Commit/Open in Fig. 4.  :contentReference[oaicite:3]{index=3}

use greyhound_ring::{ModQ, Poly, D};
use greyhound_gadget::{digits_for, g_inv_vec, g_fwd_vec};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub type PolyVec = Vec<Poly>;

/// Simple dense matrix over R_q, stored row-major.
#[derive(Clone)]
pub struct MatrixRq {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Poly>, // rows * cols
}

impl MatrixRq {
    pub fn new(rows: usize, cols: usize, data: Vec<Poly>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }
    pub fn at(&self, r: usize, c: usize) -> &Poly {
        &self.data[r * self.cols + c]
    }
    pub fn mul_vec(&self, x: &PolyVec, q: &ModQ) -> PolyVec {
        assert_eq!(x.len(), self.cols);
        let mut out = vec![Poly::zero(); self.rows];
        for r in 0..self.rows {
            let mut acc = Poly::zero();
            for c in 0..self.cols {
                acc = acc.add(&self.at(r, c).mul(&x[c], q), q);
            }
            out[r] = acc;
        }
        out
    }
    /// Uniform random matrix (toy RNG; swap with CSPRNG later).
    pub fn random(rows: usize, cols: usize, q: &ModQ, rng: &mut StdRng) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            let mut coeffs = [0u32; D];
            for j in 0..D { coeffs[j] = rng.gen::<u32>() % q.q; }
            data.push(Poly::from_coeffs(coeffs, q));
        }
        Self { rows, cols, data }
    }
}

/// Public parameters for commitments (A,B) and bases b0,b1.
#[derive(Clone)]
pub struct CommitParams {
    pub q: ModQ,
    pub n: usize,
    pub m: usize,
    pub r: usize,
    pub delta0: usize,
    pub delta1: usize,
    pub b0: u32,
    pub b1: u32,
    pub A: MatrixRq,                // n x (delta0*m)
    pub B: MatrixRq,                // n x (n*delta1*r)
    pub mu: usize,                // LWE rank for outer randomness
    pub E: Option<MatrixRq>,      // when Some, the scheme is hiding
}

impl CommitParams {
    pub fn gen(q: ModQ, n: usize, m: usize, r: usize, b0: u32, b1: u32, seed: u64) -> Self {
        let delta0 = digits_for(&q, b0);
        let delta1 = digits_for(&q, b1);
        let mut rng = StdRng::seed_from_u64(seed);
        let A = MatrixRq::random(n, delta0 * m, &q, &mut rng);
        let B = MatrixRq::random(n, n * delta1 * r, &q, &mut rng);
        // default: non-hiding
        Self { q, n, m, r, delta0, delta1, b0, b1, A, B, mu: 0, E: None }
    }

    /// Hiding extension: choose μ and E.
    pub fn with_hiding(mut self, mu: usize, seed: u64) -> Self {
        if mu > 0 {
            let mut rng = StdRng::seed_from_u64(seed ^ 0xE11E);
            self.E = Some(MatrixRq::random(self.n, mu, &self.q, &mut rng));
            self.mu = mu;
        }
        self
    }
}

#[derive(Clone)]
pub struct Decommit {
    pub s: Vec<PolyVec>,   // as before
    pub that: PolyVec,     // as before
    pub r: Option<PolyVec> // new: randomness for hiding (length μ) if hiding
}

#[derive(Clone)]
pub struct Commitment {
    pub u: PolyVec,        // length n
    pub dec: Decommit,
}

/// Non-hiding commit: u = B * \hat t
pub fn commit(pp: &CommitParams, f_cols: &[PolyVec]) -> Commitment {
    assert_eq!(f_cols.len(), pp.r);
    for col in f_cols { assert_eq!(col.len(), pp.m); }

    // s_i := G^{-1}_{b0,m}(f_i),  t_i := A s_i,  \hat t_i := G^{-1}_{b1,n}(t_i)
    let mut s_all = Vec::with_capacity(pp.r);
    let mut that_concat: PolyVec = Vec::with_capacity(pp.n * pp.delta1 * pp.r);
    for i in 0..pp.r {
        let si = greyhound_gadget::g_inv_vec(&f_cols[i], pp.b0, &pp.q);
        let ti = pp.A.mul_vec(&si, &pp.q);
        let that_i = greyhound_gadget::g_inv_vec(&ti, pp.b1, &pp.q);
        s_all.push(si);
        that_concat.extend(that_i);
    }
    let u = pp.B.mul_vec(&that_concat, &pp.q);
    Commitment { u, dec: Decommit { s: s_all, that: that_concat, r: None } }
}

/// Non-hiding open check (Eq. (4))
pub fn open_check(pp: &CommitParams, u: &PolyVec, f_cols: &[PolyVec], dec: &Decommit) -> bool {
    if dec.r.is_some() { return false; }                 // should be None in non-hiding
    if dec.s.len() != pp.r { return false; }
    if dec.that.len() != pp.n * pp.delta1 * pp.r { return false; }
    if u.len() != pp.n { return false; }

    // Check G_{b0,m} s_i = f_i
    for i in 0..pp.r {
        if dec.s[i].len() != pp.delta0 * pp.m { return false; }
        let fi_rec = greyhound_gadget::g_fwd_vec(&dec.s[i], pp.m, pp.b0, &pp.q);
        if fi_rec != f_cols[i] { return false; }
    }
    // Check A s_i = G_{b1,n} \hat t_i
    let block = pp.n * pp.delta1;
    for i in 0..pp.r {
        let that_i = &dec.that[i*block .. (i+1)*block];
        let ti_rec = greyhound_gadget::g_fwd_vec(that_i, pp.n, pp.b1, &pp.q);
        let ti_from_A = pp.A.mul_vec(&dec.s[i], &pp.q);
        if ti_rec != ti_from_A { return false; }
    }
    // Check u = B \hat t
    let u_chk = pp.B.mul_vec(&dec.that, &pp.q);
    u_chk == *u
}


/// Hiding commit: u = B \hat t + E r   (Sec. 4.5).  :contentReference[oaicite:2]{index=2}
pub fn commit_hiding(pp: &CommitParams, f_cols: &[PolyVec]) -> Commitment {
    assert!(pp.E.is_some() && pp.mu > 0, "call with_hiding() first");
    assert_eq!(f_cols.len(), pp.r);
    for col in f_cols { assert_eq!(col.len(), pp.m); }

    // same as non-hiding path: s_i, t_i, \hat t_i
    let mut s_all = Vec::with_capacity(pp.r);
    let mut that_concat: PolyVec = Vec::with_capacity(pp.n * pp.delta1 * pp.r);
    for i in 0..pp.r {
        let si = g_inv_vec(&f_cols[i], pp.b0, &pp.q);
        let ti = pp.A.mul_vec(&si, &pp.q);
        let that_i = g_inv_vec(&ti, pp.b1, &pp.q);
        s_all.push(si);
        that_concat.extend(that_i);
    }

    // r ∈ R_q^μ (toy: uniform; later swap for narrow mod-b)
    let mut rng = StdRng::seed_from_u64(0xC001);
    let mut r = Vec::with_capacity(pp.mu);
    for _ in 0..pp.mu {
        let mut c = [0u32; D];
        for t in 0..D { c[t] = rng.gen::<u32>() % pp.q.q; }
        r.push(Poly { c });
    }

    // u = B \hat t + E r
    let mut u = pp.B.mul_vec(&that_concat, &pp.q);
    let Er = pp.E.as_ref().unwrap().mul_vec(&r, &pp.q);
    for i in 0..u.len() { u[i] = u[i].add(&Er[i], &pp.q); }

    Commitment { u, dec: Decommit { s: s_all, that: that_concat, r: Some(r) } }
}

pub fn open_check_hiding(pp: &CommitParams, u: &PolyVec, f_cols: &[PolyVec], dec: &Decommit) -> bool {
    if pp.E.is_none() || pp.mu == 0 || dec.r.is_none() { return false; }
    // reuse algebraic checks from non-hiding
    if !open_check(pp, u, f_cols, &Decommit { s: dec.s.clone(), that: dec.that.clone(), r: None }) { return false; }

    // check E r == u - B \hat t
    let Er = pp.E.as_ref().unwrap().mul_vec(dec.r.as_ref().unwrap(), &pp.q);
    let Bu = pp.B.mul_vec(&dec.that, &pp.q);
    for i in 0..u.len() {
        if u[i].sub(&Bu[i], &pp.q) != Er[i] { return false; }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{commit, open_check};
    use greyhound_ring::{ModQ, Poly};

    fn rand_poly(q: &ModQ, rng: &mut StdRng) -> Poly {
        let mut c = [0u32; D];
        for j in 0..D { c[j] = rng.gen::<u32>() % q.q; }
        Poly::from_coeffs(c, q)
    }

    #[test]
    fn commit_open_roundtrip() {
        // Tiny, toy parameters (for speed). Dimensions follow Fig. 4 shapes.  :contentReference[oaicite:6]{index=6}
        let q = ModQ::new(229);
        let n = 2usize;      // SIS rank
        let m = 3usize;      // rows in each f_i
        let r = 2usize;      // number of columns
        let b0 = 6u32;
        let b1 = 7u32;

        let mut rng = StdRng::seed_from_u64(7);
        let pp = CommitParams::gen(q, n, m, r, b0, b1, 42);

        // Random message matrix S = [f1 | f2] with each f_i \in R_q^m
        let mut f_cols: Vec<PolyVec> = Vec::with_capacity(r);
        for _ in 0..r {
            let mut col = Vec::with_capacity(m);
            for _ in 0..m { col.push(rand_poly(&q, &mut rng)); }
            f_cols.push(col);
        }

        let Commitment { u, dec } = commit(&pp, &f_cols);
        assert!(open_check(&pp, &u, &f_cols, &dec));
    }
}

// crates/commit/src/lib.rs
impl MatrixRq {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![Poly::zero(); rows*cols] }
    }
    pub fn set(&mut self, r: usize, c: usize, val: Poly) {
        self.data[r * self.cols + c] = val;
    }
}
