//! Eq. (3) builder + helpers: compute w, \hat w, v; build P,h; FS challenges.

use greyhound_ring::{ModQ, Poly, D};
use greyhound_gadget::{g_inv_vec};
use greyhound_commit::{MatrixRq, CommitParams, PolyVec};
use greyhound_transcript::Fs;
use rand::{SeedableRng, rngs::StdRng};

#[derive(Clone)]
pub struct ProtoParams<'a> {
    pub commit: &'a CommitParams, // A,B,b0,b1,δ0,δ1,n,m,r,q
    pub D: MatrixRq,              // n x (δ1*r)
}

// === Add near your other structs ===
pub struct HvzkBuilders<'a> {
    pub pp: &'a CommitParams,  // A,B etc.
    pub D0: MatrixRq,          // n x (δ1 r)
    pub D1: MatrixRq,          // n x (δ1 L)
    pub E0: MatrixRq,          // n x μv
    pub L: usize,
}

pub struct HvzkPublic<'a> {
    pub a: &'a Vec<Poly>,         // δ0 m
    pub b: &'a Vec<Poly>,         // r
    pub u: &'a Vec<Poly>,         // n
    pub v: &'a Vec<Poly>,         // n
    pub j: Vec<Poly>,             // L ring polys (ct(j_i) will be checked outside)
    pub alpha: Vec<u32>,          // L field scalars
    pub sigma_inv_x: Poly,        // σ^{-1}(x)
}

// === Complete Eq.(14): columns are [what | lhat | rv | that | r | z] ===
pub fn build_eq14(
    B: &HvzkBuilders,
    q: &ModQ,
    pubin: &HvzkPublic,
) -> (MatrixRq, Vec<Poly>) {
    let pp = B.pp;
    let delta1 = pp.delta1;
    let n = pp.n;
    let rcols = pp.r;

    let off_w  = 0;
    let off_l  = off_w + delta1 * rcols;
    let off_rv = off_l + delta1 * B.L;
    let off_t  = off_rv + B.E0.cols;
    let off_r  = off_t + (n * delta1 * rcols);
    let off_z  = off_r + pp.mu;

    let rows = n       // v rows: [D0 | D1 | E0] * [what|lhat|rv] = v
             + n       // u rows: [B | E] * [that|r] = u
             + B.L     // L rows: α_i σ^{-1}(x) b^T G . what  +  e_i G_L . lhat = j_i
             + 1       // c^T G . what  - a^T . z = 0 (we’ll let caller append c, a later if desired)
             + n;      // (c^T ⊗ G_n) . that  - A . z = 0 (same, caller appends c)

    let cols = off_z + (pp.delta0 * pp.m);
    let mut P = MatrixRq::zeros(rows, cols);
    let mut h = Vec::<Poly>::with_capacity(rows);

    // Block 1: v = D0 what + D1 lhat + E0 rv
    for rr in 0..n {
        for j in 0..(delta1 * rcols) { P.set(rr, off_w + j, B.D0.at(rr, j).clone()); }
        for j in 0..(delta1 * B.L)   { P.set(rr, off_l + j, B.D1.at(rr, j).clone()); }
        for j in 0..B.E0.cols        { P.set(rr, off_rv + j, B.E0.at(rr, j).clone()); }
        h.push(pubin.v[rr].clone());
    }

    // Block 2: u = B that + E r
    for rr in 0..n {
        let dst = n + rr;
        for j in 0..(n * delta1 * rcols) { P.set(dst, off_t + j, pp.B.at(rr, j).clone()); }
        let E = pp.E.as_ref().expect("hiding E");
        for j in 0..pp.mu { P.set(dst, off_r + j, E.at(rr, j).clone()); }
        h.push(pubin.u[rr].clone());
    }

    // Precompute α_i σ^{-1}(x) * b and its G row on what
    let mut sigb = pubin.b.clone();
    for bi in &mut sigb { *bi = pubin.sigma_inv_x.mul(bi, q); }
    let sigbG = row_vec_times_G(&sigb, pp.b1, delta1, q);

    // Row group: for each i, α_i * sigbG on what  and  e_i * G_L on lhat
    // Precompute powers b1^j as ring scalars
    let mut pow = vec![1u32; delta1];
    for j in 1..delta1 { pow[j] = ((pow[j-1] as u64 * pp.b1 as u64) % (q.q as u64)) as u32; }

    for i in 0..B.L {
        let dst = 2*n + i;

        // α_i * sigbG on what
        let alpha_i = pubin.alpha[i] as u64;
        for j in 0..sigbG.len() {
            let mut c = [0u32; D];
            for t in 0..D { c[t] = ((sigbG[j].c[t] as u64 * alpha_i) % (q.q as u64)) as u32; }
            P.set(dst, off_w + j, Poly { c });
        }

        // e_i * G_{b1,L} on lhat: place δ1 digits in the i-th lhat “slot”
        let start = off_l + i*delta1;
        for j in 0..delta1 {
            let mut c = [0u32; D];
            c[0] = pow[j];
            P.set(dst, start + j, Poly { c });
        }

        // RHS j_i
        h.push(pubin.j[i].clone());
    }

    // NOTE: we *do not* add the last two Eq.(3) blocks here; the PCS glue will
    // append them (so it can plug in the Fiat–Shamir vector c and vector a).
    // Just return what we have; callers will expand P,h further.
    (P, h)
}


impl<'a> ProtoParams<'a> {
    pub fn ensure_dims(&self) {
        let pp = self.commit;
        assert_eq!(self.D.rows, pp.n);
        assert_eq!(self.D.cols, pp.delta1 * pp.r);
    }
}

// w^T = a^T [s1|...|sr]
pub fn compute_w(a: &PolyVec, s: &[PolyVec], q: &ModQ) -> PolyVec {
    let r = s.len();
    let mut w = vec![Poly::zero(); r];
    for i in 0..r {
        let mut acc = Poly::zero();
        assert_eq!(a.len(), s[i].len());
        for j in 0..a.len() {
            acc = acc.add(&a[j].mul(&s[i][j], q), q);
        }
        w[i] = acc;
    }
    w
}

// z = [s1|...|sr] c
pub fn compute_z(s: &[PolyVec], c: &[Poly], q: &ModQ) -> PolyVec {
    let r = s.len(); assert_eq!(r, c.len());
    let ell = s[0].len(); // δ0 m
    let mut z = vec![Poly::zero(); ell];
    for i in 0..r {
        for j in 0..ell {
            z[j] = z[j].add(&s[i][j].mul(&c[i], q), q);
        }
    }
    z
}

// row for b^T G_{b1,r}
pub fn row_vec_times_G(vec: &PolyVec, base: u32, delta: usize, q: &ModQ) -> PolyVec {
    let r = vec.len();
    let mut pow = vec![1u32; delta];
    for j in 1..delta { pow[j] = (pow[j-1] as u64 * base as u64 % q.q as u64) as u32; }
    let mut row = Vec::with_capacity(delta * r);
    for i in 0..r {
        for j in 0..delta {
            let mut coeffs = [0u32; D];
            for t in 0..D { coeffs[t] = ((vec[i].c[t] as u64 * pow[j] as u64) % (q.q as u64)) as u32; }
            row.push(Poly { c: coeffs });
        }
    }
    row
}

// (c^T ⊗ G_{b1,n}) block: n x (n*δ1*r)
pub fn cotimes_G_block(c: &[Poly], n: usize, base: u32, delta: usize, q: &ModQ) -> MatrixRq {
    let r = c.len();
    let cols = n * delta * r;
    let mut M = MatrixRq::zeros(n, cols);
    let mut pow = vec![1u32; delta];
    for j in 1..delta { pow[j] = (pow[j-1] as u64 * base as u64 % q.q as u64) as u32; }
    for row_n in 0..n {
        for i in 0..r {
            for j in 0..delta {
                let col = i * (n*delta) + row_n * delta + j;
                let mut coeffs = [0u32; D];
                for t in 0..D { coeffs[t] = ((c[i].c[t] as u64 * pow[j] as u64) % (q.q as u64)) as u32; }
                M.set(row_n, col, Poly { c: coeffs });
            }
        }
    }
    M
}

// Build P,h as Eq. (3)
pub fn build_linear_system(
    params: &ProtoParams,
    a: &PolyVec, b: &PolyVec, u: &PolyVec, v: &PolyVec, y_rhs: &Poly, c: &[Poly],
) -> (MatrixRq, PolyVec) {
    params.ensure_dims();
    let pp = params.commit; let q = &pp.q;

    let rows = 3*pp.n + 2;
    let cols = pp.delta1*pp.r + (pp.n*pp.delta1*pp.r) + (pp.delta0*pp.m);
    let mut P = MatrixRq::zeros(rows, cols);

    let off_w = 0usize;
    let off_t = off_w + pp.delta1 * pp.r;
    let off_z = off_t + (pp.n * pp.delta1 * pp.r);

    // Block 1: D
    for rrow in 0..pp.n {
        for j in 0..(pp.delta1*pp.r) {
            P.set(rrow, off_w + j, params.D.at(rrow, j).clone());
        }
    }
    // Block 2: B
    for rrow in 0..pp.n {
        let dst = pp.n + rrow;
        for j in 0..(pp.n*pp.delta1*pp.r) {
            P.set(dst, off_t + j, pp.B.at(rrow, j).clone());
        }
    }
    // Row 3: b^T G . w^
    let row3 = 2*pp.n;
    let row_bG = row_vec_times_G(b, pp.b1, pp.delta1, q);
    for j in 0..row_bG.len() { P.set(row3, off_w + j, row_bG[j].clone()); }

    // Row 4: c^T G . w^   and   -a^ on z
    let row4 = 2*pp.n + 1;
    let row_cG = row_vec_times_G(&c.to_vec(), pp.b1, pp.delta1, q);
    for j in 0..row_cG.len() { P.set(row4, off_w + j, row_cG[j].clone()); }
    for j in 0..a.len() { P.set(row4, off_z + j, a[j].neg(q)); }

    // Block 5: n rows — (c^T ⊗ G).t^  and  -A.z
    let block = cotimes_G_block(c, pp.n, pp.b1, pp.delta1, q);
    for rrow in 0..pp.n {
        let dst = 2*pp.n + 2 + rrow;
        for j in 0..(pp.n*pp.delta1*pp.r) { P.set(dst, off_t + j, block.at(rrow, j).clone()); }
        for j in 0..(pp.delta0*pp.m)       { P.set(dst, off_z + j, pp.A.at(rrow, j).neg(q)); }
    }

    // h = [v ; u ; y ; 0 ; 0_n]
    let mut h = Vec::<Poly>::with_capacity(rows);
    for i in 0..pp.n { h.push(v[i].clone()); }
    for i in 0..pp.n { h.push(u[i].clone()); }
    h.push(y_rhs.clone());
    h.push(Poly::zero());
    for _ in 0..pp.n { h.push(Poly::zero()); }

    (P, h)
}

// \hat w and v = D \hat w
pub fn derive_w_hat_and_v(
    pp: &CommitParams,
    dmat: &MatrixRq,
    w: &PolyVec
) -> (PolyVec, PolyVec) {
    let what = g_inv_vec(&w, pp.b1, &pp.q);
    let v    = dmat.mul_vec(&what, &pp.q);
    (what, v)
}


// Paper’s τ1=32, τ2=8 for d=64
pub fn sample_challenge(fs: &Fs, pp: &CommitParams) -> Vec<Poly> {
    fs.challenge_vec(pp.r, &pp.q, 32, 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use greyhound_commit::{commit, CommitParams, Commitment};
    use rand::{Rng, rngs::StdRng};

    #[test]
    fn eq3_is_satisfied() {
        let q = ModQ { q: 229 };
        let n=2usize; let m=3usize; let r=2usize; let b0=6u32; let b1=7u32;
        let pp = CommitParams::gen(q, n, m, r, b0, b1, 7);

        let mut rng = StdRng::seed_from_u64(9);
        // random f columns
        let mut f_cols: Vec<PolyVec> = Vec::with_capacity(r);
        for _ in 0..r {
            let mut col = Vec::with_capacity(m);
            for _ in 0..m {
                let mut c = [0u32; D];
                for t in 0..D { c[t] = rng.gen::<u32>() % q.q; }
                col.push(Poly { c });
            }
            f_cols.push(col);
        }

        // commit → u, s, \hat t
        let Commitment { u, dec } = commit(&pp, &f_cols);

        // random a (δ0 m) and b (r)
        let mut a = Vec::with_capacity(pp.delta0 * m);
        for _ in 0..(pp.delta0 * m) {
            let mut c = [0u32; D]; for t in 0..D { c[t] = rng.gen::<u32>() % q.q; }
            a.push(Poly { c });
        }
        let mut b = Vec::with_capacity(r);
        for _ in 0..r {
            let mut c = [0u32; D]; for t in 0..D { c[t] = rng.gen::<u32>() % q.q; }
            b.push(Poly { c });
        }

        // w, \hat w, v with random D
        let Dm = MatrixRq::random(n, pp.delta1 * r, &q, &mut rng);
        let w  = compute_w(&a, &dec.s, &q);
        let (what, v) = derive_w_hat_and_v(&pp, &Dm, &w);

        // FS challenge c
        let mut fs = Fs::new(b"eq3-test");
        fs.absorb_polyvec(&v).absorb_polyvec(&u);
        let c = sample_challenge(&fs, &pp);

        // z and y
        let z = compute_z(&dec.s, &c, &q);
        let mut y = Poly::zero();
        for i in 0..r { y = y.add(&w[i].mul(&b[i], &q), &q); }

        // build P,h and verify P*[\hat w || \hat t || z] = h
        let proto = ProtoParams { commit: &pp, D: Dm };
        let (P, h) = build_linear_system(&proto, &a, &b, &u, &v, &y, &c);

        let mut Z: PolyVec = Vec::new();
        Z.extend_from_slice(&what);
        Z.extend_from_slice(&dec.that);
        Z.extend_from_slice(&z);

        let lhs = P.mul_vec(&Z, &q);
        assert_eq!(lhs, h);
    }
}
