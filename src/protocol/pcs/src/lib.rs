//! Greyhound PCS glue (Fig. 4): Z_q ↔ R_q bridge, Commit, Eval.P/V (bring-up).
//! We scale the b^T G row by σ^{-1}(x) so the RHS stays y ∈ R_q (no ring inverse yet).  :contentReference[oaicite:6]{index=6}

use greyhound_ring::{ModQ, Poly, D};
use greyhound_commit as cm;
use greyhound_gadget::g_inv_vec;
use greyhound_proto as pr;
use greyhound_transcript::Fs;
use rand::SeedableRng;
use rand::Rng;

pub type PolyVec = Vec<Poly>;

#[derive(Clone)]
pub struct PcsParams {
    pub q: ModQ,
    pub N: usize,      // degree bound over Z_q[X]
    pub d: usize,      // ring dimension (fixed 64 in paper)
    pub m: usize,
    pub r: usize,
    pub commit: cm::CommitParams, // A,B,b0,b1,δ0,δ1,n
    pub D: cm::MatrixRq,          // n x (δ1*r)
}

#[derive(Clone)]
pub struct Commitment(pub PolyVec);   // u ∈ R_q^n

#[derive(Clone)]
pub struct Decommit {                 // s_i and \hat t  (from Step 3)
    pub s: Vec<PolyVec>,
    pub that: PolyVec,
}

#[derive(Clone)]
pub struct Proof {
    pub y_ring: Poly,   // prover’s y ∈ R_q (Eval.P line 11)
    pub v: PolyVec,     // n ring elements (v = D \hat w)
    pub what: PolyVec,  // bring-up: reveal witness Z = [\hat w || \hat t || z]
    pub that: PolyVec,
    pub z: PolyVec,
}

#[derive(Clone)]
pub struct HvzkParams {
    pub D0: cm::MatrixRq,   // n x (δ1 * r)
    pub D1: cm::MatrixRq,   // n x (δ1 * L)
    pub E0: cm::MatrixRq,   // n x μv
    pub L: usize,           // number of masks
    pub mu_v: usize,        // rank for r_v
}

#[derive(Clone)]
pub struct PcsParamsHvzk {
    pub pcs: PcsParams,     // same as Step 5, but commit = with_hiding(mu)
    pub hvzk: HvzkParams,
}

pub fn setup_hvzk_toy(N: usize, q: ModQ, seed: u64, L: usize, mu: usize, mu_v: usize)
-> PcsParamsHvzk {
    let base = setup_toy(N, q, seed);
    let commit = base.commit.clone().with_hiding(mu, seed ^ 0xBEEF);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xD0D0);
    let D0 = cm::MatrixRq::random(commit.n, commit.delta1 * commit.r, &q, &mut rng);
    let D1 = cm::MatrixRq::random(commit.n, commit.delta1 * L, &q, &mut rng);
    let E0 = cm::MatrixRq::random(commit.n, mu_v, &q, &mut rng);
    PcsParamsHvzk {
        pcs: PcsParams { commit, ..base },
        hvzk: HvzkParams { D0, D1, E0, L, mu_v }
    }
}

#[derive(Clone)]
pub struct ProofHvzkClear {
    pub v: PolyVec,     // first message
    pub j: PolyVec,     // L ring polys
    pub y_field: u32,   // ct(y_ring)
    pub what: PolyVec,  // witness (clear)
    pub lhat: PolyVec,
    pub rv: PolyVec,
    pub that: PolyVec,
    pub r: PolyVec,
    pub z: PolyVec,
}

pub fn eval_prove_hvzk_clear(
    pp: &PcsParamsHvzk,
    x_field: u32,
    f_coeffs: &[u32],
) -> (Commitment, ProofHvzkClear) {
    let q = &pp.pcs.q;

    // Commit (hiding)
    let blocks = (pp.pcs.N + pp.pcs.d - 1) / pp.pcs.d;
    let blocks_vec = pack_poly_to_ring_blocks(q, f_coeffs, blocks);
    let f_cols = make_columns(&blocks_vec, pp.pcs.m, pp.pcs.r);
    let cm::Commitment { u, dec } = cm::commit_hiding(&pp.pcs.commit, &f_cols);
    let comm = Commitment(u.clone());

    // y_ring and y_field
    let x_ring = embed_x(q, x_field);
    let x_d = pow_poly(x_ring.clone(), D, q);
    let sigma_inv_x = x_ring.sigma_inv(q);

    let mut y_ring = Poly::zero();
    let mut x_d_pow = Poly::monomial(0, 1 % q.q, q);
    for f_i in &blocks_vec {
        let term = sigma_inv_x.mul(&f_i.mul(&x_d_pow, q), q);
        y_ring = y_ring.add(&term, q);
        x_d_pow = x_d_pow.mul(&x_d, q);
    }
    let y_field = y_ring.ct();

    // a, b (scale b by σ^{-1}(x) as in Step 5)
    let a = build_a_digits(&pp.pcs, &x_d);
    let mut b = build_b(&pp.pcs, &x_d);
    for bi in &mut b { *bi = sigma_inv_x.mul(bi, q); }

    // w, \hat w
    let w = pr::compute_w(&a, &dec.s, q);
    let what = g_inv_vec(&w, pp.pcs.commit.b1, q);

    // masks l with ct(l)=0 and \hat l
    let mut l = Vec::with_capacity(pp.hvzk.L);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x1CE);
    for _ in 0..pp.hvzk.L {
        let mut c = [0u32; D];
        for t in 1..D { c[t] = rng.gen::<u32>() % q.q; }
        c[0] = 0;
        l.push(Poly { c });
    }
    let lhat = g_inv_vec(&l, pp.pcs.commit.b1, q);

    // first message v = D0 \hat w + D1 \hat l + E0 r_v
    let mut rv = Vec::with_capacity(pp.hvzk.mu_v);
    for _ in 0..pp.hvzk.mu_v {
        let mut c = [0u32; D];
        for t in 0..D { c[t] = rng.gen::<u32>() % q.q; }
        rv.push(Poly { c });
    }
    let mut v = pp.hvzk.D0.mul_vec(&what, q);
    let D1l = pp.hvzk.D1.mul_vec(&lhat, q);
    for i in 0..v.len() { v[i] = v[i].add(&D1l[i], q); }
    let E0rv = pp.hvzk.E0.mul_vec(&rv, q);
    for i in 0..v.len() { v[i] = v[i].add(&E0rv[i], q); }

    // FS → c and α; j_i = l_i + α_i * y_ring
    let mut fs = Fs::new(b"greyhound/pcs-hvzk");
    fs.absorb_polyvec(&v).absorb_polyvec(&u).absorb_u64(x_field as u64);
    let c = pr::sample_challenge(&fs, &pp.pcs.commit);
    let alpha = fs.alphas(pp.hvzk.L, q);

    let mut j = Vec::with_capacity(pp.hvzk.L);
    for i in 0..pp.hvzk.L {
        // j_i = l_i + α_i * y_ring
        let mut scaled = [0u32; D];
        for t in 0..D { scaled[t] = ((y_ring.c[t] as u64 * alpha[i] as u64) % (q.q as u64)) as u32; }
        j.push(l[i].add(&Poly { c: scaled }, q));
    }

    // Build Eq.(14) (top part)
    let builders = pr::HvzkBuilders { pp: &pp.pcs.commit, D0: pp.hvzk.D0.clone(), D1: pp.hvzk.D1.clone(), E0: pp.hvzk.E0.clone(), L: pp.hvzk.L };
    let pubin = pr::HvzkPublic { a: &a, b: &b, u: &u, v: &v, j: j.clone(), alpha: alpha.clone(), sigma_inv_x: sigma_inv_x.clone() };
    let (mut P, mut h) = pr::build_eq14(&builders, q, &pubin);

    // Append the last two Eq.(3) blocks:
    //  - row: c^T G . what  and  -a^ on z  equals 0
    let row_cG = pr::row_vec_times_G(
        &c.iter().cloned().collect::<Vec<_>>(),
        pp.pcs.commit.b1,
        pp.pcs.commit.delta1,
        q
    );
    // This row is right before the final n-row block:
    let row_idx = P.rows - pp.pcs.commit.n - 1;

    // Offsets: what starts at 0; z starts at the last δ0*m columns
    for jcol in 0..row_cG.len() {
        P.set(row_idx, /*off_w*/ 0 + jcol, row_cG[jcol].clone());
    }
    let off_z = P.cols - (pp.pcs.commit.delta0 * pp.pcs.commit.m);
    for jcol in 0..a.len() {
        P.set(row_idx, off_z + jcol, a[jcol].neg(q));
    }
    h.push(Poly::zero());

    // 2) Final n-row block: (c^T ⊗ G_{b1,n}) on that  and  -A on z, RHS = 0
    let block = pr::cotimes_G_block(
        &c,
        pp.pcs.commit.n,
        pp.pcs.commit.b1,
        pp.pcs.commit.delta1,
        q
    );

    // Column offset of t^: [what | lhat | rv | that | r | z]
    let off_t = (pp.pcs.commit.delta1 * pp.pcs.commit.r)  // what
            + (pp.hvzk.L * pp.pcs.commit.delta1)        // lhat
            + (pp.hvzk.mu_v);                            // rv

    for rr in 0..pp.pcs.commit.n {
        let dst = P.rows - pp.pcs.commit.n + rr;

        // (c^T ⊗ G_n) on that
        for jcol in 0..(pp.pcs.commit.n * pp.pcs.commit.delta1 * pp.pcs.commit.r) {
            P.set(dst, off_t + jcol, block.at(rr, jcol).clone());
        }

        // -A on z
        for jcol in 0..(pp.pcs.commit.delta0 * pp.pcs.commit.m) {
            P.set(dst, off_z + jcol, pp.pcs.commit.A.at(rr, jcol).neg(q));
        }

        h.push(Poly::zero());
    }

    // Witness Z = [what | lhat | rv | that | r | z]
    let z_amort = pr::compute_z(&dec.s, &c, q);
    let proof = ProofHvzkClear {
        v, j, y_field,
        what: what.clone(),
        lhat: lhat.clone(),
        rv,
        that: dec.that.clone(),
        r:   dec.r.clone().expect("hiding r"),
        z:   z_amort.clone(),
    };
    (comm, proof)
}

pub fn eval_verify_hvzk_clear(
    pp: &PcsParamsHvzk,
    comm: &Commitment,
    x_field: u32,
    proof: &ProofHvzkClear,
) -> bool {
    let q = &pp.pcs.q;

    // ct(j_i) == α_i * y_field (leaks only y)
    let mut fs = Fs::new(b"greyhound/pcs-hvzk");
    fs.absorb_polyvec(&proof.v).absorb_polyvec(&comm.0).absorb_u64(x_field as u64);
    let alpha = fs.alphas(pp.hvzk.L, q);
    for i in 0..pp.hvzk.L {
        let expect = ((alpha[i] as u64) * (proof.y_field as u64) % (q.q as u64)) as u32;
        if proof.j[i].ct() != expect { return false; }
    }

    // Rebuild a,b,c, σ^{-1}(x) and Eq.(14) fully, then check PZ=h
    let x_ring = embed_x(q, x_field);
    let x_d = pow_poly(x_ring.clone(), D, q);
    let sigma_inv_x = x_ring.sigma_inv(q);

    let a = build_a_digits(&pp.pcs, &x_d);
    let mut b = build_b(&pp.pcs, &x_d);

    let c = pr::sample_challenge(&fs, &pp.pcs.commit);

    let builders = pr::HvzkBuilders { pp: &pp.pcs.commit, D0: pp.hvzk.D0.clone(), D1: pp.hvzk.D1.clone(), E0: pp.hvzk.E0.clone(), L: pp.hvzk.L };
    let pubin = pr::HvzkPublic { a: &a, b: &b, u: &comm.0, v: &proof.v, j: proof.j.clone(), alpha, sigma_inv_x };

    // Append the same two Eq.(3) blocks (exactly as in prover)…
    // (identical code as in eval_prove_hvzk_clear for the last two blocks)
    // -- snip: paste the same block appends here --

    // Assemble Z
    let (mut P, mut h) = pr::build_eq14(&builders, q, &pubin);

    // === Append the same two Eq.(3) blocks as in the prover ===

    // 1) Single row: c^T G_{b1,r} on what  and  -a^ on z, RHS = 0
    let row_cG = pr::row_vec_times_G(
        &c.iter().cloned().collect::<Vec<_>>(),
        pp.pcs.commit.b1,
        pp.pcs.commit.delta1,
        q
    );
    // This row is right before the final n-row block:
    let row_idx = P.rows - pp.pcs.commit.n - 1;

    // Offsets: what starts at 0; z starts at the last δ0*m columns
    for jcol in 0..row_cG.len() {
        P.set(row_idx, /*off_w*/ 0 + jcol, row_cG[jcol].clone());
    }
    let off_z = P.cols - (pp.pcs.commit.delta0 * pp.pcs.commit.m);
    for jcol in 0..a.len() {
        P.set(row_idx, off_z + jcol, a[jcol].neg(q));
    }
    h.push(Poly::zero());

    // 2) Final n-row block: (c^T ⊗ G_{b1,n}) on that  and  -A on z, RHS = 0
    let block = pr::cotimes_G_block(
        &c,
        pp.pcs.commit.n,
        pp.pcs.commit.b1,
        pp.pcs.commit.delta1,
        q
    );

    // Column offset of t^: [what | lhat | rv | that | r | z]
    let off_t = (pp.pcs.commit.delta1 * pp.pcs.commit.r)  // what
            + (pp.hvzk.L * pp.pcs.commit.delta1)        // lhat
            + (pp.hvzk.mu_v);                           // rv

    for rr in 0..pp.pcs.commit.n {
        let dst = P.rows - pp.pcs.commit.n + rr;

        // (c^T ⊗ G_n) on that
        for jcol in 0..(pp.pcs.commit.n * pp.pcs.commit.delta1 * pp.pcs.commit.r) {
            P.set(dst, off_t + jcol, block.at(rr, jcol).clone());
        }

        // -A on z
        for jcol in 0..(pp.pcs.commit.delta0 * pp.pcs.commit.m) {
            P.set(dst, off_z + jcol, pp.pcs.commit.A.at(rr, jcol).neg(q));
        }

        h.push(Poly::zero());
    }

    // Assemble Z
    let mut Z: PolyVec = Vec::new();
    Z.extend_from_slice(&proof.what);
    Z.extend_from_slice(&proof.lhat);
    Z.extend_from_slice(&proof.rv);
    Z.extend_from_slice(&proof.that);
    Z.extend_from_slice(&proof.r);
    Z.extend_from_slice(&proof.z);

    P.mul_vec(&Z, &pp.pcs.q) == h
}

/// ---- Parameter picker (toy bring-up) ----
/// Choose m & r ≈ sqrt(N/d), small n, bases per Sec. 5/Table 4 patterns (toy).
pub fn setup_toy(N: usize, q: ModQ, seed: u64) -> PcsParams {
    let d = D; // 64
    let blocks = (N + d - 1) / d;
    let base = (blocks as f64).sqrt().ceil() as usize;
    let r = base;
    let m = (blocks + r - 1) / r;

    // Toy SIS ranks; for realistic runs adopt Table 4 (e.g., n=18, n1=7) and tuning.  :contentReference[oaicite:7]{index=7}
    let n = 2usize;

    // Bases (toy). See Sec. 5 for concrete choices; δ computed from q.  :contentReference[oaicite:8]{index=8}
    let b0 = 6u32;
    let b1 = 7u32;

    let commit = cm::CommitParams::gen(q, n, m, r, b0, b1, seed ^ 0xA5A5);
    let dmat = cm::MatrixRq::random(
        n,
        commit.delta1 * r,
        &q,
        &mut rand::rngs::StdRng::seed_from_u64(seed ^ 0x1111),
    );
    PcsParams { q, N, d, m, r, commit, D: dmat }
}

/// ---- Helpers: Z_q → R_q packing and ring powers ----

/// Pack field point x ∈ Z_q as ring element x̄ = Σ_{j=0}^{d-1} x^j X^j (Sec. 4.1).  :contentReference[oaicite:9]{index=9}
pub fn embed_x(q: &ModQ, x: u32) -> Poly {
    let mut c = [0u32; D];
    let mut pow = 1u64;
    for j in 0..D {
        c[j] = (pow % q.q as u64) as u32;
        pow = (pow * x as u64) % q.q as u64;
    }
    Poly { c }
}

/// Raise a ring element to a nonnegative power by square-and-multiply.
pub fn pow_poly(mut base: Poly, mut e: usize, q: &ModQ) -> Poly {
    let mut res = Poly::monomial(0, 1 % q.q, q);
    while e > 0 {
        if e & 1 == 1 { res = res.mul(&base, q); }
        base = base.mul(&base, q);
        e >>= 1;
    }
    res
}

/// Pack f ∈ Z_q[X], deg < N, into ring vector [f_0, …, f_{blocks-1}] with
/// f_i = Σ_{j=0}^{d-1} f_{id+j} X^j (missing coeffs are 0).
pub fn pack_poly_to_ring_blocks(q: &ModQ, f: &[u32], blocks: usize) -> Vec<Poly> {
    let mut out = Vec::with_capacity(blocks);
    for i in 0..blocks {
        let mut c = [0u32; D];
        for j in 0..D {
            let k = i * D + j;
            c[j] = if k < f.len() { f[k] % q.q } else { 0 };
        }
        out.push(Poly { c });
    }
    out
}

/// Build a^T = [1, x^d, …, x^{(m-1)d}] G_{b0,m}  (shape δ0 m).  :contentReference[oaicite:10]{index=10}
fn build_a_digits(pp: &PcsParams, x_d: &Poly) -> PolyVec {
    let delta0 = pp.commit.delta0;
    let b0 = pp.commit.b0;
    let q = &pp.q;

    // Vector of base ring elements a0_j = (x^d)^j
    let mut a0 = Vec::with_capacity(pp.m);
    let mut cur = Poly::monomial(0, 1 % q.q, q);
    for _ in 0..pp.m {
        a0.push(cur.clone());
        cur = cur.mul(x_d, q);
    }
    // Expand with gadget weights: for each j, push a0_j * b0^t  (t=0..δ0-1).
    let mut pow = vec![1u32; delta0];
    for t in 1..delta0 {
        pow[t] = ((pow[t-1] as u64 * b0 as u64) % (q.q as u64)) as u32;
    }
    let mut a = Vec::with_capacity(delta0 * pp.m);
    for j in 0..pp.m {
        for t in 0..delta0 {
            // scale a0_j by small scalar b0^t
            let mut c = [0u32; D];
            for u in 0..D {
                c[u] = ((a0[j].c[u] as u64 * pow[t] as u64) % (q.q as u64)) as u32;
            }
            a.push(Poly { c });
        }
    }
    a
}

/// Build b^T = [1, x^{md}, …, x^{(r-1)md}]  (length r).  :contentReference[oaicite:11]{index=11}
fn build_b(pp: &PcsParams, x_d: &Poly) -> PolyVec {
    let q = &pp.q;
    let x_md = pow_poly(x_d.clone(), pp.m, q);
    let mut b = Vec::with_capacity(pp.r);
    let mut cur = Poly::monomial(0, 1 % q.q, q);
    for _ in 0..pp.r {
        b.push(cur.clone());
        cur = cur.mul(&x_md, q);
    }
    b
}

/// Arrange ring blocks into r columns f_i ∈ R_q^m, as Fig. 4 Commit lines 2–6.  :contentReference[oaicite:12]{index=12}
fn make_columns(blocks: &[Poly], m: usize, r: usize) -> Vec<PolyVec> {
    // blocks length >= m*r; pad with zeros if needed.
    let mut cols = Vec::with_capacity(r);
    for i in 0..r {
        let mut col = Vec::with_capacity(m);
        for j in 0..m {
            let idx = i * m + j;
            col.push(if idx < blocks.len() { blocks[idx].clone() } else { Poly::zero() });
        }
        cols.push(col);
    }
    cols
}

/// ---- Commit (Fig. 4 Commit) ----  :contentReference[oaicite:13]{index=13}
pub fn commit(pp: &PcsParams, f_coeffs: &[u32]) -> (Commitment, Decommit) {
    let blocks = (pp.N + pp.d - 1) / pp.d;
    let blocks_vec = pack_poly_to_ring_blocks(&pp.q, f_coeffs, blocks);
    let f_cols = make_columns(&blocks_vec, pp.m, pp.r);
    let cm::Commitment { u, dec } = cm::commit(&pp.commit, &f_cols);
    (Commitment(u), Decommit { s: dec.s, that: dec.that })
}

/// ---- Eval.P (Fig. 4 Eval.P) ----  :contentReference[oaicite:14]{index=14}
pub fn eval_prove(
    pp: &PcsParams,
    comm: &Commitment,
    x_field: u32,
    f_coeffs: &[u32],
    dec: &Decommit,
) -> (u32 /* y_field */, Proof) {
    let q = &pp.q;

    // Pack f into ring blocks
    let blocks = (pp.N + pp.d - 1) / pp.d;
    let blocks_vec = pack_poly_to_ring_blocks(q, f_coeffs, blocks);

    // x̄, x̄^d, σ^{-1}(x̄)
    let x_ring = embed_x(q, x_field);
    let x_d = pow_poly(x_ring.clone(), D /* d */, q);
    let sigma_inv_x = x_ring.sigma_inv(q);

    // y_ring = Σ σ^{-1}(x̄) * f_i * (x̄^d)^i (Fig. 4, lines 4–5)  :contentReference[oaicite:15]{index=15}
    let mut y_ring = Poly::zero();
    let mut x_d_pow = Poly::monomial(0, 1 % q.q, q);
    for f_i in &blocks_vec {
        let term = sigma_inv_x.mul(&f_i.mul(&x_d_pow, q), q);
        y_ring = y_ring.add(&term, q);
        x_d_pow = x_d_pow.mul(&x_d, q);
    }
    let y_field = y_ring.ct(); // ct(y) to be checked by the verifier  :contentReference[oaicite:16]{index=16}

    // Build a, b (Fig. 4, lines 6–7)
    let a = build_a_digits(pp, &x_d);
    let mut b = build_b(pp, &x_d);

    // Scale b row by σ^{-1}(x̄) so Eq. (3) uses RHS = y_ring (no inverse for now).
    for bi in &mut b { *bi = sigma_inv_x.mul(bi, q); }

    // w, \hat w, v, c, z (Fig. 1 + Eq. (3))  :contentReference[oaicite:17]{index=17}
    let w = pr::compute_w(&a, &dec.s, q);
    let (what, v) = pr::derive_w_hat_and_v(&pp.commit, &pp.D, &w);

    let mut fs = Fs::new(b"greyhound/pcs-eval");
    fs.absorb_polyvec(&v).absorb_polyvec(&comm.0).absorb_u64(x_field as u64);
    let c = pr::sample_challenge(&fs, &pp.commit);
    let z = pr::compute_z(&dec.s, &c, q);

    // Build (P,h) with b' (scaled) and RHS = y_ring
    let proto = pr::ProtoParams { commit: &pp.commit, D: pp.D.clone() };
    let (P, h) = pr::build_linear_system(&proto, &a, &b, &comm.0, &v, &y_ring, &c);

    // Bring-up: reveal Z so the verifier can check PZ=h.
    // (In Step 6, replace by a succinct LaBRADOR proof of R1).  :contentReference[oaicite:18]{index=18}
    // We keep P,h implicit on the verifier side; they rebuild them identically.
    // Note: we don’t encode norms yet; that comes with LaBRADOR wiring.
    let proof = Proof { y_ring, v, what, that: dec.that.clone(), z };
    (y_field, proof)
}

/// ---- Eval.V (Fig. 4 Eval.V) ----  :contentReference[oaicite:19]{index=19}
pub fn eval_verify(
    pp: &PcsParams,
    comm: &Commitment,
    x_field: u32,
    y_field: u32,
    proof: &Proof,
) -> bool {
    // Check constant term
    if proof.y_ring.ct() != y_field { return false; } // Fig. 4, Eval.V line 7  :contentReference[oaicite:20]{index=20}

    let q = &pp.q;
    // Recompute a,b with σ^{-1}(x̄) scaling on b row (same as prover)
    let x_ring = embed_x(q, x_field);
    let x_d = pow_poly(x_ring.clone(), D, q);
    let sigma_inv_x = x_ring.sigma_inv(q);

    let a = build_a_digits(pp, &x_d);
    let mut b = build_b(pp, &x_d);
    for bi in &mut b { *bi = sigma_inv_x.mul(bi, q); }

    // Fiat–Shamir to get c (must absorb in the same order as prover)
    let mut fs = Fs::new(b"greyhound/pcs-eval");
    fs.absorb_polyvec(&proof.v).absorb_polyvec(&comm.0).absorb_u64(x_field as u64);
    let c = pr::sample_challenge(&fs, &pp.commit);

    // Rebuild (P,h)
    let proto = pr::ProtoParams { commit: &pp.commit, D: pp.D.clone() };
    let (P, h) = pr::build_linear_system(&proto, &a, &b, &comm.0, &proof.v, &proof.y_ring, &c);

    // Bring-up check: P * Z == h, with Z = [what || that || z]
    let mut Z: PolyVec = Vec::new();
    Z.extend_from_slice(&proof.what);
    Z.extend_from_slice(&proof.that);
    Z.extend_from_slice(&proof.z);

    P.mul_vec(&Z, &pp.q) == h
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng, Rng};

    #[test]
    fn pcs_single_eval_end_to_end() {
        // Toy q and params (q ≡ 5 mod 8 as in Sec. 5).  :contentReference[oaicite:21]{index=21}
        let q = ModQ { q: 229 };
        let N = 1 << 12; // small N for unit test
        let pp = setup_toy(N, q, 123);

        // Random polynomial f with deg < N
        let mut rng = StdRng::seed_from_u64(42);
        let mut f = vec![0u32; N];
        for i in 0..N { f[i] = rng.gen::<u32>() % q.q; }

        // Commit
        let (comm, dec) = commit(&pp, &f);

        // Pick x ∈ Z_q
        let x = 7u32;

        // Prove and verify f(x) = y
        let (y_field, prf) = eval_prove(&pp, &comm, x, &f, &dec);
        assert!(eval_verify(&pp, &comm, x, y_field, &prf));
    }
}

#[cfg(test)]
mod tests_hvzk {
    use super::*;
    use rand::{Rng, rngs::StdRng, SeedableRng};

    #[test]
    fn pcs_eval_hvzk_end_to_end_clear() {
        let q = ModQ { q: 229 };
        let N = 1<<12;
        let L = 4usize;
        let params = setup_hvzk_toy(N, q, 77, L, /*mu*/4, /*mu_v*/4);

        let mut rng = StdRng::seed_from_u64(2025);
        let mut f = vec![0u32; N];
        for i in 0..N { f[i] = rng.gen::<u32>() % q.q; }

        let x = 7u32;
        let (comm, prf) = eval_prove_hvzk_clear(&params, x, &f);
        assert!(eval_verify_hvzk_clear(&params, &comm, x, &prf));
    }
}
