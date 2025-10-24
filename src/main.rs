mod field;
mod mle;
mod smcheck;

// benchmark
use std::time::Instant;

// rand
use rand::rngs::OsRng;
use rand::Rng;

// ark
use ark_std::{Zero, One};
use ark_ff::UniformRand;
use ark_ff::Field;

// MyLib
use field::{Fq, Fq4, fq2fq4};
use mle::{mle_from_vec_fq4, mle_from_table, mle_from_table_fq4};
use smcheck::{fix_tau_eval_table, build_f_table, compute_a_eq_sum_i_prime_fq4, sumcheck_prove_from_table, sumcheck_prove_from_table_range, sumcheck_round_once, sumcheck_round_once_range};

pub const N: usize = 64;
pub const Q: u32 = 4294967197; // 231-227+1
pub const A: usize = 32; // 2^10


fn ringmul(a: Vec<Fq>, b: Vec<Fq>) -> (Vec<Fq>, Vec<Fq>) {
    let mut tmp: [Fq;2*N] = [Fq::zero();2*N];
    for i in 0..N{
        for j in 0..N{
            tmp[i+j] = tmp[i+j] + (a[i]*b[j]);
        }
    }
    let mut c: Vec<Fq> = Vec::with_capacity(N);
    let mut d: Vec<Fq> = Vec::with_capacity(N);
    for i in 0..N{
        c.push(tmp[i]-tmp[i+N]);
        d.push(tmp[i+N]);
    }
    (c, d)
}

fn ringadd(a: Vec<Fq>, b: Vec<Fq>) -> Vec<Fq> {
    let mut c: Vec<Fq> = Vec::with_capacity(N);
    for i in 0..N{
        c.push(a[i]+b[i]);
    }
    c
}

fn sample_fq_in_minus8_to_8<R: Rng + ?Sized>(rng: &mut R) -> Fq {
    let i: i32 = rng.gen_range(-8..=8);
    if i >= 0 {
        Fq::from(i as u64)
    } else {
        -Fq::from((-i) as u64)
    }
}

fn sample_vec_fq_range(n: usize) -> Vec<Fq> {
    let mut rng = OsRng;
    (0..n).map(|_| sample_fq_in_minus8_to_8(&mut rng)).collect()
}

fn rbeta_fq_beta8<Fq: Field + From<u64>>(z: Fq) -> Fq {
    let mut acc = z;
    for i in 1u64..=8 {
        let c = Fq::from(i);
        acc *= z - c;
        acc *= z + c;
    }
    acc
}

fn mle_eq_block_table<Fq: Field>(bits: usize, tau_block: &[Fq]) -> Vec<Fq> {
    debug_assert_eq!(tau_block.len(), bits);
    let size = 1usize << bits;
    let mut out = vec![Fq::one(); size];
    for idx in 0..size {
        let mut acc = Fq::one();
        let mut x = idx;
        for j in 0..bits {
            let rj = tau_block[j];
            let term = if (x & 1) == 1 { rj } else { Fq::one() - rj };
            acc *= term;
            x >>= 1;
        }
        out[idx] = acc;
    }
    out
}

pub fn build_f0_table_beta8<Fq: Field + From<u64>>(
    w_table: &[Fq],
    mk: usize,
    md: usize,
    tau0: &[Fq],
) -> Vec<Fq> {
    let rows_k = 1usize << mk;   // |u|
    let cols_d = 1usize << md;   // |l|
    assert_eq!(tau0.len(), mk + md, "tau0 length must be mk + md");
    assert_eq!(w_table.len(), rows_k * cols_d, "w_table size mismatch");

    let eq_u = mle_eq_block_table::<Fq>(mk, &tau0[..mk]);
    let eq_l = mle_eq_block_table::<Fq>(md, &tau0[mk..]);

    let mut out = vec![Fq::zero(); w_table.len()];

    for d in 0..cols_d {
        for k in 0..rows_k {
            let idx = k + (d << mk);                // 與你的索引一致
            let eq = eq_u[k] * eq_l[d];             // \tilde{e}_q(τ0,(u,ℓ))
            let range_poly = rbeta_fq_beta8::<Fq>(w_table[idx]); // w * Π_{i=1..8}(w±i)
            out[idx] = eq * range_poly;
        }
    }
    out
}


fn main() {
    let mut rng = OsRng;
    
    // init random mz = y (use 32 first)
    let mut m: Vec<Vec<Vec<Fq>>> = Vec::with_capacity(A);
    let mut z: Vec<Vec<Fq>> = Vec::with_capacity(A);
    let mut y: Vec<Vec<Fq>> = Vec::with_capacity(A);
    let mut r: Vec<Vec<Fq>> = Vec::with_capacity(A);

    // m
    for _ in 0..A {
        let mut mat = Vec::with_capacity(A);
        for _ in 0..A {
            let row: Vec<Fq> = (0..N).map(|_| Fq::rand(&mut rng)).collect();
            mat.push(row);
        }
        m.push(mat);
    }

    // z
    for _ in 0..A {
        let row = sample_vec_fq_range(N);
        z.push(row);
    }

    // y
    for i in 0..A {
        let mut tmp1: Vec<Fq> = vec![Fq::zero(); N];
        let mut tmp2: Vec<Fq> = vec![Fq::zero(); N];
        for j in 0..A{
            let ta: Vec<Fq>;
            let tb: Vec<Fq>;
            (ta, tb) = ringmul(m[i][j].clone(), z[j].clone());
            tmp1 = ringadd(tmp1, ta);
            tmp2 = ringadd(tmp2, tb);
        }
        y.push(tmp1);
        r.push(tmp2);
    }

    // 7. verifier randomly choose alpha in Fq4
    let alpha: Fq4 = Fq4::rand(&mut rng);

    // 8. MLE of w, alpha, M (in Fq)
    // w: aggregate x polylen Fq
    let mut w: Vec<Vec<Fq>> = Vec::with_capacity(2*A);
    let mut w_range: Vec<Vec<Fq>> = Vec::with_capacity(A);
    for i in 0..A {
        w.push(z[i].clone());
        w_range.push(z[i].clone());
    }
    for i in 0..A {
        w.push(r[i].clone());
    }
    
    let mle_w = mle_from_table(&w);
    let mle_w_range = mle_from_table(&w_range);
    println!("number of variables (w): {}", mle_w.num_vars);
    println!("number of variables (w_range): {}", mle_w_range.num_vars);

    // alpha^64-1 in Fq4
    let mut tmp_alpha = alpha.clone();
    for _ in 0..6{
        tmp_alpha = tmp_alpha.square();
    }
    tmp_alpha = tmp_alpha + Fq4::from(1u64);
    let alpha64 = -tmp_alpha;

    let mut vec_alpha: Vec<Fq4> = Vec::with_capacity(N);
    vec_alpha.push(Fq4::one());
    for i in 0..63{
        let tmp_alpha = vec_alpha[i].clone();
        vec_alpha.push(tmp_alpha * alpha.clone());
    }
    let mle_alpha = mle_from_vec_fq4(&vec_alpha);
    println!("number of variables (alpha): {}", mle_alpha.num_vars);

    
    // M: Constraints x 3*Constraints in Fq16
    let mut m_alpha: Vec<Vec<Fq4>> = Vec::with_capacity(A);
    for i in 0..A{
        let mut tmp_m1: Vec<Fq4> = Vec::with_capacity(A);
        for j in 0..A{
            let mut tmp_m = Fq4::zero();
            for k in 0..N{
                tmp_m = tmp_m * alpha.clone();
                tmp_m += fq2fq4(m[i][j][63-k]);
            }
            tmp_m1.push(tmp_m);
        }
        m_alpha.push(tmp_m1);
    }

    let mut mm: Vec<Vec<Fq4>> = Vec::with_capacity(A);
    for i in 0..A{
        let mut tmp_m: Vec<Fq4> = Vec::with_capacity(2*A);
        for j in 0..2*A{
            if j < A{
                tmp_m.push(m_alpha[i][j]);
            }
            else if i+A == j {
                tmp_m.push(alpha64.clone());
            }else{
                tmp_m.push(Fq4::zero());
            }
        }
        mm.push(tmp_m);
    }
    let mle_m = mle_from_table_fq4(&mm);
    println!("number of variables (m): {}", mle_m.num_vars);

    // 9. randomly pick tau 
    let tau: [Fq4; 5] = std::array::from_fn(|_| Fq4::rand(&mut rng));


    // // 10. PCS commit w = [s2, r, s1]

    
    // 11. sumcheck protocol: constraint
    // w table: 15 vars
    let w_table: Vec<Fq> = mle_w.evaluations.clone();
    // a table: 9 vars
    let alpha_table: Vec<Fq4> = mle_alpha.evaluations.clone();
    // M table: 6 vars
    let m_table: Vec<Fq4> = fix_tau_eval_table(&mle_m, &tau);


    // prover_time1 += start.elapsed();
    // println!("Prover1 : {:?}", prover_time1);


    // F table: 15 vars
    let f_table = build_f_table(&w_table, &alpha_table, &m_table, 6, 6);
    // LHS of proof equation
    let a = compute_a_eq_sum_i_prime_fq4(&y, alpha, tau);
    // Verifier random challenge for 15 rounds
    let challenge:[Fq4; 12] = std::array::from_fn(|_| Fq4::rand(&mut rng));
    // proof
    let proof = sumcheck_prove_from_table(f_table.clone(), &challenge);

    // prover_time2 += start.elapsed();
    // println!("Prover2 : {:?}", prover_time2);
    // start = Instant::now();
    
    // Verifier
    let mut layer = f_table;
    for (round, (&(c0,c1), &r)) in proof.g_coeffs.iter().zip(challenge.iter()).enumerate() {
        // g_t(0) + g_t(1) = (c0) + (c0+c1) = 2*c0 + c1
        let lhs_round_sum = layer.iter().copied().fold(Fq4::zero(), |acc,x| acc + x);
        let rhs_round_sum = c0 + (c0 + c1);
        if round == 0{
            assert_eq!(lhs_round_sum, a, "init sum check failed");
        }
        assert_eq!(lhs_round_sum, rhs_round_sum, "round {} sum check failed", round);

        // verifier_time += start.elapsed();
        // start = Instant::now();

        let (_, next) = sumcheck_round_once(&layer, r);

        // prover_time3 += start.elapsed();
        // start = Instant::now();

        let g_at_r = c0 + c1 * r;
        let next_sum = next.iter().copied().fold(Fq4::zero(), |acc,x| acc + x);
        assert_eq!(g_at_r, next_sum, "round {} eval check failed", round);
        layer = next;
    }

    assert_eq!(layer[0], proof.final_eval);

    // verifier_time += start.elapsed();
    // println!("Prover 3: {:?}", prover_time3);
    // println!("Verifier: {:?}", verifier_time);

    // // 12. sumcheck protocol: norm bound
    let tau1: [Fq; 11] = std::array::from_fn(|_| Fq::rand(&mut rng));
    let range_tbl: Vec<Fq> = mle_w_range.evaluations.clone();
    let range_table = build_f0_table_beta8(&range_tbl, 5, 6, &tau1);

    // Verifier random challenge for 15 rounds
    let challenge_range:[Fq; 11] = std::array::from_fn(|_| Fq::rand(&mut rng));
    // proof
    let proof_range = sumcheck_prove_from_table_range(range_table.clone(), &challenge_range);

    let mut layer_range = range_table;
    for (round, (&(c0,c1), &r)) in proof_range.g_coeffs.iter().zip(challenge_range.iter()).enumerate() {
        // g_t(0) + g_t(1) = (c0) + (c0+c1) = 2*c0 + c1
        let lhs_round_sum = layer_range.iter().copied().fold(Fq::zero(), |acc,x| acc + x);
        let rhs_round_sum = c0 + (c0 + c1);
        if round == 0{
            assert_eq!(lhs_round_sum, Fq::zero(), "range: init sum check failed");
        }
        assert_eq!(lhs_round_sum, rhs_round_sum, "range: round {} sum check failed", round);

        // verifier_time += start.elapsed();
        // start = Instant::now();

        let (_, next) = sumcheck_round_once_range(&layer_range, r);

        // prover_time3 += start.elapsed();
        // start = Instant::now();

        let g_at_r = c0 + c1 * r;
        let next_sum = next.iter().copied().fold(Fq::zero(), |acc,x| acc + x);
        assert_eq!(g_at_r, next_sum, "range: round {} eval check failed", round);
        layer_range = next;
    }

    assert_eq!(layer_range[0], proof_range.final_eval);
    


    // // 13. PCS open
    
}