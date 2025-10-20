use crate::field::{Fq, Fq4, fq2fq4};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension as _};
use ark_std::{Zero, One};

pub fn fix_tau_eval_table(
    mle_m: &DenseMultilinearExtension<Fq4>,
    tau4: &[Fq4], // len == 4
) -> Vec<Fq4> {
    // assert!(tau4.len() == 4, "tau must fix the first 4 variables");
    // assert!(mle_m.num_vars >= 4, "mle has fewer than 4 variables");
    let fixed = mle_m.fix_variables(tau4);
    fixed.evaluations.clone()
}

pub fn build_f_table(w_table: &[Fq], alpha_table: &[Fq4], m_k_table: &[Fq4], mk: usize, md: usize) -> Vec<Fq4> {
    let rows_k = 1usize << mk;
    let cols_d = 1usize << md;
    let mut out = vec![Fq4::zero(); rows_k * cols_d];
    
    for d in 0..cols_d {
        let a_d = alpha_table[d];
        for k in 0..rows_k {
            let idx = k + (d << mk);
            out[idx] = fq2fq4(w_table[idx]) * (a_d * m_k_table[k]);
        }
    }
    out
}


fn sum_even_odd(v: &[Fq4]) -> (Fq4, Fq4) {
    let mut s0 = Fq4::zero();
    let mut s1 = Fq4::zero();
    for (i, x) in v.iter().enumerate() {
        if (i & 1) == 0 { s0 += x; } else { s1 += x; }
    }
    (s0, s1)
}

fn sum_even_odd_range(v: &[Fq]) -> (Fq, Fq) {
    let mut s0 = Fq::zero();
    let mut s1 = Fq::zero();
    for (i, x) in v.iter().enumerate() {
        if (i & 1) == 0 { s0 += x; } else { s1 += x; }
    }
    (s0, s1)
}

pub fn sumcheck_round_once(layer: &[Fq4], r: Fq4) -> ((Fq4, Fq4), Vec<Fq4>) {
    let (s0, s1) = sum_even_odd(layer);
    let g_c0 = s0;
    let g_c1 = s1 - s0;

    let one_minus_r = Fq4::one() - r;
    let mut next = Vec::with_capacity(layer.len() / 2);
    let mut i = 0;
    while i < layer.len() {
        let a = layer[i];
        let b = layer[i + 1];
        next.push(a * one_minus_r + b * r);
        i += 2;
    }
    ((g_c0, g_c1), next)
}

pub fn sumcheck_round_once_range(layer: &[Fq], r: Fq) -> ((Fq, Fq), Vec<Fq>) {
    let (s0, s1) = sum_even_odd_range(layer);
    let g_c0 = s0;
    let g_c1 = s1 - s0;

    let one_minus_r = Fq::one() - r;
    let mut next = Vec::with_capacity(layer.len() / 2);
    let mut i = 0;
    while i < layer.len() {
        let a = layer[i];
        let b = layer[i + 1];
        next.push(a * one_minus_r + b * r);
        i += 2;
    }
    ((g_c0, g_c1), next)
}


pub struct SumcheckProof<Fq> {
    pub g_coeffs: Vec<(Fq, Fq)>,
    pub final_eval: Fq,
}

pub fn sumcheck_prove_from_table(
    mut layer: Vec<Fq4>, 
    rs: &[Fq4], 
) -> SumcheckProof<Fq4> {
    let mut coeffs = Vec::with_capacity(rs.len());
    for &r in rs {
        let (gc, next) = sumcheck_round_once(&layer, r);
        coeffs.push(gc);
        layer = next;
    }
    SumcheckProof { g_coeffs: coeffs, final_eval: layer[0] }
}

pub fn sumcheck_prove_from_table_range(
    mut layer: Vec<Fq>, 
    rs: &[Fq], 
) -> SumcheckProof<Fq> {
    let mut coeffs = Vec::with_capacity(rs.len());
    for &r in rs {
        let (gc, next) = sumcheck_round_once_range(&layer, r);
        coeffs.push(gc);
        layer = next;
    }
    SumcheckProof { g_coeffs: coeffs, final_eval: layer[0] }
}

// ---------------------------------------

fn eval_poly_low_to_high_at_alpha_fq4(coeffs_low_to_high: &[Fq], alpha: Fq4) -> Fq4 {
    let mut acc = Fq4::zero();
    for &c in coeffs_low_to_high.iter().rev() {
        acc = acc * alpha + fq2fq4(c);
    }
    acc
}


fn eq_weight_4(x: &[Fq4; 5], j: usize) -> Fq4 {
    let mut w = Fq4::one();
    for b in 0..5 {
        let bit = (j >> b) & 1;
        let term = if bit == 1 { x[b] } else { Fq4::one() - x[b] };
        w *= term;
    }
    w
}

pub fn compute_a_eq_sum_i_prime_fq4(
    ts: &[Vec<Fq>],
    alpha: Fq4, 
    i_prime: [Fq4; 5],
) -> Fq4 {
    // assert!(ts.len() == 16, "ts must contain 16 polynomials for 4-bit j");
    let mut t_alpha = [Fq4::zero(); 32];
    for j in 0..32 {
        t_alpha[j] = eval_poly_low_to_high_at_alpha_fq4(&ts[j], alpha);
    }

    let mut a = Fq4::zero();
    for j in 0..32 {
        let wj = eq_weight_4(&i_prime, j);
        a += wj * t_alpha[j];
    }
    a
}