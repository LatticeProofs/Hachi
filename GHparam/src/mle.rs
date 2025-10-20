use crate::field::{Fq, Fq4};
use ark_poly::DenseMultilinearExtension;
use ark_std::Zero;

fn log2_pow2(x: usize) -> usize {
    debug_assert!(x.is_power_of_two());
    x.trailing_zeros() as usize
}

pub fn mle_from_vec_fq4(vals: &Vec<Fq4>) -> DenseMultilinearExtension<Fq4> {
    assert!(!vals.is_empty(), "empty vector");

    let m = vals.len();
    let m_p2 = m.next_power_of_two();
    let n = m_p2.trailing_zeros() as usize;

    let mut evals = vec![Fq4::zero(); m_p2];
    evals[..m].copy_from_slice(vals);

    DenseMultilinearExtension::from_evaluations_vec(n, evals)
}

pub fn mle_from_table(table: &Vec<Vec<Fq>>) -> DenseMultilinearExtension<Fq> {
    let rows = table.len();
    assert!(rows > 0, "empty table");
    let cols = table[0].len();
    assert!(table.iter().all(|row| row.len() == cols), "ragged table");

    let rows_p2 = rows.next_power_of_two();
    let cols_p2 = cols.next_power_of_two();

    let r = log2_pow2(rows_p2);
    let c = log2_pow2(cols_p2);
    let n = r + c;

    let mut evals = vec![Fq::zero(); rows_p2 * cols_p2];

    for i in 0..rows {
        for j in 0..cols {
            let idx = i + (j << r);
            evals[idx] = table[i][j];
        }
    }
    DenseMultilinearExtension::from_evaluations_vec(n, evals)
}

pub fn mle_from_table_fq4(table: &Vec<Vec<Fq4>>) -> DenseMultilinearExtension<Fq4> {
    let rows = table.len();
    assert!(rows > 0, "empty table");
    let cols = table[0].len();
    assert!(table.iter().all(|row| row.len() == cols), "ragged table");

    let rows_p2 = rows.next_power_of_two();
    let cols_p2 = cols.next_power_of_two();

    let r = log2_pow2(rows_p2);
    let c = log2_pow2(cols_p2);
    let n = r + c;

    let mut evals = vec![Fq4::zero(); rows_p2 * cols_p2];

    for i in 0..rows {
        for j in 0..cols {
            let idx = i + (j << r);
            evals[idx] = table[i][j];
        }
    }
    DenseMultilinearExtension::from_evaluations_vec(n, evals)
}