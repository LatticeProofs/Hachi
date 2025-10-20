use ark_ff::{Fp, MontBackend, MontConfig};
use ark_ff::MontFp;
use ark_ff::fields::{QuadExtField, QuadExtConfig};

#[derive(MontConfig)]
#[modulus = "4294967197"] // 2^32-99
#[generator = "6"]
pub struct FqConfig;
pub type Fq = Fp<MontBackend<FqConfig, 1>, 1>;


pub type Fq2 = QuadExtField<Fq2Config>;
pub struct Fq2Config;
impl QuadExtConfig for Fq2Config {
    type BaseField = Fq;
    type BasePrimeField = Fq;
    type FrobCoeff = Fq;

    const DEGREE_OVER_BASE_PRIME_FIELD: usize = 2;
    const NONRESIDUE: Fq = MontFp!("6");
    const FROBENIUS_COEFF_C1: &'static [Self::FrobCoeff] = &[
        MontFp!("1"),
        MontFp!("4294967196"),
    ];

    #[inline(always)]
    fn mul_base_field_by_frob_coeff(fe: &mut Self::BaseField, power: usize) {
        *fe *= Self::FROBENIUS_COEFF_C1[power % 2];
    }
}

pub type Fq4 = QuadExtField<Fq4Config>;
pub struct Fq4Config;

impl QuadExtConfig for Fq4Config {
    type BaseField = Fq2;
    type BasePrimeField = Fq;
    type FrobCoeff = Fq2;

    const DEGREE_OVER_BASE_PRIME_FIELD: usize = 4;
    const NONRESIDUE: Fq2 = Fq2::new(MontFp!("0"), MontFp!("1"));
    const FROBENIUS_COEFF_C1: &'static [Self::FrobCoeff] = &[
        Fq2::new(MontFp!("1"), MontFp!("0")),
        Fq2::new(MontFp!("983270775"), MontFp!("0")),
    ];

    #[inline(always)]
    fn mul_base_field_by_frob_coeff(fe: &mut Self::BaseField, power: usize) {
        *fe *= Self::FROBENIUS_COEFF_C1[power % 2];
    }
}

pub fn fq2fq4(a: Fq) -> Fq4 {
    Fq4::new(
        Fq2::new(a, MontFp!("0")),
        Fq2::new(MontFp!("0"), MontFp!("0")),
    )
}