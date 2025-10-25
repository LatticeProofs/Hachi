//! Fiat–Shamir transcript (SHAKE128) and lattice-style challenge sampler.
//! Matches the paper’s use of SHAKE128 and the ±1/±2 challenge pattern (Sec. 5).  :contentReference[oaicite:4]{index=4}

use sha3::{Shake128, digest::{Update, ExtendableOutput, XofReader}};
use greyhound_ring::{ModQ, Poly, D};

pub struct Fs {
    st: Shake128
}

impl Fs {
    pub fn new(domain: &[u8]) -> Self {
        let mut st = Shake128::default();
        st.update(b"greyhound/fs/");
        st.update(domain);
        Self { st }
    }

    #[inline] pub fn absorb_bytes(&mut self, bytes: &[u8]) -> &mut Self { self.st.update(bytes); self }
    #[inline] pub fn absorb_u64(&mut self, x: u64) -> &mut Self { self.st.update(&x.to_le_bytes()); self }
    pub fn absorb_poly(&mut self, p: &Poly) -> &mut Self {
        let mut buf = [0u8; 4*D];
        for i in 0..D { buf[4*i..4*i+4].copy_from_slice(&p.c[i].to_le_bytes()); }
        self.st.update(&buf);
        self
    }
    pub fn absorb_polyvec(&mut self, v: &[Poly]) -> &mut Self {
        for p in v { self.absorb_poly(p); }
        self
    }

    fn reader(&self) -> Box<dyn XofReader> {
        let mut st = self.st.clone();
        Box::new(st.finalize_xof())
    }


    /// Draw a single ring challenge with τ1 entries in {±1} and τ2 entries in {±2}.
    fn sample_challenge_poly(reader: &mut dyn XofReader, q: &ModQ, tau1: usize, tau2: usize) -> Poly {
        debug_assert!(tau1 + tau2 <= D);
        let mut coeffs = [0u32; D];
        // Pick distinct positions
        let mut chosen = [false; D];
        let mut take_pos = |reader: &mut dyn XofReader| -> usize {
            loop {
                let mut b = [0u8; 2];
                reader.read(&mut b);
                let idx = (u16::from_le_bytes(b) as usize) % D;
                if !chosen[idx] { chosen[idx]=true; return idx; }
            }
        };
        // Collect positions
        let mut pos = Vec::with_capacity(tau1+tau2);
        for _ in 0..(tau1+tau2) { pos.push(take_pos(reader)); }
        // Shuffle order
        for i in (1..pos.len()).rev() {
            let mut b = [0u8; 2];
            reader.read(&mut b);
            let j = (u16::from_le_bytes(b) as usize) % (i+1);
            pos.swap(i, j);
        }
        // Assign signs/amplitudes
        let mut sign_bit = |reader: &mut dyn XofReader| -> i32 {
            let mut b = [0u8; 1]; reader.read(&mut b); (b[0] & 1) as i32
        };
        // first tau2 → ±2, rest → ±1
        for (k,&idx) in pos.iter().enumerate() {
            let amp = if k < tau2 { 2i32 } else { 1i32 };
            let s = if sign_bit(reader)==1 { -amp } else { amp };
            let x = if s>=0 { s as u32 } else { q.neg((-s) as u32) };
            coeffs[idx] = x % q.q;
        }
        Poly { c: coeffs }
    }

    /// Deterministic C^r sampler (C = { c : ||c||_1 <= κ }), instantiated with (τ1,τ2).
    /// Paper’s concrete choice: τ1=32, τ2=8 for d=64 (Sec. 5).  :contentReference[oaicite:5]{index=5}
    pub fn challenge_vec(&self, r: usize, q: &ModQ, tau1: usize, tau2: usize) -> Vec<Poly> {
        let mut rdr = self.reader();
        (0..r)
            .map(|_| Self::sample_challenge_poly(&mut *rdr, q, tau1, tau2))
            .collect()
    }

    pub fn alphas(&self, L: usize, q: &ModQ) -> Vec<u32> {
        let mut rdr = self.reader();
        let mut out = Vec::with_capacity(L);
        for _ in 0..L {
            let mut b = [0u8; 8];
            rdr.read(&mut b);
            out.push((u64::from_le_bytes(b) % (q.q as u64)) as u32);
        }
        out
    }
}
