use crate::math::fields::*;
use crate::utils::ds::*;
//move
use crate::Q;

pub type MBundle = (
    Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, 
    Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, 
    Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, 
    Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>
);
/// `WBundle::4` (the `t` slot) is borrowed so the prover can feed in a slice
/// from a memory-mapped commitment without an extra copy. The other slots are
/// owned aligned buffers built inside the prover.
pub type WBundle<'a> = (
    AlignedU32Vec, AlignedU32Vec, AlignedU32Vec, AlignedU32Vec,
    &'a [u32],     AlignedI8Vec,  AlignedU32Vec, AlignedU32Vec,
    AlignedU32Vec, AlignedU32Vec
);

fn mle_M(i: u32, u: u32, m: &MBundle, params: &crate::SetupParams) -> [u32; 4] {
    let zero = [0u32, 0u32, 0u32, 0u32];
    let u_usize = u as usize;
    let k_offset = 9 << 14;

    let get_neg_g = |u_idx: u32| -> [u32; 4] {
        let g_idx = (u_idx % 16) as usize;
        let g_val = params.g_matrix_2[g_idx];
        let neg_g = if g_val == 0 { 0 } else { 4294967197u32 - g_val };
        [neg_g, 0, 0, 0]
    };
    
    match i {
        // Row 0~3: D * w_i - G * k_i = 0
        0 => {
            if u < (1<<14) {
                m.0[4*u_usize..4*(u_usize+1)].try_into().unwrap()
            } else if u >= k_offset && u < k_offset + 16 {
                get_neg_g(u)
            } else { zero }
        }
        1 => {
            if u >= (1<<14) && u < (2<<14) {
                let idx = u_usize - (1<<14);
                m.0[4*idx..4*(idx+1)].try_into().unwrap()
            } else if u >= k_offset + 16 && u < k_offset + 32 {
                get_neg_g(u)
            } else { zero }
        }
        2 => {
            if u >= (2<<14) && u < (3<<14) {
                let idx = u_usize - (2<<14);
                m.0[4*idx..4*(idx+1)].try_into().unwrap()
            } else if u >= k_offset + 32 && u < k_offset + 48 {
                get_neg_g(u)
            } else { zero }
        }
        3 => {
            if u >= (3<<14) && u < (4<<14) {
                let idx = u_usize - (3<<14);
                m.0[4*idx..4*(idx+1)].try_into().unwrap()
            } else if u >= k_offset + 48 && u < k_offset + 64 {
                get_neg_g(u)
            } else { zero }
        }
        // Row 4: E0*k0 + E1*k1 + E2*k2 + E3*k3 = v
        4 => {
            if u >= k_offset && u < k_offset + 16 {
                let idx = u_usize - (k_offset as usize);
                m.2[4*idx..4*(idx+1)].try_into().unwrap()
            } else if u >= k_offset + 16 && u < k_offset + 32 {
                let idx = u_usize - ((k_offset + 16) as usize);
                m.3[4*idx..4*(idx+1)].try_into().unwrap()
            } else if u >= k_offset + 32 && u < k_offset + 48 {
                let idx = u_usize - ((k_offset + 32) as usize);
                m.4[4*idx..4*(idx+1)].try_into().unwrap()
            } else if u >= k_offset + 48 && u < k_offset + 64 {
                let idx = u_usize - ((k_offset + 48) as usize);
                m.5[4*idx..4*(idx+1)].try_into().unwrap()
            } else { zero }
        }
        // Row 5: D * t = u
        5 => {
            if u >= (4<<14) && u < (5<<14) {
                let idx = u_usize - (4<<14);
                m.0[4*idx..4*(idx+1)].try_into().unwrap()
            } else { zero }
        }
        // Row 6~9: b^T G * w_i
        6 => {
            if u < (1<<14) { m.10[4*u_usize..4*(u_usize+1)].try_into().unwrap() }
            else if u >= (1<<14) && u < (2<<14) { extdbl(m.13[4*(u_usize-(1<<14))..4*(u_usize-(1<<14)+1)].try_into().unwrap()) }
            else if u >= (2<<14) && u < (3<<14) { extdbl(m.12[4*(u_usize-(2<<14))..4*(u_usize-(2<<14)+1)].try_into().unwrap()) }
            else if u >= (3<<14) && u < (4<<14) { extdbl(m.11[4*(u_usize-(3<<14))..4*(u_usize-(3<<14)+1)].try_into().unwrap()) }
            else { zero }
        }
        7 => {
            if u < (1<<14) { m.11[4*u_usize..4*(u_usize+1)].try_into().unwrap() }
            else if u >= (1<<14) && u < (2<<14) { m.10[4*(u_usize-(1<<14))..4*(u_usize-(1<<14)+1)].try_into().unwrap() }
            else if u >= (2<<14) && u < (3<<14) { extdbl(m.13[4*(u_usize-(2<<14))..4*(u_usize-(2<<14)+1)].try_into().unwrap()) }
            else if u >= (3<<14) && u < (4<<14) { extdbl(m.12[4*(u_usize-(3<<14))..4*(u_usize-(3<<14)+1)].try_into().unwrap()) }
            else { zero }
        }
        8 => {
            if u < (1<<14) { m.12[4*u_usize..4*(u_usize+1)].try_into().unwrap() }
            else if u >= (1<<14) && u < (2<<14) { m.11[4*(u_usize-(1<<14))..4*(u_usize-(1<<14)+1)].try_into().unwrap() }
            else if u >= (2<<14) && u < (3<<14) { m.10[4*(u_usize-(2<<14))..4*(u_usize-(2<<14)+1)].try_into().unwrap() }
            else if u >= (3<<14) && u < (4<<14) { extdbl(m.13[4*(u_usize-(3<<14))..4*(u_usize-(3<<14)+1)].try_into().unwrap()) }
            else { zero }
        }
        9 => {
            if u < (1<<14) { m.13[4*u_usize..4*(u_usize+1)].try_into().unwrap() }
            else if u >= (1<<14) && u < (2<<14) { m.12[4*(u_usize-(1<<14))..4*(u_usize-(1<<14)+1)].try_into().unwrap() }
            else if u >= (2<<14) && u < (3<<14) { m.11[4*(u_usize-(2<<14))..4*(u_usize-(2<<14)+1)].try_into().unwrap() }
            else if u >= (3<<14) && u < (4<<14) { m.10[4*(u_usize-(3<<14))..4*(u_usize-(3<<14)+1)].try_into().unwrap() }
            else { zero }
        }
        // Row 10~13: (c^T G_1) w_i - a_i^T G J * z
        10 => {
            if u < (1<<14) { m.14[4*u_usize..4*(u_usize+1)].try_into().unwrap() }
            else if u >= (5<<14) && u < (9<<14) { extneg(m.6[4*(u_usize-(5<<14))..4*(u_usize-(5<<14)+1)].try_into().unwrap()) }
            else { zero }
        }
        11 => {
            if u >= (1<<14) && u < (2<<14) { m.14[4*(u_usize-(1<<14))..4*(u_usize-(1<<14)+1)].try_into().unwrap() }
            else if u >= (5<<14) && u < (9<<14) { extneg(m.7[4*(u_usize-(5<<14))..4*(u_usize-(5<<14)+1)].try_into().unwrap()) }
            else { zero }
        }
        12 => {
            if u >= (2<<14) && u < (3<<14) { m.14[4*(u_usize-(2<<14))..4*(u_usize-(2<<14)+1)].try_into().unwrap() }
            else if u >= (5<<14) && u < (9<<14) { extneg(m.8[4*(u_usize-(5<<14))..4*(u_usize-(5<<14)+1)].try_into().unwrap()) }
            else { zero }
        }
        13 => {
            if u >= (3<<14) && u < (4<<14) { m.14[4*(u_usize-(3<<14))..4*(u_usize-(3<<14)+1)].try_into().unwrap() }
            else if u >= (5<<14) && u < (9<<14) { extneg(m.9[4*(u_usize-(5<<14))..4*(u_usize-(5<<14)+1)].try_into().unwrap()) }
            else { zero }
        }
        // Row 14: (c^T G_1) t - D J * z
        14 => {
            if u >= (4<<14) && u < (5<<14) { m.14[4*(u_usize-(4<<14))..4*(u_usize-(4<<14)+1)].try_into().unwrap() }
            else if u >= (5<<14) && u < (9<<14) { m.1[4*(u_usize-(5<<14))..4*(u_usize-(5<<14)+1)].try_into().unwrap() }
            else { zero }
        }
        _ => zero,
    }
}

pub fn mle_m(i: u32, u: u32, m: &MBundle, params: &crate::SetupParams) -> [u32; 4] {
    let num_witness_rings = (9 << 14) + 64; // 147520
    let constraints_u32 = params.constraints as u32;
    if i < constraints_u32 && u < num_witness_rings {
        mle_M(i, u, m, params)
    } 
    // -(alpha^1024 + 1)
    else if u >= num_witness_rings && u < num_witness_rings + constraints_u32 && i as usize == (u - num_witness_rings) as usize { 
        m.15.clone().try_into().unwrap()
    } else { 
        [0u32, 0u32, 0u32, 0u32]
    }
}

fn mle_W(u: u32, l: u32, w: &WBundle<'_>) -> u32 {
    let u_usize = u as usize;
    let l_usize = l as usize;
    let k_offset = 9 << 14;
    let k_offset_usize = (9 << 14) as usize;

    if u < (1<<14) { w.0[u_usize * 1024 + l_usize] }
    else if u < (2<<14) { w.1[(u_usize - (1<<14)) * 1024 + l_usize] }
    else if u < (3<<14) { w.2[(u_usize - (2<<14)) * 1024 + l_usize] }
    else if u < (4<<14) { w.3[(u_usize - (3<<14)) * 1024 + l_usize] }
    else if u < (5<<14) { w.4[(u_usize - (4<<14)) * 1024 + l_usize] } // t
    else if u < k_offset {
        // z
        let val = w.5[(u_usize - (5<<14)) * 1024 + l_usize]; 
        if val < 0 { 4294967197u32 - ((-val) as u32) } else { val as u32 }
    } 
    else if u < k_offset + 16 { w.6[(u_usize - k_offset_usize) * 1024 + l_usize] }
    else if u < k_offset + 32 { w.7[(u_usize - (k_offset_usize + 16)) * 1024 + l_usize] }
    else if u < k_offset + 48 { w.8[(u_usize - (k_offset_usize + 32)) * 1024 + l_usize] }
    else if u < k_offset + 64 { w.9[(u_usize - (k_offset_usize + 48)) * 1024 + l_usize] }
    else { 0 }
}

pub fn mle_w(u: u32, l: u32, w: &WBundle<'_>, r: &[u32]) -> u32 {
    let num_witness_rings = (9 << 14) + 64; // 147520
    if u < num_witness_rings {
        mle_W(u, l, w)
    } else if u < num_witness_rings + 15 { 
        r[(u as usize - num_witness_rings as usize) * 1024 + l as usize]
    } else {
        0
    }
}

pub fn mle_alpha(y: u32, alpha_vec: &[u32]) -> u32{
    alpha_vec[y as usize]
}