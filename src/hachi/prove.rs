use crate::utils::fs::FastTranscript;
use crate::hachi::commit::Commitment;
use crate::hachi::setup::SetupParams;
use crate::utils::ds::AlignedU8Vec;
use crate::utils::random::generate_fs_eval_points_q;
use crate::math::fields::*;
use crate::math::field_simd::*;
use crate::math::rns::*;
use crate::mlp::mle::*;
use crate::utils::ds::AlignedU32Vec;
use crate::utils::ds::AlignedI8Vec;
use crate::utils::ds::Align64;
use crate::utils::random::{generate_sparse_c_idx, generate_random_q_element};
use std::time::Instant;
use std::arch::x86_64::*;
use crate::sumcheck::prove::sumcheck_prove;
use crate::math::eq::{build_eq_tbl, build_eq_tbl_split};
use std::slice;
use crate::prep::foldwitness::*;
use crate::hachi::verify::Challenge;

#[derive(Debug, Clone)]
pub struct SumcheckProof {
    // X 
    pub p_alg_x: Vec<[[u32; 4]; 3]>,
    pub p_norm_x: Vec<[[u32; 4]; 7]>,
    pub r_x: Vec<[u32; 4]>,
    // Y 
    pub p_alg_y: Vec<[[u32; 4]; 3]>,
    pub p_norm_y: Vec<[[u32; 4]; 7]>,
    pub r_y: Vec<[u32; 4]>,
    // Final Evaluation
    pub final_w: [u32; 4]
}

pub unsafe fn prove(
    params: &SetupParams,
    s: &AlignedU8Vec,
    commitment: Commitment,
    challenge: &Challenge
) -> (SumcheckProof, [u32; 4]) {

    let start = Instant::now();

    let a0 = &challenge.a0;
    let a1 = &challenge.a1;
    let a2 = &challenge.a2;
    let a3 = &challenge.a3;
    let b0 = &challenge.b0;
    let b1 = &challenge.b1;
    let b2 = &challenge.b2;
    let b3 = &challenge.b3;
    let c = &challenge.c;
    let alpha = &challenge.alpha;
    let tau_0 = &challenge.tau_0;
    let tau_1 = &challenge.tau_1;

    // init r
    let mut r = AlignedU32Vec{inner: vec![Align64([0u32; 16]); (params.constraints * params.n) / 16], len: params.constraints * params.n };
    r[5*params.n .. 6*params.n].copy_from_slice(commitment.r());

    // init w, z, k
    let total_len = params.height_2 * params.n;
    let total_len_4 = params.height_4 * params.n;
    let mut w0 = AlignedU32Vec { inner: vec![Align64([0u32; 16]); total_len / 16], len: total_len };
    let mut w1 = AlignedU32Vec { inner: vec![Align64([0u32; 16]); total_len / 16], len: total_len };
    let mut w2 = AlignedU32Vec { inner: vec![Align64([0u32; 16]); total_len / 16], len: total_len };
    let mut w3 = AlignedU32Vec { inner: vec![Align64([0u32; 16]); total_len / 16], len: total_len };
    let mut long_z = AlignedU32Vec { inner: vec![Align64([0u32; 16]); total_len_4 / 16], len: total_len_4 };
    let mut z = AlignedI8Vec{ inner: vec![Align64([0i8; 64]); (total_len_4 * 8) / 64], len: total_len_4 * 8 };
    let mut k0 = AlignedU32Vec { inner: vec![Align64([0u32; 16]); params.n], len: params.n<<4 };
    let mut k1 = AlignedU32Vec { inner: vec![Align64([0u32; 16]); params.n], len: params.n<<4 };
    let mut k2 = AlignedU32Vec { inner: vec![Align64([0u32; 16]); params.n], len: params.n<<4 };
    let mut k3 = AlignedU32Vec { inner: vec![Align64([0u32; 16]); params.n], len: params.n<<4 };
    
    // init v, u
    let mut v = AlignedU32Vec { inner: vec![Align64([0u32; 16]); params.n / 16], len: params.n };
    let mut u0 = vec![0u32; params.n];
    let mut u1 = vec![0u32; params.n];
    let mut u2 = vec![0u32; params.n];
    let mut u3 = vec![0u32; params.n];
    
    // let duration = start.elapsed();
    // println!("Init y, z: {:?}", duration);
    // let start = Instant::now();

    // Compute w
    let q_u32 = 4294967197u32;
    let mut w10_0 = vec![0u32; params.n * params.n]; 
    let mut w10_1 = vec![0u32; params.n * params.n];
    let mut w10_2 = vec![0u32; params.n * params.n];
    let mut w10_3 = vec![0u32; params.n * params.n]; 

    let s_ptr_packed = s.as_ptr() as *const u8;
    let c99_vec = _mm512_set1_epi32(99);
    let mask_0f = _mm_set1_epi8(0x0F);
    for j in 0..params.n {
        for l in 0..8{
            let gv0 = ((a0[j] as u64 * params.g_matrix_4[l] as u64) % (q_u32 as u64)) as u32;
            let gv1 = ((a1[j] as u64 * params.g_matrix_4[l] as u64) % (q_u32 as u64)) as u32;
            let gv2 = ((a2[j] as u64 * params.g_matrix_4[l] as u64) % (q_u32 as u64)) as u32;
            let gv3 = ((a3[j] as u64 * params.g_matrix_4[l] as u64) % (q_u32 as u64)) as u32;

            let mut lut0 = [0u32; 16];
            let mut lut1 = [0u32; 16];
            let mut lut2 = [0u32; 16];
            let mut lut3 = [0u32; 16];
            
            for v in 1..16 {
                let mut tmp0 = lut0[v - 1] as u64 + gv0 as u64;
                if tmp0 >= q_u32 as u64 { tmp0 -= q_u32 as u64; }
                lut0[v] = tmp0 as u32;

                let mut tmp1 = lut1[v - 1] as u64 + gv1 as u64;
                if tmp1 >= q_u32 as u64 { tmp1 -= q_u32 as u64; }
                lut1[v] = tmp1 as u32;

                let mut tmp2 = lut2[v - 1] as u64 + gv2 as u64;
                if tmp2 >= q_u32 as u64 { tmp2 -= q_u32 as u64; }
                lut2[v] = tmp2 as u32;

                let mut tmp3 = lut3[v - 1] as u64 + gv3 as u64;
                if tmp3 >= q_u32 as u64 { tmp3 -= q_u32 as u64; }
                lut3[v] = tmp3 as u32;
            }
            
            let lut0_vec = _mm512_loadu_si512(lut0.as_ptr() as *const _);
            let lut1_vec = _mm512_loadu_si512(lut1.as_ptr() as *const _);
            let lut2_vec = _mm512_loadu_si512(lut2.as_ptr() as *const _);
            let lut3_vec = _mm512_loadu_si512(lut3.as_ptr() as *const _);
            let h_offset_bytes = ((j*8+l) * params.n * params.n)/2;
            let s_base_ptr = s_ptr_packed.add(h_offset_bytes);
            let w0_ptr_base = w10_0.as_mut_ptr() as *mut __m512i;
            let w1_ptr_base = w10_1.as_mut_ptr() as *mut __m512i;
            let w2_ptr_base = w10_2.as_mut_ptr() as *mut __m512i;
            let w3_ptr_base = w10_3.as_mut_ptr() as *mut __m512i;
            for i in 0..params.n {
                let w_offset_vecs = (i * params.n) / 16;
                let w_offset_bytes = (i * params.n) / 2;
                let s_row_ptr = s_base_ptr.add(w_offset_bytes);
                for k_vec in (0..(params.n / 16)).step_by(2) {
                    let packed_ptr = s_row_ptr.add((k_vec / 2) * 16);
                    let packed_128 = _mm_loadu_si128(packed_ptr as *const __m128i);
                    let bytes_0 = _mm_and_si128(packed_128, mask_0f);
                    let s_vec_0 = _mm512_cvtepu8_epi32(bytes_0);
                    let bytes_1 = _mm_and_si128(_mm_srli_epi16(packed_128, 4), mask_0f);
                    let s_vec_1 = _mm512_cvtepu8_epi32(bytes_1);
                    let w_idx_0 = w_offset_vecs + k_vec;
                    let p0_0 = _mm512_permutexvar_epi32(s_vec_0, lut0_vec);
                    let p1_0 = _mm512_permutexvar_epi32(s_vec_0, lut1_vec);
                    let p2_0 = _mm512_permutexvar_epi32(s_vec_0, lut2_vec);
                    let p3_0 = _mm512_permutexvar_epi32(s_vec_0, lut3_vec);
                    let w0_0 = _mm512_loadu_si512(w0_ptr_base.add(w_idx_0));
                    let sum0_0 = _mm512_add_epi32(w0_0, p0_0);
                    let carry0_0 = _mm512_cmplt_epu32_mask(sum0_0, w0_0);
                    _mm512_storeu_si512(w0_ptr_base.add(w_idx_0), _mm512_mask_add_epi32(sum0_0, carry0_0, sum0_0, c99_vec));
                    let w1_0 = _mm512_loadu_si512(w1_ptr_base.add(w_idx_0));
                    let sum1_0 = _mm512_add_epi32(w1_0, p1_0);
                    let carry1_0 = _mm512_cmplt_epu32_mask(sum1_0, w1_0);
                    _mm512_storeu_si512(w1_ptr_base.add(w_idx_0), _mm512_mask_add_epi32(sum1_0, carry1_0, sum1_0, c99_vec));
                    let w2_0 = _mm512_loadu_si512(w2_ptr_base.add(w_idx_0));
                    let sum2_0 = _mm512_add_epi32(w2_0, p2_0);
                    let carry2_0 = _mm512_cmplt_epu32_mask(sum2_0, w2_0);
                    _mm512_storeu_si512(w2_ptr_base.add(w_idx_0), _mm512_mask_add_epi32(sum2_0, carry2_0, sum2_0, c99_vec));
                    let w3_0 = _mm512_loadu_si512(w3_ptr_base.add(w_idx_0));
                    let sum3_0 = _mm512_add_epi32(w3_0, p3_0);
                    let carry3_0 = _mm512_cmplt_epu32_mask(sum3_0, w3_0);
                    _mm512_storeu_si512(w3_ptr_base.add(w_idx_0), _mm512_mask_add_epi32(sum3_0, carry3_0, sum3_0, c99_vec));
                    let w_idx_1 = w_offset_vecs + k_vec + 1;
                    let p0_1 = _mm512_permutexvar_epi32(s_vec_1, lut0_vec);
                    let p1_1 = _mm512_permutexvar_epi32(s_vec_1, lut1_vec);
                    let p2_1 = _mm512_permutexvar_epi32(s_vec_1, lut2_vec);
                    let p3_1 = _mm512_permutexvar_epi32(s_vec_1, lut3_vec);
                    let w0_1 = _mm512_loadu_si512(w0_ptr_base.add(w_idx_1));
                    let sum0_1 = _mm512_add_epi32(w0_1, p0_1);
                    let carry0_1 = _mm512_cmplt_epu32_mask(sum0_1, w0_1);
                    _mm512_storeu_si512(w0_ptr_base.add(w_idx_1), _mm512_mask_add_epi32(sum0_1, carry0_1, sum0_1, c99_vec));
                    let w1_1 = _mm512_loadu_si512(w1_ptr_base.add(w_idx_1));
                    let sum1_1 = _mm512_add_epi32(w1_1, p1_1);
                    let carry1_1 = _mm512_cmplt_epu32_mask(sum1_1, w1_1);
                    _mm512_storeu_si512(w1_ptr_base.add(w_idx_1), _mm512_mask_add_epi32(sum1_1, carry1_1, sum1_1, c99_vec));
                    let w2_1 = _mm512_loadu_si512(w2_ptr_base.add(w_idx_1));
                    let sum2_1 = _mm512_add_epi32(w2_1, p2_1);
                    let carry2_1 = _mm512_cmplt_epu32_mask(sum2_1, w2_1);
                    _mm512_storeu_si512(w2_ptr_base.add(w_idx_1), _mm512_mask_add_epi32(sum2_1, carry2_1, sum2_1, c99_vec));
                    let w3_1 = _mm512_loadu_si512(w3_ptr_base.add(w_idx_1));
                    let sum3_1 = _mm512_add_epi32(w3_1, p3_1);
                    let carry3_1 = _mm512_cmplt_epu32_mask(sum3_1, w3_1);
                    _mm512_storeu_si512(w3_ptr_base.add(w_idx_1), _mm512_mask_add_epi32(sum3_1, carry3_1, sum3_1, c99_vec));
                }
            }
        }
    }
    let q_vec = _mm512_set1_epi32(q_u32 as i32);
    let w0_ptr = w0.as_mut_ptr() as *mut __m512i;
    let w1_ptr = w1.as_mut_ptr() as *mut __m512i;
    let w2_ptr = w2.as_mut_ptr() as *mut __m512i;
    let w3_ptr = w3.as_mut_ptr() as *mut __m512i;
    let w10_0_ptr_const = w10_0.as_ptr() as *const __m512i;
    let w10_1_ptr_const = w10_1.as_ptr() as *const __m512i;
    let w10_2_ptr_const = w10_2.as_ptr() as *const __m512i;
    let w10_3_ptr_const = w10_3.as_ptr() as *const __m512i;
    let mask_0x3 = _mm512_set1_epi32(0x3);
    let ring_size_vecs = params.n / 16;

    for i in 0..params.n {
        let src_row_offset_vecs = i * ring_size_vecs;
        
        for k in 0..16 {
            let shift_val = (k * 2) as i32; 
            let shift_vec = _mm512_set1_epi32(shift_val); 
            let dst_row_offset_vecs = (16 * i + k) * ring_size_vecs;
            
            for j_vec in 0..ring_size_vecs {
                let src_idx = src_row_offset_vecs + j_vec;
                let dst_idx = dst_row_offset_vecs + j_vec;

                let val0 = _mm512_loadu_si512(w10_0_ptr_const.add(src_idx));
                let ge_q0 = _mm512_cmpge_epu32_mask(val0, q_vec);
                let val0_mod = _mm512_mask_sub_epi32(val0, ge_q0, val0, q_vec);
                let shifted0 = _mm512_srlv_epi32(val0_mod, shift_vec);
                _mm512_storeu_si512(w0_ptr.add(dst_idx), _mm512_and_epi32(shifted0, mask_0x3));

                let val1 = _mm512_loadu_si512(w10_1_ptr_const.add(src_idx));
                let ge_q1 = _mm512_cmpge_epu32_mask(val1, q_vec);
                let val1_mod = _mm512_mask_sub_epi32(val1, ge_q1, val1, q_vec);
                let shifted1 = _mm512_srlv_epi32(val1_mod, shift_vec);
                _mm512_storeu_si512(w1_ptr.add(dst_idx), _mm512_and_epi32(shifted1, mask_0x3));

                let val2 = _mm512_loadu_si512(w10_2_ptr_const.add(src_idx));
                let ge_q2 = _mm512_cmpge_epu32_mask(val2, q_vec);
                let val2_mod = _mm512_mask_sub_epi32(val2, ge_q2, val2, q_vec);
                let shifted2 = _mm512_srlv_epi32(val2_mod, shift_vec);
                _mm512_storeu_si512(w2_ptr.add(dst_idx), _mm512_and_epi32(shifted2, mask_0x3));

                let val3 = _mm512_loadu_si512(w10_3_ptr_const.add(src_idx));
                let ge_q3 = _mm512_cmpge_epu32_mask(val3, q_vec);
                let val3_mod = _mm512_mask_sub_epi32(val3, ge_q3, val3, q_vec);
                let shifted3 = _mm512_srlv_epi32(val3_mod, shift_vec);
                _mm512_storeu_si512(w3_ptr.add(dst_idx), _mm512_and_epi32(shifted3, mask_0x3));
            }
        }
    }

    // let duration = start.elapsed();
    // println!("w computed: {:?}", duration);
    // let start = Instant::now();

    // Compute k
    let q3 = _mm512_set1_epi32(2079301633);
    let q3_inv = _mm512_set1_epi32(-1475321855);
    let q4 = _mm512_set1_epi32(2079305729);
    let q4_inv = _mm512_set1_epi32(-1659875327);
    let mut acc_k0_1 = [_mm512_set1_epi32(0); 128];
    let mut acc_k0_2 = [_mm512_set1_epi32(0); 128];
    let mut acc_k1_1 = [_mm512_set1_epi32(0); 128];
    let mut acc_k1_2 = [_mm512_set1_epi32(0); 128];
    let mut acc_k2_1 = [_mm512_set1_epi32(0); 128];
    let mut acc_k2_2 = [_mm512_set1_epi32(0); 128];
    let mut acc_k3_1 = [_mm512_set1_epi32(0); 128];
    let mut acc_k3_2 = [_mm512_set1_epi32(0); 128];

    for i in 0..params.height_2 {
        let offset = i << 10;
        let mut buf_d_q3 = Align64([0u32; 2048]);
        buf_d_q3.0[0..1024].copy_from_slice(&params.d[offset..offset+1024]);
        let mut buf_d_q4 = buf_d_q3.clone();
        params.plans[0].fwd(&mut buf_d_q3.0);
        params.plans[1].fwd(&mut buf_d_q4.0);

        let d_q3_ptr = buf_d_q3.0.as_ptr() as *const __m512i;
        let d_q4_ptr = buf_d_q4.0.as_ptr() as *const __m512i;

        let w_slices = [
            &w0[offset..offset+1024], &w1[offset..offset+1024], 
            &w2[offset..offset+1024], &w3[offset..offset+1024]
        ];
        
        for k in 0..4 {
            let mut buf_w_q3 = Align64([0u32; 2048]);
            buf_w_q3.0[0..1024].copy_from_slice(w_slices[k]);
            let mut buf_w_q4 = buf_w_q3.clone();
            
            params.plans[0].fwd(&mut buf_w_q3.0);
            params.plans[1].fwd(&mut buf_w_q4.0);
            
            let w_q3_ptr = buf_w_q3.0.as_mut_ptr() as *mut __m512i;
            let w_q4_ptr = buf_w_q4.0.as_mut_ptr() as *mut __m512i;
            
            for j in 0..128 {
                let d3_vec = _mm512_loadu_si512(d_q3_ptr.add(j));
                let d4_vec = _mm512_loadu_si512(d_q4_ptr.add(j));
                
                let w3_vec = _mm512_loadu_si512(w_q3_ptr.add(j));
                let w4_vec = _mm512_loadu_si512(w_q4_ptr.add(j));
                
                let prod3 = montproduct(d3_vec, w3_vec, q3, q3_inv);
                let prod4 = montproduct(d4_vec, w4_vec, q4, q4_inv);
                
                match k {
                    0 => {
                        acc_k0_1[j] = barrett_fake_2079301633(_mm512_add_epi32(acc_k0_1[j], prod3));
                        acc_k0_2[j] = barrett_fake_2079305729(_mm512_add_epi32(acc_k0_2[j], prod4));
                    },
                    1 => {
                        acc_k1_1[j] = barrett_fake_2079301633(_mm512_add_epi32(acc_k1_1[j], prod3));
                        acc_k1_2[j] = barrett_fake_2079305729(_mm512_add_epi32(acc_k1_2[j], prod4));
                    },
                    2 => {
                        acc_k2_1[j] = barrett_fake_2079301633(_mm512_add_epi32(acc_k2_1[j], prod3));
                        acc_k2_2[j] = barrett_fake_2079305729(_mm512_add_epi32(acc_k2_2[j], prod4));
                    },
                    _ => {
                        acc_k3_1[j] = barrett_fake_2079301633(_mm512_add_epi32(acc_k3_1[j], prod3));
                        acc_k3_2[j] = barrett_fake_2079305729(_mm512_add_epi32(acc_k3_2[j], prod4));
                    }
                }
            }
        }
    }

    let q_vec = _mm512_set1_epi32(((1u64<<32)-99) as i32);
    let r_base_ptr = r.as_mut_ptr() as *mut __m512i;

    let mut extract_k_and_r = |
        acc1: &mut [__m512i; 128], 
        acc2: &mut [__m512i; 128], 
        r_out_ptr: *mut __m512i, 
        k_out_ptr: *mut __m512i
    | {
        let mask3 = _mm512_set1_epi32(3);

        for i in 0..128 {
            acc1[i] = barrett_mul_2097152_2079301633(acc1[i]);
            acc2[i] = barrett_mul_2097152_2079305729(acc2[i]);
        }

        params.plans[0].inv(slice::from_raw_parts_mut(acc1.as_mut_ptr() as *mut u32, 2048));
        params.plans[1].inv(slice::from_raw_parts_mut(acc2.as_mut_ptr() as *mut u32, 2048));

        let mut full_res = Align64([0u32; 2048]);
        irns_2048(acc1, acc2, full_res.0.as_mut_ptr() as *mut __m512i, 0);
    
        let full_res_ptr = full_res.0.as_ptr() as *const __m512i;
        for j in 0..64 {
            let low_vec = _mm512_load_si512(full_res_ptr.add(j));
            let high_vec = _mm512_load_si512(full_res_ptr.add(64 + j));
            _mm512_storeu_si512(r_out_ptr.add(j), high_vec);
            let lt_mask = _mm512_cmplt_epu32_mask(low_vec, high_vec);
            let diff_vec = _mm512_sub_epi32(low_vec, high_vec);
            let mut k_vec = _mm512_mask_add_epi32(diff_vec, lt_mask, diff_vec, q_vec);
            for step in 0..16 {
                let decomp_val = _mm512_and_epi32(k_vec, mask3);
                _mm512_storeu_si512(k_out_ptr.add((step << 6) + j), decomp_val);
                k_vec = _mm512_srli_epi32(k_vec, 2);
            }
        }
    };

    let k0_ptr = k0.as_mut_ptr() as *mut __m512i;
    let k1_ptr = k1.as_mut_ptr() as *mut __m512i;
    let k2_ptr = k2.as_mut_ptr() as *mut __m512i;
    let k3_ptr = k3.as_mut_ptr() as *mut __m512i;

    extract_k_and_r(&mut acc_k0_1, &mut acc_k0_2, r_base_ptr, k0_ptr);
    extract_k_and_r(&mut acc_k1_1, &mut acc_k1_2, r_base_ptr.add(1<<6), k1_ptr);
    extract_k_and_r(&mut acc_k2_1, &mut acc_k2_2, r_base_ptr.add(1<<7), k2_ptr);
    extract_k_and_r(&mut acc_k3_1, &mut acc_k3_2, r_base_ptr.add(3<<6), k3_ptr);

    // let duration = start.elapsed();
    // println!("k_i computed: {:?}", duration);
    // let start = Instant::now();

    // Compute v
    let mut acc_v1 = [_mm512_set1_epi32(0); 128];
    let mut acc_v2 = [_mm512_set1_epi32(0); 128];

    let e_slice = unsafe { std::slice::from_raw_parts(params.e.inner.as_ptr() as *const u32, 64 * 1024) };
    let k_slices = [
        unsafe { std::slice::from_raw_parts(k0.inner.as_ptr() as *const u32, 16 * 1024) },
        unsafe { std::slice::from_raw_parts(k1.inner.as_ptr() as *const u32, 16 * 1024) },
        unsafe { std::slice::from_raw_parts(k2.inner.as_ptr() as *const u32, 16 * 1024) },
        unsafe { std::slice::from_raw_parts(k3.inner.as_ptr() as *const u32, 16 * 1024) },
    ];

    for m in 0..4 {
        for l in 0..16 {
            let e_offset = (m * 16 + l) << 10;
            let k_offset = l << 10;
            
            let mut buf_e_q3 = Align64([0u32; 2048]);
            buf_e_q3.0[0..1024].copy_from_slice(&e_slice[e_offset .. e_offset+1024]);
            let mut buf_e_q4 = buf_e_q3.clone();
            
            let mut buf_k_q3 = Align64([0u32; 2048]);
            buf_k_q3.0[0..1024].copy_from_slice(&k_slices[m][k_offset .. k_offset+1024]);
            let mut buf_k_q4 = buf_k_q3.clone();

            params.plans[0].fwd(&mut buf_e_q3.0);
            params.plans[1].fwd(&mut buf_e_q4.0);
            params.plans[0].fwd(&mut buf_k_q3.0);
            params.plans[1].fwd(&mut buf_k_q4.0);
            
            let e_q3_ptr = buf_e_q3.0.as_ptr() as *const __m512i;
            let e_q4_ptr = buf_e_q4.0.as_ptr() as *const __m512i;
            let k_q3_ptr = buf_k_q3.0.as_ptr() as *const __m512i;
            let k_q4_ptr = buf_k_q4.0.as_ptr() as *const __m512i;
            
            for j in 0..128 {
                let e3_vec = _mm512_load_si512(e_q3_ptr.add(j));
                let k3_vec = _mm512_load_si512(k_q3_ptr.add(j));
                acc_v1[j] = barrett_fake_2079301633(_mm512_add_epi32(acc_v1[j], montproduct(e3_vec, k3_vec, q3, q3_inv)));
                
                let e4_vec = _mm512_load_si512(e_q4_ptr.add(j));
                let k4_vec = _mm512_load_si512(k_q4_ptr.add(j));
                acc_v2[j] = barrett_fake_2079305729(_mm512_add_epi32(acc_v2[j], montproduct(e4_vec, k4_vec, q4, q4_inv)));
            }
        }
    }

    for i in 0..128 {
        acc_v1[i] = barrett_mul_2097152_2079301633(acc_v1[i]);
        acc_v2[i] = barrett_mul_2097152_2079305729(acc_v2[i]);
    }
    params.plans[0].inv(slice::from_raw_parts_mut(acc_v1.as_mut_ptr() as *mut u32, 2048));
    params.plans[1].inv(slice::from_raw_parts_mut(acc_v2.as_mut_ptr() as *mut u32, 2048));
    
    let mut full_res_v = Align64([0u32; 2048]);
    irns_2048(&acc_v1, &acc_v2, full_res_v.0.as_mut_ptr() as *mut __m512i, 0);
    
    let full_res_v_ptr = full_res_v.0.as_ptr() as *const __m512i;
    
    let rv_ptr = r_base_ptr.add(1<<8); 
    let v_ptr_simd = v.inner.as_mut_ptr() as *mut __m512i;
    
    for j in 0..64 {
        let low_vec = _mm512_load_si512(full_res_v_ptr.add(j));
        let high_vec = _mm512_load_si512(full_res_v_ptr.add(64 + j));
        _mm512_storeu_si512(rv_ptr.add(j), high_vec);
        let lt_mask = _mm512_cmplt_epu32_mask(low_vec, high_vec);
        let diff_vec = _mm512_sub_epi32(low_vec, high_vec);
        let v_vec = _mm512_mask_add_epi32(diff_vec, lt_mask, diff_vec, q_vec);
        _mm512_storeu_si512(v_ptr_simd.add(j), v_vec);
    }

    // let duration = start.elapsed();
    // println!("v computed: {:?}", duration);
    // let start = Instant::now();
    
    // Compute z
    fold_witness_u32(&mut long_z, &c, &s);
    let half_q = q_u32 / 2;
    let q_vec = _mm512_set1_epi32(q_u32 as i32);
    let half_q_vec = _mm512_set1_epi32(half_q as i32);
    
    let two_vec = _mm512_set1_epi32(2);
    let mask3_vec = _mm512_set1_epi32(3);
    
    let long_z_ptr_base = long_z.as_ptr() as *const __m512i;
    let z_ptr_base = z.as_mut_ptr();
    
    for i in 0..params.height_4 {
        let offset_long_vecs = (i * params.n) / 16;
        let long_z_ptr = long_z_ptr_base.add(offset_long_vecs);
        let z_row_ptr = z_ptr_base.add(i * 8 * params.n);
        
        let z_ptrs = [
            z_row_ptr,
            z_row_ptr.add(params.n),
            z_row_ptr.add(2 * params.n),
            z_row_ptr.add(3 * params.n),
            z_row_ptr.add(4 * params.n),
            z_row_ptr.add(5 * params.n),
            z_row_ptr.add(6 * params.n),
            z_row_ptr.add(7 * params.n),
        ];

        for j_vec in 0..(params.n / 16) {
            let val_u32 = _mm512_loadu_si512(long_z_ptr.add(j_vec));
            let gt_mask = _mm512_cmpgt_epu32_mask(val_u32, half_q_vec);
            let mut v = _mm512_mask_sub_epi32(val_u32, gt_mask, val_u32, q_vec);
            
            let mut limbs = [_mm512_setzero_si512(); 8];

            for l in 0..8 {
                let v_plus_2 = _mm512_add_epi32(v, two_vec);
                let mut limb = _mm512_and_epi32(v_plus_2, mask3_vec);
                limb = _mm512_sub_epi32(limb, two_vec);
                v = _mm512_srai_epi32(v_plus_2, 2);
                limbs[l] = limb;
            }

            let j_offset = j_vec * 16;

            for l in 0..8 {
                let l_8 = _mm512_cvtepi32_epi8(limbs[l]);
                _mm_storeu_si128(z_ptrs[l].add(j_offset) as *mut __m128i, l_8);
            }
        }
    }

    // let duration = start.elapsed();
    // println!("fold witness computed: {:?}", duration);
    // let start = Instant::now();

    // compute u_i
    let c99_vec = _mm512_set1_epi32(99);
    macro_rules! add_mod_q {
        ($a:expr, $b:expr) => {{
            let sum = _mm512_add_epi32($a, $b);
            let carry = _mm512_cmplt_epu32_mask(sum, $a);
            _mm512_mask_add_epi32(sum, carry, sum, c99_vec)
        }};
    }
    for i in 0..params.n {
        for j in 0..16{
            let gv = params.g_matrix_2[j] as u64;
            let f_b0 = ((b0[i] as u64 * gv) % (q_u32 as u64)) as u32;
            let f_b1 = ((b1[i] as u64 * gv) % (q_u32 as u64)) as u32;
            let f_b2 = ((b2[i] as u64 * gv) % (q_u32 as u64)) as u32;
            let f_b3 = ((b3[i] as u64 * gv) % (q_u32 as u64)) as u32;
            let f_b1_2 = ((2u64 * b1[i] as u64 * gv) % (q_u32 as u64)) as u32;
            let f_b2_2 = ((2u64 * b2[i] as u64 * gv) % (q_u32 as u64)) as u32;
            let f_b3_2 = ((2u64 * b3[i] as u64 * gv) % (q_u32 as u64)) as u32;
            let mut lut_b0 = [0u32; 16];
            let mut lut_b1 = [0u32; 16];
            let mut lut_b2 = [0u32; 16];
            let mut lut_b3 = [0u32; 16];
            let mut lut_b1_2 = [0u32; 16];
            let mut lut_b2_2 = [0u32; 16];
            let mut lut_b3_2 = [0u32; 16];

            for v in 1..4 {
                let mut t0 = lut_b0[v - 1] as u64 + f_b0 as u64;
                if t0 >= q_u32 as u64 { t0 -= q_u32 as u64; }
                lut_b0[v] = t0 as u32;

                let mut t1 = lut_b1[v - 1] as u64 + f_b1 as u64;
                if t1 >= q_u32 as u64 { t1 -= q_u32 as u64; }
                lut_b1[v] = t1 as u32;

                let mut t2 = lut_b2[v - 1] as u64 + f_b2 as u64;
                if t2 >= q_u32 as u64 { t2 -= q_u32 as u64; }
                lut_b2[v] = t2 as u32;

                let mut t3 = lut_b3[v - 1] as u64 + f_b3 as u64;
                if t3 >= q_u32 as u64 { t3 -= q_u32 as u64; }
                lut_b3[v] = t3 as u32;

                let mut t1_2 = lut_b1_2[v - 1] as u64 + f_b1_2 as u64;
                if t1_2 >= q_u32 as u64 { t1_2 -= q_u32 as u64; }
                lut_b1_2[v] = t1_2 as u32;

                let mut t2_2 = lut_b2_2[v - 1] as u64 + f_b2_2 as u64;
                if t2_2 >= q_u32 as u64 { t2_2 -= q_u32 as u64; }
                lut_b2_2[v] = t2_2 as u32;

                let mut t3_2 = lut_b3_2[v - 1] as u64 + f_b3_2 as u64;
                if t3_2 >= q_u32 as u64 { t3_2 -= q_u32 as u64; }
                lut_b3_2[v] = t3_2 as u32;
            }
            let vec_b0 = _mm512_loadu_si512(lut_b0.as_ptr() as *const _);
            let vec_b1 = _mm512_loadu_si512(lut_b1.as_ptr() as *const _);
            let vec_b2 = _mm512_loadu_si512(lut_b2.as_ptr() as *const _);
            let vec_b3 = _mm512_loadu_si512(lut_b3.as_ptr() as *const _);
            let vec_b1_2 = _mm512_loadu_si512(lut_b1_2.as_ptr() as *const _);
            let vec_b2_2 = _mm512_loadu_si512(lut_b2_2.as_ptr() as *const _);
            let vec_b3_2 = _mm512_loadu_si512(lut_b3_2.as_ptr() as *const _);

            let offset = (i*16+j) * params.n;
            let w0_ptr = w0.as_ptr().add(offset) as *const __m512i;
            let w1_ptr = w1.as_ptr().add(offset) as *const __m512i;
            let w2_ptr = w2.as_ptr().add(offset) as *const __m512i;
            let w3_ptr = w3.as_ptr().add(offset) as *const __m512i;

            let u0_ptr = u0.as_mut_ptr() as *mut __m512i;
            let u1_ptr = u1.as_mut_ptr() as *mut __m512i;
            let u2_ptr = u2.as_mut_ptr() as *mut __m512i;
            let u3_ptr = u3.as_mut_ptr() as *mut __m512i;

            for k in 0..(params.n / 16) {
                let w0_v = _mm512_loadu_si512(w0_ptr.add(k));
                let w1_v = _mm512_loadu_si512(w1_ptr.add(k));
                let w2_v = _mm512_loadu_si512(w2_ptr.add(k));
                let w3_v = _mm512_loadu_si512(w3_ptr.add(k));

                // u0 = b0*w0 + 2*b3*w1 + 2*b2*w2 + 2*b1*w3
                let p0_u0 = _mm512_permutexvar_epi32(w0_v, vec_b0);
                let p1_u0 = _mm512_permutexvar_epi32(w1_v, vec_b3_2);
                let p2_u0 = _mm512_permutexvar_epi32(w2_v, vec_b2_2);
                let p3_u0 = _mm512_permutexvar_epi32(w3_v, vec_b1_2);
                let sum_u0_01 = add_mod_q!(p0_u0, p1_u0);
                let sum_u0_23 = add_mod_q!(p2_u0, p3_u0);
                let sum_u0 = add_mod_q!(sum_u0_01, sum_u0_23);
                let curr_u0 = _mm512_loadu_si512(u0_ptr.add(k));
                _mm512_storeu_si512(u0_ptr.add(k), add_mod_q!(curr_u0, sum_u0));

                // u1 = b1*w0 + b0*w1 + 2*b3*w2 + 2*b2*w3
                let p0_u1 = _mm512_permutexvar_epi32(w0_v, vec_b1);
                let p1_u1 = _mm512_permutexvar_epi32(w1_v, vec_b0);
                let p2_u1 = _mm512_permutexvar_epi32(w2_v, vec_b3_2);
                let p3_u1 = _mm512_permutexvar_epi32(w3_v, vec_b2_2);
                let sum_u1_01 = add_mod_q!(p0_u1, p1_u1);
                let sum_u1_23 = add_mod_q!(p2_u1, p3_u1);
                let sum_u1 = add_mod_q!(sum_u1_01, sum_u1_23);
                let curr_u1 = _mm512_loadu_si512(u1_ptr.add(k));
                _mm512_storeu_si512(u1_ptr.add(k), add_mod_q!(curr_u1, sum_u1));

                // u2 = b2*w0 + b1*w1 + b0*w2 + 2*b3*w3
                let p0_u2 = _mm512_permutexvar_epi32(w0_v, vec_b2);
                let p1_u2 = _mm512_permutexvar_epi32(w1_v, vec_b1);
                let p2_u2 = _mm512_permutexvar_epi32(w2_v, vec_b0);
                let p3_u2 = _mm512_permutexvar_epi32(w3_v, vec_b3_2);
                let sum_u2_01 = add_mod_q!(p0_u2, p1_u2);
                let sum_u2_23 = add_mod_q!(p2_u2, p3_u2);
                let sum_u2 = add_mod_q!(sum_u2_01, sum_u2_23);
                let curr_u2 = _mm512_loadu_si512(u2_ptr.add(k));
                _mm512_storeu_si512(u2_ptr.add(k), add_mod_q!(curr_u2, sum_u2));

                // u3 = b3*w0 + b2*w1 + b1*w2 + b0*w3
                let p0_u3 = _mm512_permutexvar_epi32(w0_v, vec_b3);
                let p1_u3 = _mm512_permutexvar_epi32(w1_v, vec_b2);
                let p2_u3 = _mm512_permutexvar_epi32(w2_v, vec_b1);
                let p3_u3 = _mm512_permutexvar_epi32(w3_v, vec_b0);
                let sum_u3_01 = add_mod_q!(p0_u3, p1_u3);
                let sum_u3_23 = add_mod_q!(p2_u3, p3_u3);
                let sum_u3 = add_mod_q!(sum_u3_01, sum_u3_23);
                let curr_u3 = _mm512_loadu_si512(u3_ptr.add(k));
                _mm512_storeu_si512(u3_ptr.add(k), add_mod_q!(curr_u3, sum_u3));
            }
        }
    }
    for k in 0..params.n {
        u0[k] %= q_u32;
        u1[k] %= q_u32;
        u2[k] %= q_u32;
        u3[k] %= q_u32;
    }

    // let duration = start.elapsed();
    // println!("u_i computed: {:?}", duration);
    // let start = Instant::now();

    // compute r: Mz = y + r*(x^1024+1)
    let q_base_u32 = 4294967197u32;
    let q_vec = _mm512_set1_epi32(q_base_u32 as i32);
    let c99_vec = _mm512_set1_epi32(99);
    let mut lut_vecs = [_mm512_setzero_si512(); 16];
    for j in 0..16 {
        let mut lut = [0u32; 16];
        for v in 1..4 {
            lut[v] = (lut[v-1] + params.g_matrix_2[j]) % q_base_u32;
        }
        lut_vecs[j] = _mm512_loadu_si512(lut.as_ptr() as *const _);
    }

    let r10_ptr = r.as_mut_ptr().add(10 * params.n);
    let r11_ptr = r.as_mut_ptr().add(11 * params.n);
    let r12_ptr = r.as_mut_ptr().add(12 * params.n);
    let r13_ptr = r.as_mut_ptr().add(13 * params.n);
    let r14_ptr = r.as_mut_ptr().add(14 * params.n);

    for i in 0..params.n {
        let c_slice = &c[i << 4 .. (i + 1) << 4];
        for j in 0..16{
            let lut_vec = lut_vecs[j];
            let offset = (i << 14) + (j << 10);
            
            let w0_ptr = w0.as_ptr().add(offset);
            let w1_ptr = w1.as_ptr().add(offset);
            let w2_ptr = w2.as_ptr().add(offset);
            let w3_ptr = w3.as_ptr().add(offset);
            let t_ptr  = commitment.t().as_ptr().add(offset);
            
            for k in 0..16 {
                let idx_raw = c_slice[k] as u32;
                accum_high_part_simd(r10_ptr, w0_ptr, idx_raw, lut_vec, q_vec, c99_vec);
                accum_high_part_simd(r11_ptr, w1_ptr, idx_raw, lut_vec, q_vec, c99_vec);
                accum_high_part_simd(r12_ptr, w2_ptr, idx_raw, lut_vec, q_vec, c99_vec);
                accum_high_part_simd(r13_ptr, w3_ptr, idx_raw, lut_vec, q_vec, c99_vec);
                accum_high_part_simd(r14_ptr, t_ptr,  idx_raw, lut_vec, q_vec, c99_vec);
            }
        }
    }
    // r14 - Az
    let mut acc_q3 = [[_mm512_setzero_si512(); 128]; 8];
    let mut acc_q4 = [[_mm512_setzero_si512(); 128]; 8];
    
    let q3_vec = _mm512_set1_epi32(2079301633);
    let q3_inv_vec = _mm512_set1_epi32(-1475321855);
    let q4_vec = _mm512_set1_epi32(2079305729);
    let q4_inv_vec = _mm512_set1_epi32(-1659875327);

    for i in 0..(1<<13) {
        let offset_a = i << 10;
        let offset_z = i << 13;
        
        let mut buf_a_q3 = Align64([0u32; 2048]);
        let mut buf_a_q4 = Align64([0u32; 2048]);
        buf_a_q3.0[0..1024].copy_from_slice(&params.d[offset_a..offset_a+1024]);
        buf_a_q4.0[0..1024].copy_from_slice(&params.d[offset_a..offset_a+1024]);
        params.plans[0].fwd(&mut buf_a_q3.0);
        params.plans[1].fwd(&mut buf_a_q4.0);
        let a_q3_ptr = buf_a_q3.0.as_mut_ptr() as *mut __m512i;
        let a_q4_ptr = buf_a_q4.0.as_mut_ptr() as *mut __m512i;
        
        for j in 0..8 {
            let mut buf_z_q3 = Align64([0u32; 2048]);
            let mut buf_z_q4 = Align64([0u32; 2048]);
            let z_ptr = z.as_ptr().add(offset_z + j * 1024);
            let z_q3_ptr = buf_z_q3.0.as_mut_ptr() as *mut __m512i;
            let z_q4_ptr = buf_z_q4.0.as_mut_ptr() as *mut __m512i;
            for k in 0..64 {
                let z_i8 = _mm_loadu_si128(z_ptr.add(k * 16) as *const __m128i);
                let z_i32 = _mm512_cvtepi8_epi32(z_i8);
                
                let mask_neg = _mm512_cmplt_epi32_mask(z_i32, _mm512_setzero_si512());
                let z_q3 = _mm512_mask_add_epi32(z_i32, mask_neg, z_i32, q3_vec);
                let z_q4 = _mm512_mask_add_epi32(z_i32, mask_neg, z_i32, q4_vec);
                
                _mm512_store_si512(z_q3_ptr.add(k), z_q3);
                _mm512_store_si512(z_q4_ptr.add(k), z_q4);
            }

            params.plans[0].fwd(&mut buf_z_q3.0);
            params.plans[1].fwd(&mut buf_z_q4.0);
            
            for k in 0..128 {
                // Q3
                let a3_vec = _mm512_loadu_si512(a_q3_ptr.add(k));
                let z3_vec = _mm512_loadu_si512(z_q3_ptr.add(k));
                let prod3 = montproduct(a3_vec, z3_vec, q3_vec, q3_inv_vec);
                let sum3 = _mm512_add_epi32(acc_q3[j][k], prod3);
                let ge_q3 = _mm512_cmpge_epu32_mask(sum3, q3_vec);
                acc_q3[j][k] = _mm512_mask_sub_epi32(sum3, ge_q3, sum3, q3_vec);
                // Q4
                let a4_vec = _mm512_loadu_si512(a_q4_ptr.add(k));
                let z4_vec = _mm512_loadu_si512(z_q4_ptr.add(k));
                let prod4 = montproduct(a4_vec, z4_vec, q4_vec, q4_inv_vec);
                let sum4 = _mm512_add_epi32(acc_q4[j][k], prod4);
                let ge_q4 = _mm512_cmpge_epu32_mask(sum4, q4_vec);
                acc_q4[j][k] = _mm512_mask_sub_epi32(sum4, ge_q4, sum4, q4_vec);
            }
        }
    }
    
    let r_slice = &mut r[14 * params.n .. 15 * params.n];
    for j in 0..8 {
        for k in 0..128 {
            acc_q3[j][k] = barrett_mul_2097152_2079301633(acc_q3[j][k]);
            acc_q4[j][k] = barrett_mul_2097152_2079305729(acc_q4[j][k]);
        }
        params.plans[0].inv(std::slice::from_raw_parts_mut(acc_q3[j].as_mut_ptr() as *mut u32, 2048));
        params.plans[1].inv(std::slice::from_raw_parts_mut(acc_q4[j].as_mut_ptr() as *mut u32, 2048));
        let mut full_res_az = Align64([0u32; 2048]);
        irns_2048(&acc_q3[j], &acc_q4[j], full_res_az.0.as_mut_ptr() as *mut __m512i, 0);
        let gv = params.g_matrix_2[j] as u64;
        for k in 0..1024 {
            let high_val = full_res_az.0[1024 + k] as u64;
            let term = (high_val * gv) % (q_base_u32 as u64);
            if r_slice[k] >= term as u32 {
                r_slice[k] -= term as u32;
            } else {
                r_slice[k] = r_slice[k] + q_base_u32 - term as u32;
            }
        }
    }

    // let duration = start.elapsed();
    // println!("r computed: {:?}", duration);
    // let start = Instant::now();

    let wbundle: WBundle = (w0, w1, w2, w3, commitment.t(), z, k0, k1, k2, k3);

    // compute alpha
    let mut alpha_vec = vec![0u32; 4*1025];
    alpha_vec[0] = 1;
    alpha_vec[1] = 0;
    alpha_vec[2] = 0;
    alpha_vec[3] = 0;
    for i in 1..1025{
        let (prev, current) = alpha_vec.split_at_mut(4*i);
        extmul(&mut current[0..4], &prev[4*(i-1)..4*i], alpha);
    }
    // alpha_vec[4*1024] = - (alpha^1024 + 1)
    alpha_vec[4096] = ((params.q64 - ((alpha_vec[4096] as u64 + 1u64) % params.q64)) % params.q64) as u32;
    alpha_vec[4097] = ((params.q64 - (alpha_vec[4097] as u64 % params.q64)) % params.q64) as u32;
    alpha_vec[4098] = ((params.q64 - (alpha_vec[4098] as u64 % params.q64)) % params.q64) as u32;
    alpha_vec[4099] = ((params.q64 - (alpha_vec[4099] as u64 % params.q64)) % params.q64) as u32;

    // let duration = start.elapsed();
    // println!("alpha computed: {:?}", duration);
    // let start = Instant::now();

    // substitute alpha into M
    let height_2_ext = params.height_2*4;
    let height_z_ext = params.height_4*8*4;
    
    // M(alpha)
    let mut D_alpha = vec![0u32; height_2_ext];
    let mut DJ_alpha = vec![0u32; height_z_ext];
    let mut E0_alpha = vec![0u32; 1<<6];
    let mut E1_alpha = vec![0u32; 1<<6];
    let mut E2_alpha = vec![0u32; 1<<6];
    let mut E3_alpha = vec![0u32; 1<<6];
    let mut a0GJ_alpha = vec![0u32; height_z_ext];
    let mut a1GJ_alpha = vec![0u32; height_z_ext];
    let mut a2GJ_alpha = vec![0u32; height_z_ext];
    let mut a3GJ_alpha = vec![0u32; height_z_ext];
    let mut b0G_alpha = vec![0u32; height_2_ext];
    let mut b1G_alpha = vec![0u32; height_2_ext];
    let mut b2G_alpha = vec![0u32; height_2_ext];
    let mut b3G_alpha = vec![0u32; height_2_ext];
    let mut cG_alpha = vec![0u32; height_2_ext];
    
    let tmp = alpha_vec[0..4].to_vec(); 
    for i in 0..params.height_2{
        for j in 0..params.n{
            unsafe {
                ext_base_mla(&mut D_alpha[4*i..4*(i+1)], &alpha_vec[4*j..4*(j+1)], params.d[i*params.n+j]);
            }
        }
    }

    for i in 0..params.height_4{
        let d_alpha_i = [
            D_alpha[4*i],
            D_alpha[4*i+1],
            D_alpha[4*i+2],
            D_alpha[4*i+3],
        ];

        for j in 0..8 {
            let idx = i*8 + j;
            let gv = params.g_matrix_2[j] as u64;

            for c in 0..4 {
                let val = ((d_alpha_i[c] as u64 * gv) % params.q64) as u32;
                let neg_val = if val == 0 { 0 } else { params.q - val };
                DJ_alpha[4*idx + c] = neg_val;
            }
        }
    }

    for i in 0..16 {
        let offset_0 = i * params.n;
        let offset_1 = (16 + i) * params.n;
        let offset_2 = (32 + i) * params.n;
        let offset_3 = (48 + i) * params.n;
        for j in 0..params.n {
            unsafe {
                ext_base_mla(&mut E0_alpha[4*i..4*(i+1)], &alpha_vec[4*j..4*(j+1)], params.e[offset_0 + j]);
                ext_base_mla(&mut E1_alpha[4*i..4*(i+1)], &alpha_vec[4*j..4*(j+1)], params.e[offset_1 + j]);
                ext_base_mla(&mut E2_alpha[4*i..4*(i+1)], &alpha_vec[4*j..4*(j+1)], params.e[offset_2 + j]);
                ext_base_mla(&mut E3_alpha[4*i..4*(i+1)], &alpha_vec[4*j..4*(j+1)], params.e[offset_3 + j]);
            }
        }
    }

    for i in 0..params.n{
        for j in 0..16{
            let gv = params.g_matrix_2[j];
            let ngv = ((params.q64 - gv as u64) % params.q64) as u32;
            for k in 0..16{
                if c[i*16+k] < 1024{
                    let idx = c[i*16+k] as usize;
                    ext_base_mla(&mut cG_alpha[4*(i*16+j)..4*(i*16+j+1)], &alpha_vec[4*idx..4*(idx+1)], gv);
                }else{
                    let idx = (c[i*16+k] - 1024) as usize;
                    ext_base_mla(&mut cG_alpha[4*(i*16+j)..4*(i*16+j+1)], &alpha_vec[4*idx..4*(idx+1)], ngv);
                }
            }
        }
    }

    for i in 0..params.n {
        for j in 0..16 {
            let idx = i * 16 + j;
            let gv = params.g_matrix_2[j] as u64;
            unsafe {
                let val0 = ((b0[i] as u64 * gv) % params.q64) as u32;
                ext_base_mla(&mut b0G_alpha[4*idx..4*(idx+1)], &tmp, val0);
                let val1 = ((b1[i] as u64 * gv) % params.q64) as u32;
                ext_base_mla(&mut b1G_alpha[4*idx..4*(idx+1)], &tmp, val1);
                let val2 = ((b2[i] as u64 * gv) % params.q64) as u32;
                ext_base_mla(&mut b2G_alpha[4*idx..4*(idx+1)], &tmp, val2);
                let val3 = ((b3[i] as u64 * gv) % params.q64) as u32;
                ext_base_mla(&mut b3G_alpha[4*idx..4*(idx+1)], &tmp, val3);
            }
        }
    }
    
    for i in 0..params.n{
        let idx0 = i*64;
        for j in 0..8{
            let idx1 = idx0 + j*8;
            let gv1 = params.g_matrix_4[j] as u64;
            for k in 0..8{
                let idx2 = idx1 + k;
                let gv2 = (gv1 * params.g_matrix_2[k] as u64)%params.q64;
                unsafe {
                    ext_base_mla(&mut a0GJ_alpha[4*idx2..4*(idx2+1)], &tmp, ((a0[i] as u64 * gv2)%params.q64) as u32); 
                    ext_base_mla(&mut a1GJ_alpha[4*idx2..4*(idx2+1)], &tmp, ((a1[i] as u64 * gv2)%params.q64) as u32);
                    ext_base_mla(&mut a2GJ_alpha[4*idx2..4*(idx2+1)], &tmp, ((a2[i] as u64 * gv2)%params.q64) as u32);
                    ext_base_mla(&mut a3GJ_alpha[4*idx2..4*(idx2+1)], &tmp, ((a3[i] as u64 * gv2)%params.q64) as u32);
                }
            }
        }
    }

    let mbundle: MBundle = (
        D_alpha,                       // 0
        DJ_alpha,                      // 1
        E0_alpha,                      // 2
        E1_alpha,                      // 3
        E2_alpha,                      // 4
        E3_alpha,                      // 5
        a0GJ_alpha,                    // 6
        a1GJ_alpha,                    // 7
        a2GJ_alpha,                    // 8
        a3GJ_alpha,                    // 9
        b0G_alpha,                     // 10
        b1G_alpha,                     // 11
        b2G_alpha,                     // 12
        b3G_alpha,                     // 13
        cG_alpha,                      // 14
        alpha_vec[4096..4100].to_vec(), // 15
    );
    
    // let duration = start.elapsed();
    // println!("M_alpha computed: {:?}", duration);
    // let start = Instant::now();

    
    // y(alpha)
    let mut v_alpha = vec![0u32; 1<<2];
    let mut u_alpha  = vec![0u32; 1<<2];
    let mut u0_alpha = vec![0u32; 1<<2];
    let mut u1_alpha = vec![0u32; 1<<2];
    let mut u2_alpha = vec![0u32; 1<<2];
    let mut u3_alpha = vec![0u32; 1<<2];

    unsafe {
        for j in 0..params.n{
            ext_base_mla(&mut v_alpha,  &alpha_vec[4*j..4*(j+1)],  v[j]);
            ext_base_mla(&mut u_alpha,  &alpha_vec[4*j..4*(j+1)],  commitment.u()[j]);
            ext_base_mla(&mut u0_alpha, &alpha_vec[4*j..4*(j+1)], u0[j]);
            ext_base_mla(&mut u1_alpha, &alpha_vec[4*j..4*(j+1)], u1[j]);
            ext_base_mla(&mut u2_alpha, &alpha_vec[4*j..4*(j+1)], u2[j]);
            ext_base_mla(&mut u3_alpha, &alpha_vec[4*j..4*(j+1)], u3[j]);
        }
    }

    // let duration = start.elapsed();
    // println!("y(alpha) computed: {:?}", duration);
    // let start = Instant::now();

    // MLE
    let (tbl_tau0_low, tbl_tau0_high) = unsafe { build_eq_tbl_split(tau_0) };
    let tbl_tau1 = unsafe { build_eq_tbl(tau_1) };

    // let duration = start.elapsed();
    // println!("eq_table computed: {:?}", duration);
    // let start = Instant::now();

    let mut a = [0u32; 4];
    unsafe {
        extmla(&mut a, &tbl_tau1[4], &v_alpha);
        extmla(&mut a, &tbl_tau1[5], &u_alpha);
        extmla(&mut a, &tbl_tau1[6], &u0_alpha);
        extmla(&mut a, &tbl_tau1[7], &u1_alpha);
        extmla(&mut a, &tbl_tau1[8], &u2_alpha);
        extmla(&mut a, &tbl_tau1[9], &u3_alpha);
    }

    // let duration = start.elapsed();
    // println!("a computed: {:?}", duration);
    // let start = Instant::now();

    let total_rings = 9 * (1 << 14) + 64 + params.constraints;

    let mut M_table = vec![[0u32; 4]; 1 << 18];
    for x in 0..total_rings {
        let mut sum = [0u32; 4];
        for i in 0..params.constraints {
            let m_val = unsafe { mle_m(i as u32, x as u32, &mbundle, &params) };
            let mut term = [0u32; 4];
            unsafe { extmul(&mut term, &tbl_tau1[i as usize], &m_val) };
            sum = unsafe { extadd(&sum, &term) };
        }
        M_table[x as usize] = sum;
    }

    // let duration = start.elapsed();
    // println!("M_table computed: {:?}", duration);
    // let start = Instant::now();

    let mut W_compact: Vec<u32> = vec![0u32; 1 << 28];
    for u in 0..total_rings {
        for l in 0..1024 {
            W_compact[u as usize * 1024 + l as usize] = unsafe { mle_w(u as u32, l as u32, &wbundle, &r) as u32 };
        }
    }
    
    // let duration = start.elapsed();
    // println!("W_table computed: {:?}", duration);
    // let start = Instant::now();

    let mut alpha_vec_ext = Vec::<[u32; 4]>::with_capacity(1024);
    for i in 0..1024 {
        alpha_vec_ext.push(alpha_vec[4*i..4*(i+1)].try_into().unwrap());
    }

    // let duration = start.elapsed();
    // println!("alpha_table computed: {:?}", duration);
    // let start = Instant::now();

    let mut ind_pos_compact = vec![0u8; 1 << 28];
    let mut ind_bal_compact = vec![0u8; 1 << 28];
    
    let pos_len = 5 * (1 << 14) * 1024;
    let bal_len = 4 * (1 << 14) * 1024;
    let k_len   = 4 * 16 * 1024;

    ind_pos_compact[0..pos_len].fill(1u8);
    ind_bal_compact[pos_len..(pos_len + bal_len)].fill(1u8);
    let k_offset = pos_len + bal_len;
    ind_pos_compact[k_offset..(k_offset + k_len)].fill(1u8);

    // let duration = start.elapsed();
    // println!("ind_tables computed: {:?}", duration);
    // let start = Instant::now();

    // Prover
    let proof = unsafe { 
        sumcheck_prove(&W_compact, M_table, alpha_vec_ext, tbl_tau0_low, tbl_tau0_high, &ind_pos_compact, &ind_bal_compact, &tau_0) 
    };

    // let duration = start.elapsed();
    // println!("sumcheck_prove computed: {:?}", duration);
    // let start = Instant::now();

    (proof, a)
}