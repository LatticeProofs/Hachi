use std::ops::Range;

use bytemuck::cast_slice;
use memmap2::Mmap;

use crate::utils::ds::*;
use crate::prep::commit::*;
use crate::hachi::setup::SetupParams;

/// A commitment (`u`, `r`, `t`) backed either by owned aligned buffers or by a
/// memory-mapped commitment file.
///
/// Consumers access the three vectors via [`Commitment::u`], [`Commitment::r`],
/// and [`Commitment::t`], which return `&[u32]` regardless of the underlying
/// storage. This lets the prover read directly from the mmap with zero copy.
pub struct Commitment {
    storage: CommitmentStorage,
}

enum CommitmentStorage {
    Owned {
        u: AlignedU32Vec,
        r: AlignedU32Vec,
        t: AlignedU32Vec,
    },
    /// `u_bytes`, `r_bytes`, `t_bytes` are byte ranges into `map` covering the
    /// raw little-endian `u32` payloads. They are 4-byte aligned by virtue of
    /// the file layout (header is a multiple of 4 bytes).
    Mmap {
        map: Mmap,
        u_bytes: Range<usize>,
        r_bytes: Range<usize>,
        t_bytes: Range<usize>,
    },
}

impl Commitment {
    /// Build an owned commitment from three aligned `u32` vectors.
    pub fn new(u: AlignedU32Vec, r: AlignedU32Vec, t: AlignedU32Vec) -> Self {
        Self {
            storage: CommitmentStorage::Owned { u, r, t },
        }
    }

    /// Build a commitment that borrows its three vectors from a memory map.
    ///
    /// The byte ranges must each be 4-byte aligned within `map` so that the
    /// payload can be reinterpreted as `[u32]` without copying.
    pub fn from_mmap(
        map: Mmap,
        u_bytes: Range<usize>,
        r_bytes: Range<usize>,
        t_bytes: Range<usize>,
    ) -> Self {
        Self {
            storage: CommitmentStorage::Mmap { map, u_bytes, r_bytes, t_bytes },
        }
    }

    pub fn u(&self) -> &[u32] {
        match &self.storage {
            CommitmentStorage::Owned { u, .. } => u,
            CommitmentStorage::Mmap { map, u_bytes, .. } => cast_slice(&map[u_bytes.clone()]),
        }
    }

    pub fn r(&self) -> &[u32] {
        match &self.storage {
            CommitmentStorage::Owned { r, .. } => r,
            CommitmentStorage::Mmap { map, r_bytes, .. } => cast_slice(&map[r_bytes.clone()]),
        }
    }

    pub fn t(&self) -> &[u32] {
        match &self.storage {
            CommitmentStorage::Owned { t, .. } => t,
            CommitmentStorage::Mmap { map, t_bytes, .. } => cast_slice(&map[t_bytes.clone()]),
        }
    }
}

pub fn Commit(params: &SetupParams, s: &AlignedU8Vec) -> Commitment {
    let total_len = params.height_2 * params.n;
    let mut r = AlignedU32Vec {
        inner: vec![Align64([0u32; 16]); params.n / 16],
        len: params.n,
    };
    let mut t = AlignedU32Vec {
        inner: vec![Align64([0u32; 16]); total_len / 16],
        len: total_len,
    };
    let mut u = AlignedU32Vec {
        inner: vec![Align64([0u32; 16]); params.n / 16],
        len: params.n,
    };
    unsafe {
        commit(&mut u, &mut r[0..params.n], &mut t, s, &params.d);
    }
    Commitment::new(u, r, t)
}