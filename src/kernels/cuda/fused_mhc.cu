// src/kernels/cuda/fused_mhc.cu - see include/strata/kernels/fused_gr.hpp (glm5-next mHC section).
//
// GLM-5.3-Flash's hyper-connection (mHC) in the shape of Strata's GR read (fused_gr.cu): the previous half's write
// folded into the read, the weights read once for every token of a verify window, and the projection split over
// many blocks that write partial dots (the v3 idea).  The projection is llama.cpp's arithmetic (rms_norm, then a
// Q8_0 x Q8_1 dot of the normalised activation), so the coefficients match its CUDA backend, not just the maths.
//
//   mhc_down (grid SL slices of the 4*n_embd vector):
//       R'   = apply ? out_prev*post_prev[d] + sum_s comb_prev[d,s]*R[s]  : R      (computed on the fly, not stored)
//       inv  = rsqrt(mean(R'^2) + eps) over all 4*n_embd (each block reduces it)
//       part[t][slice][r] = hc_fn[r, slice] . q8_1(R'[slice] * inv)
//   mhc_up (one block per token):
//       mixes = sum_slices part,  pre/post/comb (Sinkhorn) as llama.cpp b11461 glm5-next.cpp:445-501
//       R_out <- R' (when apply),  block = sum_h pre[h]*R'[h],  y = rms_norm(block) * w_norm   (attn_norm / ffn_norm)
//
// R' cannot be written by mhc_down (comb mixes the four streams: another block may still read the old R), so
// mhc_up writes it; it owns its token's columns, so R_out == R is fine there.
// Checked against tools/glm5_ref (which matches the llama.cpp dump) by tests/cuda/glm5_mhc_test.cu.
#include "strata/kernels/fused_gr.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int HC = 4;
constexpr int MIX = (2 + HC) * HC;          // 24
constexpr int QK = 32;                      // Q8_0 block
constexpr int SLICE = 1024;                 // floats of R' per down block (a quarter of one 4096 stream)
constexpr int DOWN_THREADS = 256;           // 8 warps x 3 rows = 24 rows
constexpr int UP_THREADS = 512;

struct Q8_0 { __half d; int8_t qs[QK]; };
static_assert(sizeof(Q8_0) == 34, "Q8_0 block");

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }

// R' for stream d, column i of token t.
__device__ __forceinline__ float fold(const MhcMulti& m, int t, int d, int i) {
    const float* R = m.R[t];
    if (!m.apply) return R[d * m.n_embd + i];
    const float* c = m.comb_prev + t * HC * HC + d * HC;
    float v = m.out_prev[t][i] * m.post_prev[t * HC + d];
#pragma unroll
    for (int s = 0; s < HC; ++s) v += c[s] * R[s * m.n_embd + i];
    return v;
}

// FAST (STRATA_MHC_FAST=1, 2026-10-17, opt-in): the 24 rows split over 3 blocks per slice (gridDim.y; one row per warp
// instead of three) and a pad float after each 32-value block of xs, so a warp's lanes (one block each) read 32
// different banks instead of one.  Every (token, slice, row) partial is computed by the same lanes in the same order,
// and every block reduces the same sum of squares the same way: the outputs are bitwise the default kernel's.
template <bool FAST>
__global__ void __launch_bounds__(DOWN_THREADS) mhc_down_kernel(MhcMulti m) {
    // llama.cpp's arithmetic for hc_fn: rms_norm over all 4*n_embd values, then a Q8_0 x Q8_1 dot (the normalised
    // activation quantised per 32 values: d = amax / 127, q = round(x / d), as ggml-cuda's quantize_q8_1).  Every
    // block needs the token's whole sum of squares first, so each block reduces it itself (16 blocks x 64 KB per
    // token from L2) instead of a third launch.
    constexpr int XQ = FAST ? QK + 1 : QK;   // floats from one 32-value block of xs to the next
    __shared__ float xs[kFusedGrMaxT][SLICE / QK * XQ];
    __shared__ float red[DOWN_THREADS / 32];
    __shared__ float inv_rms[kFusedGrMaxT];
    const int slice = blockIdx.x;
    const int base = slice * SLICE;          // offset in the 4*n_embd vector
    const int d = base / m.n_embd, i0 = base % m.n_embd;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int D = HC * m.n_embd;

    for (int t = 0; t < m.n_tok; ++t) {
        float ss = 0.f;
        for (int j = threadIdx.x; j < D; j += DOWN_THREADS) {
            const float v = fold(m, t, j / m.n_embd, j % m.n_embd);
            ss += v * v;
        }
        ss = warp_sum(ss);
        if (lane == 0) red[wid] = ss;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.f;
            for (int w = 0; w < DOWN_THREADS / 32; ++w) s += red[w];
            inv_rms[t] = rsqrtf(s / (float)D + m.rms_eps);
            if (slice == 0 && blockIdx.y == 0) m.ss_part[t * m.n_slices] = s;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < SLICE; j += DOWN_THREADS) xs[t][j / QK * XQ + j % QK] = fold(m, t, d, i0 + j) * inv_rms[t];
    }
    __syncthreads();

    // 3 rows per warp (FAST: one - block y takes rows 8y..8y+7); lane = one 32-value block of the slice (1024 / 32 =
    // 32 blocks); each weight block read once and used for every token.  The activation block is quantised here
    // exactly once per (token, block) per row pass - the same codes for every row.
    const int nb_row = HC * m.n_embd / QK;
    const int r_end = FAST ? (int(blockIdx.y) + 1) * (DOWN_THREADS / 32) : MIX;
    for (int r = (FAST ? int(blockIdx.y) * (DOWN_THREADS / 32) : 0) + wid; r < r_end; r += DOWN_THREADS / 32) {
        const Q8_0 blk = static_cast<const Q8_0*>(m.fn)[(size_t)r * nb_row + base / QK + lane];
        const float dw = __half2float(blk.d);
        for (int t = 0; t < m.n_tok; ++t) {
            const float* x = xs[t] + lane * XQ;
            float amax = 0.f;
#pragma unroll
            for (int k = 0; k < QK; ++k) amax = fmaxf(amax, fabsf(x[k]));
            const float dx = amax / 127.0f;
            int sumi = 0;
            if (amax != 0.0f) {
#pragma unroll
                for (int k = 0; k < QK; ++k) sumi += (int)blk.qs[k] * (int)roundf(x[k] / dx);
            }
            float acc = dw * (dx * (float)sumi);
            acc = warp_sum(acc);
            if (lane == 0) m.part[((size_t)t * m.n_slices + slice) * MIX + r] = acc;
        }
    }
}

// FAST (STRATA_MHC_FAST=1): the slice partials loaded 8 at a time before they are added (in the same order), so the 16
// loads are not one dependent chain of L2 round trips; bitwise the default.
template <bool FAST>
__global__ void __launch_bounds__(UP_THREADS) mhc_up_kernel(MhcMulti m) {
    __shared__ float coef[HC + HC + HC * HC];   // pre, post, comb[d*HC + s]
    __shared__ float red[UP_THREADS / 32];
    __shared__ float inv_y;
    const int t = blockIdx.x;
    const int n = m.n_embd;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;

    __shared__ float mixes[MIX + 1];   // [MIX] = sum of squares
    // the slice partials: one thread per row (and one for the squares), each summed in slice order
    if (threadIdx.x < MIX) {   // the slices' partial dots (already of the normalised activation), in slice order
        float v = 0.f;
        int s = 0;
        if constexpr (FAST) {
            for (; s + 8 <= m.n_slices; s += 8) {
                float p[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) p[k] = m.part[((size_t)t * m.n_slices + s + k) * MIX + threadIdx.x];
#pragma unroll
                for (int k = 0; k < 8; ++k) v += p[k];
            }
        }
        for (; s < m.n_slices; ++s) v += m.part[((size_t)t * m.n_slices + s) * MIX + threadIdx.x];
        mixes[threadIdx.x] = v;
        if (m.mixes_out) m.mixes_out[t * MIX + threadIdx.x] = v;
    }
    __syncthreads();
    // pre/post and the Sinkhorn on warp 0: lane k < 16 holds comb element k = dst + 4*src (llama.cpp's layout).
    // Every sum is taken in llama.cpp's order (eps first, then ascending index) by reading the 4 lanes in turn.
    if (wid == 0) {
        if (lane < HC) {
            coef[lane] = sigm(mixes[lane] * m.scale[0] + m.base[lane]) + m.hc_eps;
            coef[HC + lane] = 2.f * sigm(mixes[HC + lane] * m.scale[1] + m.base[HC + lane]);
        }
        const int k = lane & (HC * HC - 1), d = k % HC, s = k / HC;
        float c = mixes[2 * HC + k] * m.scale[2] + m.base[2 * HC + k];
        float mx = -3.402823466e+38f;
#pragma unroll
        for (int j = 0; j < HC; ++j) mx = fmaxf(mx, __shfl_sync(0xffffffffu, c, s * HC + j));
        c = expf(c - mx);
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < HC; ++j) sum += __shfl_sync(0xffffffffu, c, s * HC + j);   // softmax over dst
        c = c * (1.f / sum) + m.hc_eps;
        for (int it = 0; it < m.iters; ++it) {
            if (it > 0) {   // rows: per src, over dst
                float r = m.hc_eps;
#pragma unroll
                for (int j = 0; j < HC; ++j) r += __shfl_sync(0xffffffffu, c, s * HC + j);
                c *= 1.f / r;
            }
            float q = m.hc_eps;   // columns: per dst, over src
#pragma unroll
            for (int j = 0; j < HC; ++j) q += __shfl_sync(0xffffffffu, c, d + HC * j);
            c *= 1.f / q;
        }
        if (lane < HC * HC) coef[2 * HC + d * HC + s] = c;
    }
    __syncthreads();

    float* y = m.y[t];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += UP_THREADS) {
        float r[HC];
#pragma unroll
        for (int d = 0; d < HC; ++d) r[d] = fold(m, t, d, i);
        if (m.apply) {   // this thread owns column i of every stream: all four were read above
#pragma unroll
            for (int d = 0; d < HC; ++d) m.R_out[t][d * n + i] = r[d];
        }
        float b = 0.f;
#pragma unroll
        for (int h = 0; h < HC; ++h) b = fmaf(coef[h], r[h], b);
        y[i] = b;
        ss += b * b;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int w = 0; w < UP_THREADS / 32; ++w) s += red[w];
        inv_y = rsqrtf(s / (float)n + m.rms_eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += UP_THREADS) y[i] = y[i] * inv_y * m.w_norm[i];
    if (threadIdx.x < HC) m.post_out[t * HC + threadIdx.x] = coef[HC + threadIdx.x];
    if (threadIdx.x < HC * HC) m.comb_out[t * HC * HC + threadIdx.x] = coef[2 * HC + threadIdx.x];
    if (m.pre_out && threadIdx.x < HC) m.pre_out[t * HC + threadIdx.x] = coef[threadIdx.x];
}

// Last layer: the plain write (nothing follows to fold it into).  res[t] (in place): [HC][n_embd].
__global__ void mhc_write_kernel(MhcMulti m) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.n_embd) return;
    float r[HC];
#pragma unroll
    for (int d = 0; d < HC; ++d) r[d] = fold(m, t, d, i);
#pragma unroll
    for (int d = 0; d < HC; ++d) m.R_out[t][d * m.n_embd + i] = r[d];
}

__global__ void mhc_exit_mean_kernel(MhcMulti m) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.n_embd) return;
    const float* R = m.R[t];
    float v = 0.f;
#pragma unroll
    for (int s = 0; s < HC; ++s) v += R[s * m.n_embd + i];
    m.y[t][i] = v * (1.f / HC);
}

// ---- the prompt path's batched read (2026-10-09): mhc_down / mhc_up per token, the tokens at fixed strides ----
__device__ __forceinline__ float fold_b(const MhcBatch& m, int t, int d, int i) {
    const float* R = m.R + (size_t)t * HC * m.n_embd;
    if (!m.apply) return R[d * m.n_embd + i];
    const float* c = m.comb_prev + t * HC * HC + d * HC;
    float v = m.out_prev[(size_t)t * m.n_embd + i] * m.post_prev[t * HC + d];
#pragma unroll
    for (int s = 0; s < HC; ++s) v += c[s] * R[s * m.n_embd + i];
    return v;
}

// mhc_down_kernel for one (slice, token): grid (n_slices, n_tok)
__global__ void __launch_bounds__(DOWN_THREADS) mhc_down_batch_kernel(MhcBatch m) {
    __shared__ float xs[SLICE];
    __shared__ float red[DOWN_THREADS / 32];
    __shared__ float inv_rms;
    const int slice = blockIdx.x, t = blockIdx.y;
    const int base = slice * SLICE;
    const int d = base / m.n_embd, i0 = base % m.n_embd;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int D = HC * m.n_embd;
    float ss = 0.f;
    for (int j = threadIdx.x; j < D; j += DOWN_THREADS) {
        const float v = fold_b(m, t, j / m.n_embd, j % m.n_embd);
        ss += v * v;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int w = 0; w < DOWN_THREADS / 32; ++w) s += red[w];
        inv_rms = rsqrtf(s / (float)D + m.rms_eps);
        if (slice == 0) m.ss_part[(size_t)t * m.n_slices] = s;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < SLICE; j += DOWN_THREADS) xs[j] = fold_b(m, t, d, i0 + j) * inv_rms;
    __syncthreads();
    const int nb_row = HC * m.n_embd / QK;
    for (int r = wid; r < MIX; r += DOWN_THREADS / 32) {
        const Q8_0 blk = static_cast<const Q8_0*>(m.fn)[(size_t)r * nb_row + base / QK + lane];
        const float dw = __half2float(blk.d);
        const float* x = xs + lane * QK;
        float amax = 0.f;
#pragma unroll
        for (int k = 0; k < QK; ++k) amax = fmaxf(amax, fabsf(x[k]));
        const float dx = amax / 127.0f;
        int sumi = 0;
        if (amax != 0.0f) {
#pragma unroll
            for (int k = 0; k < QK; ++k) sumi += (int)blk.qs[k] * (int)roundf(x[k] / dx);
        }
        float acc = dw * (dx * (float)sumi);
        acc = warp_sum(acc);
        if (lane == 0) m.part[((size_t)t * m.n_slices + slice) * MIX + r] = acc;
    }
}

// mhc_up_kernel, the tokens at fixed strides
__global__ void __launch_bounds__(UP_THREADS) mhc_up_batch_kernel(MhcBatch m) {
    __shared__ float coef[HC + HC + HC * HC];
    __shared__ float red[UP_THREADS / 32];
    __shared__ float inv_y;
    const int t = blockIdx.x;
    const int n = m.n_embd;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __shared__ float mixes[MIX + 1];
    if (threadIdx.x < MIX) {
        float v = 0.f;
        for (int s = 0; s < m.n_slices; ++s) v += m.part[((size_t)t * m.n_slices + s) * MIX + threadIdx.x];
        mixes[threadIdx.x] = v;
    }
    __syncthreads();
    if (wid == 0) {
        if (lane < HC) {
            coef[lane] = sigm(mixes[lane] * m.scale[0] + m.base[lane]) + m.hc_eps;
            coef[HC + lane] = 2.f * sigm(mixes[HC + lane] * m.scale[1] + m.base[HC + lane]);
        }
        const int k = lane & (HC * HC - 1), d = k % HC, s = k / HC;
        float c = mixes[2 * HC + k] * m.scale[2] + m.base[2 * HC + k];
        float mx = -3.402823466e+38f;
#pragma unroll
        for (int j = 0; j < HC; ++j) mx = fmaxf(mx, __shfl_sync(0xffffffffu, c, s * HC + j));
        c = expf(c - mx);
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < HC; ++j) sum += __shfl_sync(0xffffffffu, c, s * HC + j);
        c = c * (1.f / sum) + m.hc_eps;
        for (int it = 0; it < m.iters; ++it) {
            if (it > 0) {
                float r = m.hc_eps;
#pragma unroll
                for (int j = 0; j < HC; ++j) r += __shfl_sync(0xffffffffu, c, s * HC + j);
                c *= 1.f / r;
            }
            float q = m.hc_eps;
#pragma unroll
            for (int j = 0; j < HC; ++j) q += __shfl_sync(0xffffffffu, c, d + HC * j);
            c *= 1.f / q;
        }
        if (lane < HC * HC) coef[2 * HC + d * HC + s] = c;
    }
    __syncthreads();
    float* y = m.y + (size_t)t * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += UP_THREADS) {
        float r[HC];
#pragma unroll
        for (int d = 0; d < HC; ++d) r[d] = fold_b(m, t, d, i);
        if (m.apply) {
#pragma unroll
            for (int d = 0; d < HC; ++d) m.R_out[(size_t)t * HC * n + d * n + i] = r[d];
        }
        float b = 0.f;
#pragma unroll
        for (int h = 0; h < HC; ++h) b = fmaf(coef[h], r[h], b);
        y[i] = b;
        ss += b * b;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int w = 0; w < UP_THREADS / 32; ++w) s += red[w];
        inv_y = rsqrtf(s / (float)n + m.rms_eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += UP_THREADS) y[i] = y[i] * inv_y * m.w_norm[i];
    if (threadIdx.x < HC) m.post_out[t * HC + threadIdx.x] = coef[HC + threadIdx.x];
    if (threadIdx.x < HC * HC) m.comb_out[t * HC * HC + threadIdx.x] = coef[2 * HC + threadIdx.x];
}

}  // namespace

void fused_mhc_read_batch(const MhcBatch& m, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    mhc_down_batch_kernel<<<dim3((unsigned)m.n_slices, (unsigned)m.n_tok), DOWN_THREADS, 0, st>>>(m);
    mhc_up_batch_kernel<<<(unsigned)m.n_tok, UP_THREADS, 0, st>>>(m);
}

bool fused_mhc_supported(int64_t n_embd, int64_t hc, int64_t mix_rows) {
    return hc == HC && mix_rows == MIX && n_embd > 0 && (HC * n_embd) % SLICE == 0 && n_embd % SLICE == 0;
}

int fused_mhc_slices(int64_t n_embd) { return (int)(HC * n_embd / SLICE); }

void fused_mhc_read_multi(const MhcMulti& m, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    // STRATA_MHC_FAST=1 (2026-10-17, opt-in): the FAST kernels above (bitwise the default).  Read once; set before capture.
    static const bool fast = [] { const char* v = std::getenv("STRATA_MHC_FAST"); return v && v[0] == '1'; }();
    if (fast) {
        static_assert(MIX % (DOWN_THREADS / 32) == 0, "FAST: whole row groups");
        mhc_down_kernel<true><<<dim3((unsigned)m.n_slices, MIX / (DOWN_THREADS / 32)), DOWN_THREADS, 0, st>>>(m);
        mhc_up_kernel<true><<<m.n_tok, UP_THREADS, 0, st>>>(m);
        return;
    }
    mhc_down_kernel<false><<<m.n_slices, DOWN_THREADS, 0, st>>>(m);
    mhc_up_kernel<false><<<m.n_tok, UP_THREADS, 0, st>>>(m);
}

void fused_mhc_write_multi(const MhcMulti& m, void* stream) {
    dim3 g((m.n_embd + 255) / 256, m.n_tok);
    mhc_write_kernel<<<g, 256, 0, (cudaStream_t)stream>>>(m);
}

void fused_mhc_exit_mean(const MhcMulti& m, void* stream) {
    dim3 g((m.n_embd + 255) / 256, m.n_tok);
    mhc_exit_mean_kernel<<<g, 256, 0, (cudaStream_t)stream>>>(m);
}

}  // namespace strata::kernels
