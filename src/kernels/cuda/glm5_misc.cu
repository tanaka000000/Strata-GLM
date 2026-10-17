// src/kernels/cuda/glm5_misc.cu - see include/strata/kernels/glm5_misc.hpp.
#include "strata/kernels/glm5_misc.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// one warp per output row o, every token: the weight row is read once
__global__ void __launch_bounds__(256) f32_gemv_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                       float* __restrict__ y, int n_in, int n_out, int n_tok, float scale) {
    const int o = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (o >= n_out) return;
    const float* wr = w + (size_t) o * n_in;
    for (int t = 0; t < n_tok; ++t) {
        const float* xt = x + (size_t) t * n_in;
        float acc = 0.f;
        for (int i = lane; i < n_in; i += 32) acc = fmaf(wr[i], xt[i], acc);
        for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        if (lane == 0) y[(size_t) t * n_out + o] = acc * scale;
    }
}

// STRATA_F32_GEMV_FAST=1 (2026-10-17, opt-in; 1-4 tokens): the same warp-per-row sums with two warps per block (the
// router's 288 rows on 144 blocks, not 36), each weight value loaded once for every token, and 8 loads issued before
// they are used.  Each token's FMA chain runs over i = lane, lane + 32, ... in the same order as above, and the same
// xor tree: bitwise f32_gemv_kernel's output.
template <int NT>
__global__ void __launch_bounds__(64) f32_gemv_fast_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                           float* __restrict__ y, int n_in, int n_out, float scale) {
    constexpr int U = 8;
    const int o = blockIdx.x * 2 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (o >= n_out) return;
    const float* wr = w + (size_t) o * n_in;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    int i = lane;
    for (; i + 32 * (U - 1) < n_in; i += 32 * U) {
        float wv[U];
#pragma unroll
        for (int u = 0; u < U; ++u) wv[u] = wr[i + 32 * u];
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int u = 0; u < U; ++u) acc[t] = fmaf(wv[u], x[(size_t) t * n_in + i + 32 * u], acc[t]);
    }
    for (; i < n_in; i += 32) {
        const float wv = wr[i];
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = fmaf(wv, x[(size_t) t * n_in + i], acc[t]);
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float a = acc[t];
        for (int s = 16; s > 0; s >>= 1) a += __shfl_xor_sync(0xffffffffu, a, s);
        if (lane == 0) y[(size_t) t * n_out + o] = a * scale;
    }
}

__global__ void transpose_abd_kernel(const float* __restrict__ src, float* __restrict__ dst, int a, int b, int d) {
    const int ia = blockIdx.y, ib = blockIdx.z;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < d; k += gridDim.x * blockDim.x)
        dst[((size_t) ib * a + ia) * d + k] = src[((size_t) ia * b + ib) * d + k];
}

__global__ void pack3_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                             float* __restrict__ dst, int n) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float* o = dst + (size_t) t * 3 * n;
        o[i] = q[(size_t) t * n + i];
        o[n + i] = k[(size_t) t * n + i];
        o[2 * n + i] = v[(size_t) t * n + i];
    }
}

__global__ void latent_store_kernel(const float* __restrict__ kv, uint16_t* __restrict__ pool,
                                    const int32_t* __restrict__ page_table, const int32_t* __restrict__ steps,
                                    int step_count, int page_size, int dim) {
    const int t = blockIdx.x;
    const int cell = steps[t * step_count + 0];
    const long long page = page_table[cell / page_size];
    if (page < 0) return;
    uint16_t* row = pool + (size_t) (page * page_size + cell % page_size) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) row[i] = __half_as_ushort(__float2half(kv[(size_t) t * dim + i]));
}

// q4_0 as ggml's quantize_row_q4_0_ref: per 32 values the one of largest magnitude m, d = m / -8, q = min(15,
// (int)(x / d + 8.5)), value i < 16 in the low nibble of qs[i], i >= 16 in the high nibble of qs[i - 16]
struct Q4Block { uint16_t d; uint8_t qs[16]; };
static_assert(sizeof(Q4Block) == 18, "q4_0 block");
__global__ void latent_store_q4_kernel(const float* __restrict__ kv, uint8_t* __restrict__ pool,
                                       const int32_t* __restrict__ page_table, const int32_t* __restrict__ steps,
                                       int step_count, int page_size, int dim) {
    const int t = blockIdx.x, b = threadIdx.x, nb = dim / 32;
    if (b >= nb) return;
    const int cell = steps[t * step_count + 0];
    const long long page = page_table[cell / page_size];
    if (page < 0) return;
    const float* x = kv + (size_t) t * dim + (size_t) b * 32;
    float amax = 0.f, m = 0.f;
    for (int i = 0; i < 32; ++i) { const float v = x[i]; if (fabsf(v) > amax) { amax = fabsf(v); m = v; } }
    const float d = m / -8.0f, id = d != 0.f ? 1.0f / d : 0.f;
    Q4Block* blk = reinterpret_cast<Q4Block*>(pool + (size_t) (page * page_size + cell % page_size) * (size_t) (nb * 18)) + b;
    blk->d = __half_as_ushort(__float2half(d));
    for (int j = 0; j < 16; ++j) {
        const int q0 = min(15, (int) (x[j] * id + 8.5f)), q1 = min(15, (int) (x[16 + j] * id + 8.5f));
        blk->qs[j] = (uint8_t) (q0 | (q1 << 4));
    }
}

}  // namespace

void glm5_latent_store_q4(const float* kv, uint8_t* pool, const int32_t* page_table, const int32_t* steps, int step_count,
                          int page_size, int dim, int n_tok, void* stream) {
    latent_store_q4_kernel<<<(unsigned) n_tok, 32, 0, (cudaStream_t) stream>>>(kv, pool, page_table, steps, step_count,
                                                                              page_size, dim);
    check("glm5_latent_store_q4");
}

void glm5_f32_gemv(const float* w, const float* x, float* y, int n_in, int n_out, int n_tok, float scale, void* stream) {
    static const bool fast = [] { const char* v = std::getenv("STRATA_F32_GEMV_FAST"); return v && v[0] == '1'; }();
    if (fast && n_tok >= 1 && n_tok <= 4) {
        const unsigned g = (unsigned) ((n_out + 1) / 2);
        cudaStream_t s = (cudaStream_t) stream;
        switch (n_tok) {
            case 1: f32_gemv_fast_kernel<1><<<g, 64, 0, s>>>(w, x, y, n_in, n_out, scale); break;
            case 2: f32_gemv_fast_kernel<2><<<g, 64, 0, s>>>(w, x, y, n_in, n_out, scale); break;
            case 3: f32_gemv_fast_kernel<3><<<g, 64, 0, s>>>(w, x, y, n_in, n_out, scale); break;
            default: f32_gemv_fast_kernel<4><<<g, 64, 0, s>>>(w, x, y, n_in, n_out, scale); break;
        }
        check("glm5_f32_gemv");
        return;
    }
    f32_gemv_kernel<<<(unsigned) ((n_out + 7) / 8), 256, 0, (cudaStream_t) stream>>>(w, x, y, n_in, n_out, n_tok, scale);
    check("glm5_f32_gemv");
}

void glm5_transpose_abd(const float* src, float* dst, int a, int b, int d, void* stream) {
    transpose_abd_kernel<<<dim3((unsigned) ((d + 255) / 256), (unsigned) a, (unsigned) b), 256, 0, (cudaStream_t) stream>>>(
        src, dst, a, b, d);
    check("glm5_transpose_abd");
}

void glm5_pack3(const float* q, const float* k, const float* v, float* dst, int n, int n_tok, void* stream) {
    pack3_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) n_tok), 256, 0, (cudaStream_t) stream>>>(q, k, v, dst, n);
    check("glm5_pack3");
}

void glm5_latent_store(const float* kv, uint16_t* pool, const int32_t* page_table, const int32_t* steps, int step_count,
                       int page_size, int dim, int n_tok, void* stream) {
    latent_store_kernel<<<(unsigned) n_tok, 256, 0, (cudaStream_t) stream>>>(kv, pool, page_table, steps, step_count,
                                                                            page_size, dim);
    check("glm5_latent_store");
}

// the MMQ experts' SwiGLU, in place: row r's first n_ff of its 2 * n_ff gate|up values become silu(gate) * up, with
// GLM's clamp when limit > 0 (gate <= limit, |up| <= limit; Project Maya's swiglu_rows, MIT)
static __global__ void swiglu_rows_kernel(float* gu, int64_t rows, int n_ff, float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff;
    const int c = (int) (i - r * n_ff);
    float* row = gu + r * 2 * n_ff;
    float g = row[c], u = row[n_ff + c];
    if (limit > 0.f) { g = fminf(g, limit); u = fminf(fmaxf(u, -limit), limit); }
    row[c] = (g / (1.0f + expf(-g))) * u;
}

static __global__ void swiglu2_kernel(float* g, const float* __restrict__ u, int64_t n, float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = g[i], b = u[i];
    if (limit > 0.f) { a = fminf(a, limit); b = fminf(fmaxf(b, -limit), limit); }
    g[i] = (a / (1.0f + expf(-a))) * b;
}

void glm5_swiglu2(float* g, const float* u, int64_t n, float limit, void* stream) {
    if (n <= 0) return;
    swiglu2_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(g, u, n, limit);
    check("glm5_swiglu2");
}

void glm5_swiglu_rows(float* gu, int64_t rows, int n_ff, float limit, void* stream) {
    if (rows <= 0) return;
    const int64_t n = rows * n_ff;
    swiglu_rows_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(gu, rows, n_ff, limit);
    check("glm5_swiglu_rows");
}

static __global__ void copy_rows_kernel(const float* __restrict__ src, const int32_t* __restrict__ idx,
                                        float* __restrict__ dst, int n_embd) {
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n_embd) dst[(size_t) idx[r] * n_embd + c] = src[(size_t) r * n_embd + c];
}

void glm5_copy_rows(const float* src, const int32_t* idx, float* dst, int n_rows, int n_embd, void* stream) {
    if (n_rows <= 0) return;
    copy_rows_kernel<<<dim3((unsigned) ((n_embd + 255) / 256), (unsigned) n_rows), 256, 0, (cudaStream_t) stream>>>(
        src, idx, dst, n_embd);
    check("glm5_copy_rows");
}

static __global__ void set2_kernel(int32_t* dst, int32_t a, int32_t b) { dst[0] = a; dst[1] = b; }

void glm5_set2(int32_t* dst, int32_t a, int32_t b, void* stream) {
    set2_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(dst, a, b);
    check("glm5_set2");
}

// the layer walk's combine: token u_tok[u] += sum over its entries j (in order) of e_w[j] * rows[e_row[j]]
static __global__ void scatter_add_kernel(const float* __restrict__ rows, const int32_t* __restrict__ u_tok,
                                          const int32_t* __restrict__ u_off, const int32_t* __restrict__ e_row,
                                          const float* __restrict__ e_w, float* __restrict__ acc, int n_embd) {
    const int u = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    float* a = acc + (size_t) u_tok[u] * n_embd + col;
    float v = *a;
    for (int j = u_off[u]; j < u_off[u + 1]; ++j) v += rows[(size_t) e_row[j] * n_embd + col] * e_w[j];
    *a = v;
}

void glm5_scatter_add(const float* rows, const int32_t* u_tok, const int32_t* u_off, const int32_t* e_row, const float* e_w,
                      int n_u, float* acc, int n_embd, void* stream) {
    if (n_u <= 0) return;
    scatter_add_kernel<<<dim3((unsigned) ((n_embd + 255) / 256), (unsigned) n_u), 256, 0, (cudaStream_t) stream>>>(
        rows, u_tok, u_off, e_row, e_w, acc, n_embd);
    check("glm5_scatter_add");
}

}  // namespace strata::kernels
