// glm5_batch_test [reps]: the opt-in decode paths of 2026-10-17 (STRATA_GLM5_BATCH_MAXT, STRATA_GLM5_SKIP_T1,
// STRATA_GLM5_MTP_BATCH, STRATA_MMVQ_ROWS) against the paths they replace, on GLM-5.3-Flash's shapes with synthetic
// Q8_0 weights (no files needed).
//   1. bitwise: the table launch (2-8 columns) against the separate native_mmvq calls; the batched per-head launch
//      on [token][head] layouts (strided columns, no transposes) against the per-head calls on the transposed
//      layout; and every column of a multi-column call against a one-column call on that column.
//   2. a hash of every output, so a run with STRATA_MMVQ_ROWS (or STRATA_MMVQ_TABLE_ROWS) = 2 / 4 can be compared with
//      a default run (same lines).  Likewise STRATA_MHC_FAST=1 and STRATA_F32_GEMV_FAST=1 (the mHC read, the router / indexer F32 GEMV).
//   3. speed: each path captured in a CUDA graph (as the engine runs it), weights rotated through ~400 MB so the
//      5090's L2 does not hold them; us per call and the weight bytes / time.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/glm5_misc.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <cuda_runtime.h>

#include <algorithm>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

static cudaStream_t st;
static uint8_t* pool = nullptr;          // synthetic Q8_0 blocks
static size_t pool_bytes = 0;
static int fails = 0, reps = 50;
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t nxt() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t) rng; }

static size_t qbytes(int n_in, int n_out) { return (size_t) n_in / 32 * 34 * n_out; }
static size_t al(size_t x) { const size_t q = 34 * 128; return (x + q - 1) / q * q; }   // whole blocks, 4-byte aligned
static uint8_t* wat(size_t off) { return pool + off; }

static float* fdev(size_t n) {   // random floats in [-1, 1)
    std::vector<float> h(n);
    for (auto& v : h) v = (float) ((int) (nxt() % 20001) - 10000) / 10000.0f;
    float* d; CK(cudaMalloc(&d, n * 4)); CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());   // a pageable copy may still be in flight; st does not wait for it
    return d;
}
static std::vector<float> get(const float* d, size_t n) { std::vector<float> h(n); CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h; }
static uint64_t fnv(const std::vector<float>& v) { uint64_t h = 1469598103934665603ull; const uint8_t* p = (const uint8_t*) v.data(); for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 1099511628211ull; return h; }
static void same(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    const bool ok = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    size_t bad = 0; for (size_t i = 0; i < a.size() && i < b.size(); ++i) bad += std::memcmp(&a[i], &b[i], 4) != 0;
    std::printf("  %-58s %s", what, ok ? "bitwise OK" : "DIFFERS");
    if (!ok) std::printf(" (%zu of %zu values)", bad, a.size());
    std::printf("   hash %016llx\n", (unsigned long long) fnv(a));
    fails += !ok;
}

// time `body(c)` (c = the weight copy) captured `reps` times in one graph; returns us per call
template <class F> static double timed(F body, int copies) {
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int r = 0; r < reps; ++r) body(r % copies);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e30f;
    for (int k = 0; k < 5; ++k) {
        cudaEventRecord(a, st); CK(cudaGraphLaunch(ge, st)); cudaEventRecord(b, st); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
    }
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaEventDestroy(a); cudaEventDestroy(b);
    return best * 1000.0 / reps;
}
static void report(const char* what, double us, size_t bytes) {
    std::printf("  %-58s %8.2f us  %7.0f GB/s\n", what, us, bytes / (us * 1e3));
}

// ---- 1. the table of projections sharing x (KDA: q, k, v, f_a, g_a, beta; DSA: q_a, kv_a, indexer k, gate)
static void table_case(const char* name, int n_in, std::vector<int> outs, int max_n) {
    const int nb = (int) outs.size();
    size_t set = 0; std::vector<size_t> off(nb);
    for (int b = 0; b < nb; ++b) { off[b] = set; set += al(qbytes(n_in, outs[b])); }
    const int copies = (int) std::max<size_t>(1, pool_bytes / set);
    int tot = 0; for (int o : outs) tot += o;
    float* x = fdev((size_t) 8 * n_in);
    void* q8; CK(cudaMalloc(&q8, native_q8_1_bytes(n_in, 8)));
    std::vector<float*> ya(nb), yb(nb);
    for (int b = 0; b < nb; ++b) { CK(cudaMalloc(&ya[b], (size_t) 8 * outs[b] * 4)); CK(cudaMalloc(&yb[b], (size_t) 8 * outs[b] * 4)); }
    std::printf("%s: %d products of %d, %d rows (%.1f MB of weights)\n", name, nb, n_in, tot, set / 1e6);
    for (int n = 1; n <= max_n; ++n) {
        native_quantize_q8_1(x, q8, n_in, n, st);
        const void* wp[8]; int no[8];
        for (int b = 0; b < nb; ++b) { wp[b] = wat(off[b]); no[b] = outs[b]; }
        for (int b = 0; b < nb; ++b) native_mmvq(8, wp[b], q8, ya[b], n_in, outs[b], n, st);
        if (!native_mmvq_table_q8_0(wp, no, nb, q8, yb.data(), n_in, st, n)) { std::printf("  table refused n=%d\n", n); ++fails; continue; }
        CK(cudaStreamSynchronize(st));
        std::vector<float> a, bb;
        for (int b = 0; b < nb; ++b) { auto u = get(ya[b], (size_t) n * outs[b]), v = get(yb[b], (size_t) n * outs[b]); a.insert(a.end(), u.begin(), u.end()); bb.insert(bb.end(), v.begin(), v.end()); }
        char t[128]; std::snprintf(t, sizeof t, "n=%d table vs separate", n); same(t, bb, a);
        if (n > 1) {   // each column against a one-column call on it
            std::vector<float> one;
            for (int j = 0; j < n; ++j) {
                for (int b = 0; b < nb; ++b) native_mmvq(8, wp[b], (const uint8_t*) q8 + (size_t) j * n_in / 32 * 36, ya[b], n_in, outs[b], 1, st);
                CK(cudaStreamSynchronize(st));
                for (int b = 0; b < nb; ++b) { auto u = get(ya[b], outs[b]); auto v = get(yb[b] + (size_t) j * outs[b], outs[b]); if (std::memcmp(u.data(), v.data(), u.size() * 4)) { std::printf("  n=%d column %d member %d differs from a one-column call\n", n, j, b); ++fails; } }
            }
        }
        const double us_sep = timed([&](int c) { for (int b = 0; b < nb; ++b) native_mmvq(8, wat(off[b] + c * set), q8, ya[b], n_in, outs[b], n, st); }, copies);
        const double us_tab = timed([&](int c) { const void* w2[8]; for (int b = 0; b < nb; ++b) w2[b] = wat(off[b] + c * set); native_mmvq_table_q8_0(w2, no, nb, q8, yb.data(), n_in, st, n); }, copies);
        std::snprintf(t, sizeof t, "n=%d separate (%d launches)", n, nb); report(t, us_sep, set);
        std::snprintf(t, sizeof t, "n=%d table (1 launch)", n); report(t, us_tab, set);
    }
    cudaFree(x); cudaFree(q8); for (int b = 0; b < nb; ++b) { cudaFree(ya[b]); cudaFree(yb[b]); }
}

// ---- 2. the MLA per-head products: [t][h][n_in] -> [t][h][n_out], NH heads
static void absorb_case(const char* name, int n_in, int n_out, int NH, int max_n) {
    const size_t set = al(qbytes(n_in, n_out) * NH), row = (size_t) n_in / 32 * 34, qcol = (size_t) n_in / 32 * 36;
    const int copies = (int) std::max<size_t>(1, pool_bytes / set);
    float* src = fdev((size_t) 8 * NH * n_in);      // [t][h][n_in]
    float *srcT, *y0T, *y0, *y1;
    CK(cudaMalloc(&srcT, (size_t) 8 * NH * n_in * 4)); CK(cudaMalloc(&y0T, (size_t) 8 * NH * n_out * 4));
    CK(cudaMalloc(&y0, (size_t) 8 * NH * n_out * 4)); CK(cudaMalloc(&y1, (size_t) 8 * NH * n_out * 4));
    void *q8a, *q8b; CK(cudaMalloc(&q8a, (size_t) 8 * NH * qcol)); CK(cudaMalloc(&q8b, (size_t) 8 * NH * qcol));
    std::printf("%s: %d heads of %d -> %d (%.1f MB of weights)\n", name, NH, n_in, n_out, set / 1e6);
    for (int n = 1; n <= max_n; ++n) {
        const uint8_t* w = wat(0);
        // the default path: transpose, quantize, per head (n columns each), transpose back
        auto ref = [&](const uint8_t* wc) {
            glm5_transpose_abd(src, srcT, n, NH, n_in, st);
            native_quantize_q8_1(srcT, q8a, n_in * NH * n, 1, st);
            for (int h = 0; h < NH; ++h)
                native_mmvq(8, wc + (size_t) h * n_out * row, (const uint8_t*) q8a + (size_t) h * n * qcol, y0T + (size_t) h * n * n_out, n_in, n_out, n, st);
            glm5_transpose_abd(y0T, y0, NH, n, n_out, st);
        };
        // the 2026-10-10 batch on the transposed layout (one token only before; any n now)
        auto bat = [&](const uint8_t* wc) {
            glm5_transpose_abd(src, srcT, n, NH, n_in, st);
            native_quantize_q8_1(srcT, q8a, n_in * NH * n, 1, st);
            native_mmvq_batched_q8_0(wc, (size_t) n_out * row, q8a, n * qcol, y0T, (size_t) n * n_out, n_in, n_out, NH, st, n);
            glm5_transpose_abd(y0T, y1, NH, n, n_out, st);
        };
        // direct: quantize [t][h] as it is, strided columns, the output straight into [t][h]
        auto direct = [&](const uint8_t* wc) {
            native_quantize_q8_1(src, q8b, n_in * NH * n, 1, st);
            native_mmvq_batched_q8_0(wc, (size_t) n_out * row, q8b, qcol, y1, (size_t) n_out, n_in, n_out, NH, st, n,
                                     (size_t) NH * qcol, (size_t) NH * n_out);
        };
        ref(w); CK(cudaStreamSynchronize(st)); auto a = get(y0, (size_t) n * NH * n_out);
        bat(w); CK(cudaStreamSynchronize(st)); auto b = get(y1, a.size());
        char t[128]; std::snprintf(t, sizeof t, "n=%d batched (transposed) vs per head", n); same(t, b, a);
        CK(cudaMemset(y1, 0xff, a.size() * 4));
        direct(w); CK(cudaStreamSynchronize(st)); b = get(y1, a.size());
        std::snprintf(t, sizeof t, "n=%d batched direct [t][h] vs per head", n); same(t, b, a);
        std::snprintf(t, sizeof t, "n=%d per head + 2 transposes (%d launches)", n, NH + 3); report(t, timed([&](int c) { ref(wat(c * set)); }, copies), set);
        std::snprintf(t, sizeof t, "n=%d batched + 2 transposes (4 launches)", n); report(t, timed([&](int c) { bat(wat(c * set)); }, copies), set);
        std::snprintf(t, sizeof t, "n=%d batched direct (2 launches)", n); report(t, timed([&](int c) { direct(wat(c * set)); }, copies), set);
        std::snprintf(t, sizeof t, "n=%d   of which the product alone", n);
        report(t, timed([&](int c) { native_mmvq_batched_q8_0(wat(c * set), (size_t) n_out * row, q8b, qcol, y1, (size_t) n_out, n_in, n_out, NH, st, n, (size_t) NH * qcol, (size_t) NH * n_out); }, copies), set);
        std::snprintf(t, sizeof t, "n=%d   one transpose of [t][h][%d]", n, n_in);
        report(t, timed([&](int) { glm5_transpose_abd(src, srcT, n, NH, n_in, st); }, 1), (size_t) n * NH * n_in * 8);
    }
    cudaFree(src); cudaFree(srcT); cudaFree(y0T); cudaFree(y0); cudaFree(y1); cudaFree(q8a); cudaFree(q8b);
}

// ---- 3. single matrices (native_small_mmvq_kernel): speed, and a hash for the STRATA_MMVQ_ROWS comparison
static void plain_case(const char* name, int n_in, int n_out, int max_n) {
    const size_t set = al(qbytes(n_in, n_out));
    const int copies = (int) std::max<size_t>(1, pool_bytes / set);
    float* x = fdev((size_t) 8 * n_in);
    float* y; CK(cudaMalloc(&y, (size_t) 8 * n_out * 4));
    void* q8; CK(cudaMalloc(&q8, native_q8_1_bytes(n_in, 8)));
    for (int n = 1; n <= max_n; ++n) {
        native_quantize_q8_1(x, q8, n_in, n, st);
        native_mmvq(8, wat(0), q8, y, n_in, n_out, n, st);
        CK(cudaStreamSynchronize(st));
        char t[128]; std::snprintf(t, sizeof t, "%s %d->%d n=%d (%.1f MB)", name, n_in, n_out, n, set / 1e6);
        const auto yv = get(y, (size_t) n * n_out);
        std::printf("  %-58s hash %016llx  y[0..2] %g %g %g\n", t, (unsigned long long) fnv(yv), yv[0], yv[1], yv[2]);
        report("", timed([&](int c) { native_mmvq(8, wat(c * set), q8, y, n_in, n_out, n, st); }, copies), set);
    }
    cudaFree(x); cudaFree(y); cudaFree(q8);
}

// ---- the shared expert with gate / up in one launch (lfuse 6: STRATA_GLM5_SHEXP_PAIR) against the default (lfuse 0)
static void shexp_case(int n_embd, int n_ff, float limit, int max_n) {
    const size_t gb = al(qbytes(n_embd, n_ff)), set = 3 * gb;
    const int copies = (int) std::max<size_t>(1, pool_bytes / set);
    float* x = fdev((size_t) 8 * n_embd);
    float *g, *u, *gs, *o0, *o1;
    CK(cudaMalloc(&g, (size_t) 8 * n_ff * 4)); CK(cudaMalloc(&u, (size_t) 8 * n_ff * 4)); CK(cudaMalloc(&gs, 64));
    CK(cudaMalloc(&o0, (size_t) 8 * n_embd * 4)); CK(cudaMalloc(&o1, (size_t) 8 * n_embd * 4));
    void* q8; CK(cudaMalloc(&q8, native_q8_1_bytes(n_ff > n_embd ? n_ff : n_embd, 8)));
    auto nw_at = [&](int c) {
        NativeSharedWeights nw;
        nw.gate_type = nw.up_type = nw.down_type = 8;
        nw.gate_data = wat(c * set); nw.up_data = wat(c * set + gb); nw.down_data = wat(c * set + 2 * gb); nw.q8_1 = q8;
        return nw;
    };
    std::printf("shared expert %d x %d (limit %g, %.1f MB):\n", n_embd, n_ff, limit, set / 1e6);
    for (int n = 1; n <= max_n; ++n) {
        shared_expert_multi(n, x, nullptr, nw_at(0), nullptr, g, u, gs, o0, n_embd, n_ff, st, nullptr, 0, limit);
        shared_expert_multi(n, x, nullptr, nw_at(0), nullptr, g, u, gs, o1, n_embd, n_ff, st, nullptr, 6, limit);
        CK(cudaStreamSynchronize(st));
        char t[128]; std::snprintf(t, sizeof t, "n=%d gate+up fused vs separate", n);
        same(t, get(o1, (size_t) n * n_embd), get(o0, (size_t) n * n_embd));
        std::snprintf(t, sizeof t, "n=%d default", n);
        report(t, timed([&](int c) { shared_expert_multi(n, x, nullptr, nw_at(c), nullptr, g, u, gs, o0, n_embd, n_ff, st, nullptr, 0, limit); }, copies), set);
        std::snprintf(t, sizeof t, "n=%d gate+up in one launch", n);
        report(t, timed([&](int c) { shared_expert_multi(n, x, nullptr, nw_at(c), nullptr, g, u, gs, o1, n_embd, n_ff, st, nullptr, 6, limit); }, copies), set);
    }
    cudaFree(x); cudaFree(g); cudaFree(u); cudaFree(gs); cudaFree(o0); cudaFree(o1); cudaFree(q8);
}

// ---- 4. the mHC read (STRATA_MHC_FAST) and the F32 router / indexer GEMV (STRATA_F32_GEMV_FAST): hashes and speed
static void mhc_case(int n_tok, bool apply) {
    const int N = 4096, HCn = 4, ns = fused_mhc_slices(N);
    MhcMulti a;
    a.n_tok = n_tok; a.n_embd = N; a.n_slices = ns; a.apply = apply; a.iters = 20; a.rms_eps = 1e-5f; a.hc_eps = 1e-6f;
    float* R = fdev((size_t) n_tok * HCn * N);
    float* Ro; CK(cudaMalloc(&Ro, (size_t) n_tok * HCn * N * 4));
    float* op = fdev((size_t) n_tok * N);
    float* y; CK(cudaMalloc(&y, (size_t) n_tok * N * 4));
    for (int t = 0; t < n_tok; ++t) { a.R[t] = R + (size_t) t * HCn * N; a.R_out[t] = Ro + (size_t) t * HCn * N; a.out_prev[t] = op + (size_t) t * N; a.y[t] = y + (size_t) t * N; }
    a.post_prev = fdev((size_t) n_tok * HCn); a.comb_prev = fdev((size_t) n_tok * 16);
    a.fn = wat(0); a.scale = fdev(3); a.base = fdev(24); a.w_norm = fdev(N);
    float *po, *co; CK(cudaMalloc(&po, n_tok * HCn * 4)); CK(cudaMalloc(&co, n_tok * 16 * 4));
    a.post_out = po; a.comb_out = co;
    CK(cudaMalloc(&a.part, (size_t) n_tok * ns * 24 * 4)); CK(cudaMalloc(&a.ss_part, (size_t) n_tok * ns * 4));
    fused_mhc_read_multi(a, st); CK(cudaStreamSynchronize(st));
    auto h = get(y, (size_t) n_tok * N), r = get(Ro, apply ? (size_t) n_tok * HCn * N : 0), p = get(po, n_tok * HCn), c = get(co, n_tok * 16);
    h.insert(h.end(), r.begin(), r.end()); h.insert(h.end(), p.begin(), p.end()); h.insert(h.end(), c.begin(), c.end());
    char t[128]; std::snprintf(t, sizeof t, "mHC read n=%d apply=%d", n_tok, (int) apply);
    std::printf("  %-58s hash %016llx\n", t, (unsigned long long) fnv(h));
    report("", timed([&](int) { fused_mhc_read_multi(a, st); }, 1), (size_t) 24 * 16384 / 32 * 34);
}
static void gemv_case(const char* name, int n_in, int n_out, int max_n) {
    float* w = fdev((size_t) n_in * n_out);
    float* x = fdev((size_t) 4 * n_in);
    float* y; CK(cudaMalloc(&y, (size_t) 4 * n_out * 4));
    for (int n = 1; n <= max_n; ++n) {
        glm5_f32_gemv(w, x, y, n_in, n_out, n, 0.5f, st); CK(cudaStreamSynchronize(st));
        char t[128]; std::snprintf(t, sizeof t, "%s %d->%d n=%d", name, n_in, n_out, n);
        std::printf("  %-58s hash %016llx\n", t, (unsigned long long) fnv(get(y, (size_t) n * n_out)));
        report("", timed([&](int) { glm5_f32_gemv(w, x, y, n_in, n_out, n, 0.5f, st); }, 1), (size_t) n_in * n_out * 4);
    }
    cudaFree(w); cudaFree(x); cudaFree(y);
}

int main(int argc, char** argv) {
    if (argc > 1) reps = std::atoi(argv[1]);
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    pool_bytes = (size_t) 400 << 20;
    {   // valid Q8_0 blocks: fp16 scale in [2^-9, 2^-5), random int8
        std::vector<uint8_t> h(pool_bytes);
        for (size_t i = 0; i + 34 <= pool_bytes; i += 34) {
            const uint16_t d = (uint16_t) (0x1800 + nxt() % 0x1000);
            std::memcpy(&h[i], &d, 2);
            for (int k = 0; k < 32; k += 4) { const uint32_t r = nxt(); std::memcpy(&h[i + 2 + k], &r, 4); }
        }
        CK(cudaMalloc(&pool, pool_bytes));
        CK(cudaMemcpy(pool, h.data(), pool_bytes, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
    }
    const char* rows = std::getenv("STRATA_MMVQ_ROWS");
    const char* trows = std::getenv("STRATA_MMVQ_TABLE_ROWS");
    std::printf("glm5_batch_test: %d calls per timing, STRATA_MMVQ_ROWS=%s STRATA_MMVQ_TABLE_ROWS=%s\n", reps,
                rows ? rows : "(unset)", trows ? trows : "(unset)");
    table_case("KDA projections", 4096, {8192, 8192, 8192, 128, 128, 64}, 3);
    table_case("DSA projections", 4096, {1536, 512, 128, 128}, 3);
    absorb_case("MLA wk_b (q -> q_abs)", 256, 512, 64, 3);
    absorb_case("MLA wv_b (lat -> v)", 512, 256, 64, 3);
    shexp_case(4096, 2048, 7.0f, 3);
    std::printf("single matrices:\n");
    plain_case("KDA attn_output", 8192, 4096, 3);
    plain_case("DSA attn_output", 16384, 4096, 1);
    plain_case("DSA attn_q_b", 1536, 16384, 1);
    plain_case("indexer attn_q_b", 1536, 4096, 1);
    plain_case("KDA f_b / g_b", 128, 8192, 1);
    plain_case("shared gate / up", 4096, 2048, 1);
    plain_case("shared down", 2048, 4096, 1);
    plain_case("MTP eh_proj", 8192, 4096, 1);
    const char* mf = std::getenv("STRATA_MHC_FAST"); const char* gf = std::getenv("STRATA_F32_GEMV_FAST");
    std::printf("mHC read (STRATA_MHC_FAST=%s) and F32 GEMV (STRATA_F32_GEMV_FAST=%s):\n", mf ? mf : "(unset)", gf ? gf : "(unset)");
    for (int n = 1; n <= 3; ++n) { mhc_case(n, false); mhc_case(n, true); }
    gemv_case("router", 4096, 288, 3);
    gemv_case("indexer proj", 4096, 32, 3);
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
