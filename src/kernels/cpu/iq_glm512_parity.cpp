// src/kernels/cpu/iq_glm512_parity.cpp - the GLM AVX-512 rows (iq512_glm_gu / iq512_glm_rows, STRATA_GLM_IQ512) against
// ggml-cpu's vec_dot one token at a time, and against the AVX-2 rows (iq256_rows).  No GPU, no model: random blocks
// (every grid index, sign and scale pattern), random activations quantized to Q8_K by ggml.
//
//     iq_glm512_parity      IQ3_S rows of 4096 and IQ4_XS rows of 2048, 1..8 tokens: rel <= 1e-5 against ggml (the
//                           order of the float additions differs), each token's rows bit for bit the same alone and in
//                           a group; then the engine's dispatch (native_gu_rows with the clamp, native_down_rows) with
//                           STRATA_GLM_IQ512=1.  A CPU without AVX-512 F/BW/VL/VNNI/VBMI skips (exit 0).
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int kMaxT = 8;
constexpr int kRows = 63;   // odd: the down kernel's last row runs alone

uint64_t mix(uint64_t& s) {   // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// random bytes, then a sane fp16 scale at the front of every block (scale: its size)
void fill_blocks(uint8_t* p, size_t bytes, int type, uint64_t& seed, float scale = 1e-3f) {
    for (size_t i = 0; i < bytes; i += 8) {
        const uint64_t v = mix(seed);
        std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i));
    }
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const ggml_fp16_t h = ggml_fp32_to_fp16((0.5f + (float) (mix(seed) >> 40) * (1.f / 16777216.f)) * scale);
        std::memcpy(p + o, &h, 2);
    }
}

struct Acts {
    std::vector<std::vector<uint8_t>> q;
    const void* p[kMaxT];
    Acts(int n, unsigned seed) : q(kMaxT, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_K, n))) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) n);
        for (int t = 0; t < kMaxT; ++t) {
            for (auto& v : x) v = nd(rng);
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), q[(size_t) t].data(), n);
            p[t] = q[(size_t) t].data();
        }
    }
};

struct Err {
    double rel = 0;   // sum |a - b| / sum |b|
    double max = 0;   // max |a - b| / |b| over the values with |b| >= 1% of the mean |b|
};

Err compare(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += std::fabs((double) a[i] - b[i]); den += std::fabs((double) b[i]); }
    Err e;
    e.rel = num / (den + 1e-30);
    const double floor = 0.01 * den / (double) a.size();
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs((double) b[i]) >= floor) e.max = std::max(e.max, std::fabs((double) a[i] - b[i]) / std::fabs((double) b[i]));
    return e;
}

int failures = 0;
double worst_rel = 0, worst_max = 0;

// rows [1, 1 + kRows) of a matrix of type `type`, n wide: the kernel (gu: as gate and as up of a two-matrix blob)
// against ggml's dot per token, against itself token by token, and against the AVX-2 rows
void check_rows(int type, int n, uint64_t seed) {
    const size_t rb = ggml_row_size((ggml_type) type, n);
    const int r0 = 1, r1 = 1 + kRows;
    std::vector<uint8_t> w(rb * (size_t) (r1 + 1) * 2);   // gate rows, then up rows
    fill_blocks(w.data(), w.size(), type, seed);
    const size_t up_off = rb * (size_t) (r1 + 1);
    const Acts acts(n, (unsigned) seed);
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu((ggml_type) type)->vec_dot;
    // ggml per token: ref[t][r] (gate rows), refu[t][r] (up rows)
    std::vector<float> ref((size_t) kMaxT * kRows), refu((size_t) kMaxT * kRows);
    for (int t = 0; t < kMaxT; ++t)
        for (int r = r0; r < r1; ++r) {
            dot(n, &ref[(size_t) t * kRows + r - r0], 0, w.data() + (size_t) r * rb, 0, acts.p[t], 0, 1);
            dot(n, &refu[(size_t) t * kRows + r - r0], 0, w.data() + up_off + (size_t) r * rb, 0, acts.p[t], 0, 1);
        }
    std::vector<float> alone((size_t) kMaxT * kRows);   // each token's rows by itself (nt = 1)
    for (int t = 0; t < kMaxT; ++t) {
        std::vector<float> o1((size_t) r1);
        float* p1 = o1.data();
        cpu::iq512_glm_rows(type, w.data(), rb, n, acts.p + t, 1, &p1, r0, r1);
        std::copy(o1.begin() + r0, o1.end(), alone.begin() + (size_t) t * kRows);
    }
    for (int nt = 1; nt <= kMaxT; ++nt) {
        std::vector<std::vector<float>> o(nt, std::vector<float>((size_t) r1)), g(o), u(o), a2(o);
        float* op[kMaxT];
        float* gp[kMaxT];
        float* upp[kMaxT];
        float* ap[kMaxT];
        for (int t = 0; t < nt; ++t) { op[t] = o[t].data(); gp[t] = g[t].data(); upp[t] = u[t].data(); ap[t] = a2[t].data(); }
        cpu::iq512_glm_rows(type, w.data(), rb, n, acts.p, nt, op, r0, r1);
        cpu::iq512_glm_gu(type, w.data(), rb, up_off, n, acts.p, nt, gp, upp, r0, r1);
        std::vector<float> mine, mineg, mineu, want, wantu;
        size_t differ = 0, differ2 = 0, differ_gu = 0;
        for (int t = 0; t < nt; ++t)
            for (int r = r0; r < r1; ++r) {
                const size_t k = (size_t) t * kRows + r - r0;
                mine.push_back(o[t][r]);
                mineg.push_back(g[t][r]);
                mineu.push_back(u[t][r]);
                want.push_back(ref[k]);
                wantu.push_back(refu[k]);
                differ_gu += std::memcmp(&o[t][r], &g[t][r], 4) != 0;   // rows kernel = gate rows of the pair kernel
                differ += std::memcmp(&o[t][r], &alone[k], 4) != 0;
            }
        const Err e = compare(mine, want), eg = compare(mineg, want), eu = compare(mineu, wantu);
        // the AVX-2 rows (iq_avx2.cpp: gather + AVX-VNNI where the CPU has them)
        Err e2;
        if (cpu::cpu_avx2_ok()) {
            cpu::iq256_rows(type, w.data(), rb, n, acts.p, nt, ap, r0, r1);
            std::vector<float> avx2;
            for (int t = 0; t < nt; ++t)
                for (int r = r0; r < r1; ++r) { avx2.push_back(a2[t][r]); differ2 += std::memcmp(&a2[t][r], &o[t][r], 4) != 0; }
            e2 = compare(mine, avx2);
        }
        const double rel = std::max({e.rel, eg.rel, eu.rel}), mx = std::max({e.max, eg.max, eu.max});
        worst_rel = std::max(worst_rel, rel);
        worst_max = std::max(worst_max, mx);
        std::printf("  %-6s n %4d nt %d: vs ggml rel %.2e (max %.2e)  vs AVX-2 rel %.2e (%zu/%zu rows differ)  "
                    "group vs alone: %zu differ\n",
                    ggml_type_name((ggml_type) type), n, nt, rel, mx, e2.rel, differ2, mine.size(), differ + differ_gu);
        if (!(rel <= 1e-5) || differ || differ_gu) { std::printf("  MISMATCH\n"); ++failures; }
    }
}

// the engine's dispatch with STRATA_GLM_IQ512=1: native_gu_rows (gate, up, clamp 10, SiLU) and native_down_rows
void check_dispatch(uint64_t seed) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(21, 23, 4096, 2048, f, err, 10.f)) { std::printf("  native_fmt: %s\n", err.c_str()); ++failures; return; }
    std::vector<uint8_t> blob(f.bytes);
    fill_blocks(blob.data(), f.down_off, 21, seed, 3e-3f);   // large enough that some gates pass the clamp
    fill_blocks(blob.data() + f.down_off, f.bytes - f.down_off, 23, seed);
    const Acts a(4096, 7), h(2048, 8);
    const int R = 96;
    const ggml_vec_dot_t gdot = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S)->vec_dot;
    const ggml_vec_dot_t ddot = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_XS)->vec_dot;
    for (int nt = 1; nt <= kMaxT; ++nt) {
        std::vector<std::vector<float>> ff(nt, std::vector<float>(2048)), out(nt, std::vector<float>(4096));
        float* fp[kMaxT];
        float* op[kMaxT];
        for (int t = 0; t < nt; ++t) { fp[t] = ff[t].data(); op[t] = out[t].data(); }
        cpu::native_gu_rows(f, blob.data(), a.p, nt, fp, 0, R);
        cpu::native_down_rows(f, blob.data(), h.p, nt, op, 0, R);
        std::vector<float> mf, wf, md, wd;
        int clamped = 0;
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < R; ++r) {
                float g = 0, u = 0, d = 0;
                gdot(4096, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, a.p[t], 0, 1);
                gdot(4096, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, a.p[t], 0, 1);
                ddot(2048, &d, 0, blob.data() + f.down_off + (size_t) r * f.d_row, 0, h.p[t], 0, 1);
                clamped += g > 10.f || std::fabs(u) > 10.f;
                mf.push_back(ff[t][r]);
                wf.push_back(cpu::native_swiglu(g, u, 10.f));
                md.push_back(out[t][r]);
                wd.push_back(d);
            }
        const Err ef = compare(mf, wf), ed = compare(md, wd);
        worst_rel = std::max({worst_rel, ef.rel, ed.rel});
        worst_max = std::max({worst_max, ef.max, ed.max});
        std::printf("  dispatch nt %d: gate/up+clamp vs ggml rel %.2e (max %.2e, %d of %d clamped)  down rel %.2e (max %.2e)\n",
                    nt, ef.rel, ef.max, clamped, nt * R, ed.rel, ed.max);
        if (!(ef.rel <= 1e-5) || !(ed.rel <= 1e-5) || clamped == 0) { std::printf("  MISMATCH\n"); ++failures; }
    }
}

}  // namespace

int main() {
#if defined(_WIN32)
    _putenv_s("STRATA_GLM_IQ512", "1");
#else
    setenv("STRATA_GLM_IQ512", "1", 1);
#endif
    if (!cpu::cpu_avx512_hw_ok()) {
        std::printf("iq_glm512_parity: this CPU has no AVX-512 F/BW/VL/VNNI/VBMI - skipped\n");
        return 0;
    }
    ggml_cpu_init();
    std::printf("iq_glm512_parity:\n");
    check_rows(GGML_TYPE_IQ3_S, 4096, 11);
    check_rows(GGML_TYPE_IQ4_XS, 2048, 12);
    check_rows(GGML_TYPE_IQ4_XS, 4096, 13);
    check_dispatch(14);
    std::printf("worst: rel %.2e, max %.2e\n%s\n", worst_rel, worst_max, failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
