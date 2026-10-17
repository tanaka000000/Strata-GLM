// src/kernels/cpu/glm_expert_bench.cpp - GLM-5.3 expert rows on the CPU: ms per expert through the engine's dispatch
// (native_gu_rows with the clamped SwiGLU, native_down_rows), one thread and many, on the GLM geometry (gate/up
// IQ3_S 4096 -> 2048, down IQ4_XS 2048 -> 4096 by default).  No model: random blocks over --mb MB of expert blobs, far
// more than an L3 holds, so the weights stream from DRAM as in decode.  The numbers are timing only.
//
//     glm_expert_bench [--gu 21] [--down 23] [--embd 4096] [--ff 2048] [--nt 1,2,3] [--mb 512] [--reps 3]
//                      [--threads 1,8,20] [--cpu0 2] [--stride 2]
//     threads > 1: each thread its own experts (no sharing), pinned to logical CPUs cpu0, cpu0+stride, ...
//
//     glm_expert_bench --pool 20 [--per-layer 3] [--layers 42] [--nt 1] [--mb 2048] [--reps 20] [--tasks 0]
//     the engine's pool instead (2026-10-17): run_split_multi_native on `per-layer` experts a layer, `layers` layers a
//     window, `reps` windows; ms a window against the one-thread time over the pool's threads.  The STRATA_POOL_*
//     variables act as in the engine (STRATA_POOL_PROF=1 prints the phases).
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace strata::kernels::cpu;

namespace {

std::vector<int> ints(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

void pin(int cpu) {
#if defined(_WIN32)
    if (cpu >= 0) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);
#else
    (void) cpu;
#endif
}

struct Run {
    double gu_ms = 0, down_ms = 0;
};

// one thread: experts [e0, e1) of the blob array, nt tokens each, `reps` passes; ms per expert (gate/up, down)
Run run_experts(const NativeFmt& f, const std::vector<uint8_t>& blobs, int e0, int e1, int nt, int reps,
                const std::vector<std::vector<uint8_t>>& act, std::vector<std::vector<float>>& ff,
                const std::vector<std::vector<uint8_t>>& hq, std::vector<std::vector<float>>& out) {
    const void* a[8];
    float* fp[8];
    const void* h[8];
    float* op[8];
    for (int t = 0; t < nt; ++t) { a[t] = act[t].data(); fp[t] = ff[t].data(); h[t] = hq[t].data(); op[t] = out[t].data(); }
    Run r;
    using C = std::chrono::steady_clock;
    for (int rep = 0; rep < reps; ++rep) {
        double gu = 0, dn = 0;
        for (int e = e0; e < e1; ++e) {
            const uint8_t* b = blobs.data() + (size_t) e * f.bytes;
            const auto t0 = C::now();
            native_gu_rows(f, b, a, nt, fp, 0, (int) f.n_ff);
            const auto t1 = C::now();
            native_down_rows(f, b, h, nt, op, 0, (int) f.n_embd);
            const auto t2 = C::now();
            gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
            dn += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        if (rep > 0 || reps == 1) { r.gu_ms += gu; r.down_ms += dn; }   // the first pass warms the page tables
    }
    const int counted = reps > 1 ? reps - 1 : 1;
    r.gu_ms /= (double) counted * (e1 - e0);
    r.down_ms /= (double) counted * (e1 - e0);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    int gu = 21, dn = 23, embd = 4096, nff = 2048, mb = 512, reps = 3, cpu0 = 2, stride = 2;
    int pool_workers = 0, per_layer = 3, layers = 42, tasks = 0;
    std::vector<int> nts{1, 2, 3}, threads{1, 8, 20};
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--gu") gu = std::atoi(argv[i + 1]);
        else if (k == "--down") dn = std::atoi(argv[i + 1]);
        else if (k == "--embd") embd = std::atoi(argv[i + 1]);
        else if (k == "--ff") nff = std::atoi(argv[i + 1]);
        else if (k == "--nt") nts = ints(argv[i + 1]);
        else if (k == "--mb") mb = std::atoi(argv[i + 1]);
        else if (k == "--reps") reps = std::max(1, std::atoi(argv[i + 1]));
        else if (k == "--threads") threads = ints(argv[i + 1]);
        else if (k == "--cpu0") cpu0 = std::atoi(argv[i + 1]);
        else if (k == "--stride") stride = std::max(1, std::atoi(argv[i + 1]));
        else if (k == "--pool") pool_workers = std::atoi(argv[i + 1]);
        else if (k == "--per-layer") per_layer = std::max(1, std::atoi(argv[i + 1]));
        else if (k == "--layers") layers = std::max(1, std::atoi(argv[i + 1]));
        else if (k == "--tasks") tasks = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    NativeFmt f;
    std::string err;
    if (!native_fmt(gu, dn, embd, nff, f, err, 10.0f)) {   // GLM-5.3's clamp: swiglu_limit 10
        std::fprintf(stderr, "glm_expert_bench: %s\n", err.c_str());
        return 1;
    }
    const int n_exp = std::max(8, (int) (((int64_t) mb << 20) / (int64_t) f.bytes));
    std::printf("glm_expert_bench: gate/up type %d, down type %d, %d -> %d -> %d, %.2f MB an expert, %d experts (%.0f MB)\n",
                gu, dn, embd, nff, embd, f.bytes / 1048576.0, n_exp, n_exp * (double) f.bytes / 1048576.0);
    std::vector<uint8_t> blobs((size_t) n_exp * f.bytes);
    std::mt19937_64 rng(12345);
    for (size_t i = 0; i < blobs.size(); i += 8) { const uint64_t v = rng(); std::memcpy(blobs.data() + i, &v, std::min<size_t>(8, blobs.size() - i)); }
    // the fp16 scales of every block would be random bit patterns (NaN, huge): harmless for the timing, but keep them
    // finite by clearing the exponent's top bits of each 16-bit word that is a block's d (the first two bytes of a block)
    std::vector<float> x(embd), hv(nff);
    for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.f;
    for (auto& v : hv) v = (float) ((int) (rng() % 2001) - 1000) / 1000.f;
    const int maxt = *std::max_element(nts.begin(), nts.end());
    const int maxth = *std::max_element(threads.begin(), threads.end());
    std::vector<std::vector<std::vector<uint8_t>>> act(maxth), hq(maxth);
    std::vector<std::vector<std::vector<float>>> ff(maxth), out(maxth);
    for (int th = 0; th < maxth; ++th)
        for (int t = 0; t < maxt; ++t) {
            act[th].emplace_back(f.act_bytes); native_quant_act(f, x.data(), act[th].back().data());
            hq[th].emplace_back(f.h_bytes); native_quant_h(f, hv.data(), hq[th].back().data());
            ff[th].emplace_back(nff); out[th].emplace_back(embd);
        }
    if (pool_workers > 0) {
        const ThreadAffinity host = pin_current_thread(planned_host_core());
        ExpertPool pool(pool_workers, true, true, PoolAffinity::All, tasks);
        const int th = pool.workers() + (pool.host_works() ? 1 : 0);
        std::vector<float> outs((size_t) per_layer * maxt * embd);
        for (int nt : nts) {
            std::vector<ExpertJobMulti> jobs((size_t) per_layer);
            int next = 0;
            double ms = 0;
            for (int rep = 0; rep <= reps; ++rep) {   // rep 0 warms up
                if (rep == 1) { pool.prof_report(nullptr, 0); pool.ms_multi_q = 0; }
                const auto t0 = std::chrono::steady_clock::now();
                for (int l = 0; l < layers; ++l) {
                    for (int k = 0; k < per_layer; ++k) {
                        ExpertJobMulti& j = jobs[(size_t) k];
                        j.blob = blobs.data() + (size_t) (next++ % n_exp) * f.bytes;
                        j.nt = nt;
                        for (int t = 0; t < nt; ++t) {
                            j.nact[t] = act[0][t].data();
                            j.out[t] = outs.data() + ((size_t) k * maxt + t) * embd;
                        }
                    }
                    pool.run_split_multi_native(f, jobs.data(), per_layer);
                }
                if (rep > 0) ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
            ms /= reps;
            const Run one = run_experts(f, blobs, 0, std::min(n_exp, 8), nt, 3, act[0], ff[0], hq[0], out[0]);
            const double ideal = (double) layers * per_layer * (one.gu_ms + one.down_ms) / th;
            std::printf("pool %d threads  nt %d  %d experts x %d layers: %.2f ms a window (%.3f a layer); one thread's "
                        "time over %d threads %.2f ms (gate/up %.3f + down %.3f ms an expert)\n", th, nt, per_layer, layers,
                        ms, ms / layers, th, ideal, one.gu_ms, one.down_ms);
            pool.prof_report(stdout, reps);
            pool.ms_multi_q = 0;
        }
        restore_thread_affinity(host);
        return 0;
    }
    for (int nth : threads) {
        for (int nt : nts) {
            std::vector<Run> res(nth);
            std::vector<std::thread> ts;
            const int per = n_exp / nth;
            std::atomic<int> ready{0};
            const auto t0 = std::chrono::steady_clock::now();
            for (int th = 0; th < nth; ++th)
                ts.emplace_back([&, th] {
                    pin(cpu0 + th * stride);
                    ready.fetch_add(1);
                    while (ready.load() < nth) std::this_thread::yield();
                    res[th] = run_experts(f, blobs, th * per, (th + 1) * per, nt, reps, act[th], ff[th], hq[th], out[th]);
                });
            for (auto& t : ts) t.join();
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            double gu_ms = 0, dn_ms = 0;
            for (const Run& r : res) { gu_ms += r.gu_ms; dn_ms += r.down_ms; }
            gu_ms /= nth; dn_ms /= nth;
            const double gu_mb = 2.0 * f.up_off / 1048576.0, dn_mb = (f.bytes - f.down_off) / 1048576.0;
            std::printf("threads %2d  nt %d : gate/up %.3f ms (%.1f GB/s a thread)  down %.3f ms (%.1f GB/s a thread)  "
                        "all threads %.1f GB/s  (wall %.1f s)\n",
                        nth, nt, gu_ms, gu_mb / 1024.0 / (gu_ms / 1000.0), dn_ms, dn_mb / 1024.0 / (dn_ms / 1000.0),
                        nth * (gu_mb + dn_mb) / 1024.0 / ((gu_ms + dn_ms) / 1000.0), wall);
        }
    }
    return 0;
}
