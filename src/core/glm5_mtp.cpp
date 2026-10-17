// src/core/glm5_mtp.cpp - see include/strata/core/glm5_mtp.hpp.  The DSA block is verify_glm5.cpp's, on blk.45 and
// the drafter's own latent cache and indexer pool.
#include "strata/core/glm5_mtp.hpp"

#include "strata/core/layout.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm5_misc.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace strata::core {

struct Glm5Mtp::Impl {
    const WeightTable* wt = nullptr;
    const NativeHead* head = nullptr;
    int64_t N = 0, NH = 0, QLR = 0, KVR = 0, QK = 0, VD = 0, IH = 0, ID = 0, NFF = 0, n_vocab = 0;
    int64_t max_cells = 0, max_blocks = 0, cap = 0;
    int MT = 0;
    float rms_eps = 1e-5f, ln_eps = 1e-6f, clamp_sh = 0.f;
    strata::kernels::QsaShapes s;
    cudaStream_t cs = nullptr;
    // weights
    const WeightRef *eh = nullptr, *enorm = nullptr, *hnorm = nullptr, *shn = nullptr, *an = nullptr, *fn = nullptr;
    const WeightRef *wqa = nullptr, *wqan = nullptr, *wqb = nullptr, *wkb = nullptr, *wvb = nullptr, *wkv = nullptr,
                    *wkvn = nullptr, *wo = nullptr, *wik = nullptr, *wiqb = nullptr, *wikn = nullptr, *wikb = nullptr,
                    *wip = nullptr, *wape = nullptr, *wig = nullptr, *wsg = nullptr, *wsu = nullptr, *wsd = nullptr;
    // state
    uint16_t* latent = nullptr;   // [max_cells][KVR] fp16
    float *tail = nullptr, *tail_snap = nullptr;
    uint16_t* pooled = nullptr;   // BF16 (2026-10-09), as the verifier's
    int32_t* page_table = nullptr;
    // scratch [MT][...]
    int32_t *tok = nullptr, *steps = nullptr, *sel = nullptr, *out_tok = nullptr;
    float *emb = nullptr, *eN = nullptr, *hN = nullptr, *cat = nullptr, *u = nullptr, *x = nullptr, *ao = nullptr;
    float *qr = nullptr, *qb = nullptr, *qT = nullptr, *qaT = nullptr, *qa = nullptr, *kv = nullptr, *rk = nullptr,
          *rg = nullptr, *iq = nullptr, *iw = nullptr, *scores = nullptr, *lat = nullptr, *latT = nullptr, *vbT = nullptr,
          *vb = nullptr, *mla_scratch = nullptr, *sg = nullptr, *su = nullptr, *sgs = nullptr, *sh = nullptr,
          *hp = nullptr, *logits = nullptr;
    uint8_t* q8 = nullptr;
    std::vector<int32_t> h_steps;
    // phase B: blk.45's routed experts in VRAM, [288][blob] native blobs (gate | up | down), and the router
    uint8_t* experts = nullptr;
    strata::kernels::NativeExpertLayout L;
    int64_t NE = 0, K = 8;
    float wscale = 1.f, clamp_exp = 0.f;
    const WeightRef *wr = nullptr, *wb = nullptr;
    float *rlog = nullptr, *rw = nullptr, *parts = nullptr, *moe = nullptr;
    int32_t* rids = nullptr;
    int32_t* plan = nullptr;   // device: start[cap+1] | dst[cap] | tok[cap] | count
    unsigned long long* gptr = nullptr;
    uint8_t* xq = nullptr;
    void* gscratch = nullptr;
    std::vector<int32_t> h_ids, h_plan;
    std::vector<unsigned long long> h_ptr;
    std::vector<void*> allocs;
    template <typename T> T* dev(size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(T) + 256) != cudaSuccess) { cudaGetLastError(); allocs.push_back(nullptr); return nullptr; }
        cudaMemset(p, 0, n * sizeof(T) + 256);
        allocs.push_back(p);
        return (T*) p;
    }
};

Glm5Mtp::~Glm5Mtp() {
    if (!m_) return;
    if (m_->cs) { cudaStreamSynchronize(m_->cs); cudaStreamDestroy(m_->cs); }
    for (void* p : m_->allocs) cudaFree(p);
    delete m_;
}

bool Glm5Mtp::init(const WeightTable& wt, const ModelGeometry& g, const NativeHead* head, int64_t max_cells, int max_t,
                   const std::string& native_shard1, const std::string& pack_dir, std::string& err) {
    using namespace strata::kernels;
    m_ = new Impl();
    Impl& m = *m_;
    m.wt = &wt;
    m.head = head;
    m.N = g.n_embd; m.NH = g.n_head; m.QLR = g.q_lora_rank; m.KVR = g.kv_lora_rank; m.QK = g.mla_key_dim; m.VD = g.mla_value_dim;
    m.IH = g.idx_q_heads; m.ID = g.idx_key_dim; m.NFF = g.n_ff;
    m.rms_eps = g.rms_eps; m.ln_eps = g.norm_eps;
    m.clamp_sh = g.swiglu_clamp_shared.empty() ? 0.f : g.swiglu_clamp_shared.back();
    m.max_cells = max_cells;
    m.max_blocks = max_cells / 4 + 2;
    m.MT = max_t;
    m.s = qsa_real_shapes();
    m.s.n_head = g.n_head; m.s.n_head_kv = g.n_head_kv; m.s.head_dim = g.head_dim; m.s.idx_n_head = g.idx_q_heads;
    m.s.idx_dim = g.idx_key_dim;
    m.cap = qsa_selection_width(kTopkMaxCells, m.s);
    if (head == nullptr || !head->loaded()) { err = "glm5 mtp: needs the native head"; return false; }
    if (native_embed() == nullptr) { err = "glm5 mtp: needs the native token embedding"; return false; }
    const int64_t L = g.n_layers;   // the NextN layer's index (blk.45)
    auto W = [&](const char* suffix) -> const WeightRef* {
        const std::string name = "blk." + std::to_string(L) + "." + suffix;
        const WeightRef* r = wt.find(name);
        if (r == nullptr && err.empty()) err = "glm5 mtp: " + name + " is missing";
        return r;
    };
    m.eh = W("nextn.eh_proj.weight"); m.enorm = W("nextn.enorm.weight"); m.hnorm = W("nextn.hnorm.weight");
    m.shn = W("nextn.shared_head_norm.weight"); m.an = W("attn_norm.weight"); m.fn = W("ffn_norm.weight");
    m.wqa = W("attn_q_a.weight"); m.wqan = W("attn_q_a_norm.weight"); m.wqb = W("attn_q_b.weight");
    m.wkb = W("attn_k_b.weight"); m.wvb = W("attn_v_b.weight"); m.wkv = W("attn_kv_a_mqa.weight");
    m.wkvn = W("attn_kv_a_norm.weight"); m.wo = W("attn_output.weight"); m.wik = W("indexer.attn_k.weight");
    m.wiqb = W("indexer.attn_q_b.weight"); m.wikn = W("indexer.k_norm.weight"); m.wikb = W("indexer.k_norm.bias");
    m.wip = W("indexer.proj.weight"); m.wape = W("indexer_compressor_ape.weight"); m.wig = W("indexer_compressor_gate.weight");
    m.wsg = W("ffn_gate_shexp.weight"); m.wsu = W("ffn_up_shexp.weight"); m.wsd = W("ffn_down_shexp.weight");
    if (!err.empty()) return false;
    for (const WeightRef* q : {m.eh, m.wqa, m.wqb, m.wkb, m.wvb, m.wkv, m.wo, m.wik, m.wiqb, m.wig, m.wsg, m.wsu, m.wsd})
        if (q->native_data == nullptr) { err = "glm5 mtp: a blk.45 projection is not served natively (--native)"; return false; }
    for (const WeightRef* f : {m.enorm, m.hnorm, m.shn, m.an, m.fn, m.wqan, m.wkvn, m.wikn, m.wikb, m.wip, m.wape})
        if (f->data == nullptr || f->bytes < (uint64_t) f->elements * 4) { err = "glm5 mtp: a blk.45 norm is not F32 in the arena"; return false; }
    const WeightRef* wout = wt.find("output.weight");
    m.n_vocab = wout ? wout->ne1 : 0;
    if (m.n_vocab <= 0) { err = "glm5 mtp: output.weight is missing"; return false; }
    if (cudaStreamCreateWithFlags(&m.cs, cudaStreamNonBlocking) != cudaSuccess) { err = "glm5 mtp: stream"; return false; }
    const size_t T = (size_t) m.MT;
    m.latent = m.dev<uint16_t>((size_t) max_cells * m.KVR);
    m.tail = m.dev<float>(3 * 2 * m.ID); m.tail_snap = m.dev<float>(3 * 2 * m.ID);
    m.pooled = m.dev<uint16_t>((size_t) (max_cells / 4 + 1) * m.ID);
    m.page_table = m.dev<int32_t>((size_t) (max_cells / 4 + 1));
    m.tok = m.dev<int32_t>(T); m.steps = m.dev<int32_t>(T * kStepCount); m.sel = m.dev<int32_t>(T * m.cap);
    m.out_tok = m.dev<int32_t>(T);
    m.emb = m.dev<float>(T * m.N); m.eN = m.dev<float>(T * m.N); m.hN = m.dev<float>(T * m.N); m.cat = m.dev<float>(T * 2 * m.N);
    m.u = m.dev<float>(T * m.N); m.x = m.dev<float>(T * m.N); m.ao = m.dev<float>(T * m.N);
    m.qr = m.dev<float>(T * m.QLR); m.qb = m.dev<float>(T * m.NH * m.QK); m.qT = m.dev<float>(T * m.NH * m.QK);
    m.qaT = m.dev<float>(T * m.NH * m.KVR); m.qa = m.dev<float>(T * m.NH * m.KVR); m.kv = m.dev<float>(T * m.KVR);
    m.rk = m.dev<float>(T * m.ID); m.rg = m.dev<float>(T * m.ID); m.iq = m.dev<float>(T * m.IH * m.ID); m.iw = m.dev<float>(T * m.IH);
    m.scores = m.dev<float>(T * m.max_blocks);
    m.lat = m.dev<float>(T * m.NH * m.KVR); m.latT = m.dev<float>(T * m.NH * m.KVR);
    m.vbT = m.dev<float>(T * m.NH * m.VD); m.vb = m.dev<float>(T * m.NH * m.VD);
    m.mla_scratch = m.dev<float>(T * mla_decode_attn_scratch_floats(m.cap, m.NH));
    m.sg = m.dev<float>(T * m.NFF); m.su = m.dev<float>(T * m.NFF); m.sgs = m.dev<float>(T + 4); m.sh = m.dev<float>(T * m.N);
    m.hp = m.dev<float>(T * m.N); m.logits = m.dev<float>(T * m.n_vocab);
    m.q8 = m.dev<uint8_t>((size_t) (m.NH * T) * (m.KVR / 32) * 36 + T * (2 * m.N / 32) * 36);
    for (void* p : m.allocs) if (p == nullptr) { err = "glm5 mtp: a device buffer does not fit"; return false; }
    {
        std::vector<int32_t> pt((size_t) (max_cells / 4 + 1));
        for (size_t i = 0; i < pt.size(); ++i) pt[i] = (int32_t) i;
        cudaMemcpy(m.page_table, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice);
    }
    m.h_steps.assign(T * kStepCount, 0);
    // ---- phase B: blk.45's routed experts into VRAM (STRATA_GLM5_MTP_EXPERTS=0: the shared expert alone, phase A)
    const char* ev = std::getenv("STRATA_GLM5_MTP_EXPERTS");
    if (!(ev != nullptr && std::atoi(ev) == 0) && !pack_dir.empty() && !native_shard1.empty()) {
        m.NE = g.n_expert; m.K = g.n_expert_used; m.wscale = g.expert_weights_scale;
        m.clamp_exp = g.swiglu_clamp_exp.empty() ? 0.f : g.swiglu_clamp_exp.back();
        m.wr = W("ffn_gate_inp.weight"); m.wb = W("exp_probs_b.bias");
        if (!err.empty()) return false;
        std::ifstream lay(pack_dir + "/native_experts.txt");
        std::string line, want = std::to_string(L) + " ";
        long long gu = -1, dt = -1, off = 0, blob = 0, go = 0, uo = 0, dofs = 0;
        std::string shard;
        while (std::getline(lay, line))
            if (line.rfind(want, 0) == 0) {
                std::istringstream is(line);
                long long l2;
                is >> l2 >> gu >> dt >> off >> blob >> go >> uo >> dofs >> shard;
                break;
            }
        if (gu < 0 || shard.empty() || shard.find(',') != std::string::npos) {
            err = "glm5 mtp: native_experts.txt has no single-shard line for blk." + std::to_string(L);
            return false;
        }
        if (!native_expert_supported((int) gu, (int) dt, m.N, m.NFF)) {
            err = "glm5 mtp: blk.45's expert types " + std::to_string(gu) + "/" + std::to_string(dt) + " have no GPU kernel";
            return false;
        }
        m.L = native_expert_layout((int) gu, (int) dt, m.N, m.NFF);
        m.L.swiglu_limit = m.clamp_exp;
        if ((long long) m.L.bytes != blob) { err = "glm5 mtp: blk.45's blob size disagrees with the layout"; return false; }
        const size_t slash = native_shard1.find_last_of("/\\");
        const std::string path = (slash == std::string::npos ? std::string() : native_shard1.substr(0, slash + 1)) + shard;
        m.experts = m.dev<uint8_t>((size_t) m.NE * m.L.bytes);
        if (m.experts == nullptr) {
            err = "glm5 mtp: blk.45's experts (" + std::to_string((m.NE * (long long) m.L.bytes) >> 20) +
                  " MiB) do not fit in VRAM - raise --vram-reserve-mib";
            return false;
        }
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) { err = "glm5 mtp: cannot open " + path; return false; }
        const long long role_off[3] = {go, uo, dofs};
        const size_t role_bytes[3] = {m.L.up_off, m.L.up_off, m.L.bytes - m.L.down_off};
        const size_t dst_off[3] = {0, m.L.up_off, m.L.down_off};
        std::vector<uint8_t> buf;
        for (int r = 0; r < 3; ++r) {
            buf.resize((size_t) m.NE * role_bytes[r]);
            if (_fseeki64(f, role_off[r], SEEK_SET) != 0 || std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
                std::fclose(f);
                err = "glm5 mtp: reading blk.45's experts from " + path + " failed";
                return false;
            }
            cudaMemcpy2D(m.experts + dst_off[r], m.L.bytes, buf.data(), role_bytes[r], role_bytes[r], (size_t) m.NE,
                         cudaMemcpyHostToDevice);
        }
        std::fclose(f);
        const int64_t cap = (int64_t) T * m.K;
        m.rlog = m.dev<float>(T * m.NE); m.rw = m.dev<float>((size_t) cap); m.rids = m.dev<int32_t>((size_t) cap);
        m.parts = m.dev<float>((size_t) cap * m.N); m.moe = m.dev<float>(T * m.N);
        m.plan = m.dev<int32_t>((size_t) (3 * cap + 2)); m.gptr = m.dev<unsigned long long>((size_t) cap);
        m.xq = m.dev<uint8_t>(T * (m.N / 32) * 36);
        m.gscratch = m.dev<uint8_t>(native_expert_scratch_bytes(cap, m.NFF));
        for (void* p : m.allocs) if (p == nullptr) { err = "glm5 mtp: a device buffer does not fit"; return false; }
        m.h_ids.resize((size_t) cap); m.h_plan.resize((size_t) (3 * cap + 2)); m.h_ptr.resize((size_t) cap);
        std::fprintf(stderr, "strata glm5 mtp: blk.%lld's %lld routed experts in VRAM (%.2f GiB, types %lld/%lld) from %s\n",
                     (long long) L, (long long) m.NE, (double) m.NE * m.L.bytes / 1073741824.0, gu, dt, shard.c_str());
    }
    std::fprintf(stderr, "strata glm5 mtp: NextN layer blk.%lld bound (%s), %lld-cell cache\n", (long long) L,
                 m.experts ? "routed + shared experts" : "shared expert only, phase A", (long long) max_cells);
    return true;
}

// STRATA_GLM5_MTP_BATCH=1 (2026-10-17, opt-in): the drafter's DSA block as the verify window's batched one - the four
// projections of x in one launch, the 64 per-head wk_b / wv_b products in one launch each reading and writing the
// [row][head] layouts (no transposes), the shared expert's gate and up in one launch.  Every value bitwise the
// default's; the drafter runs outside a graph, so this removes ~130 launches per layer() call.
static const bool mtp_batch = [] { const char* v = std::getenv("STRATA_GLM5_MTP_BATCH"); return v && v[0] == '1'; }();

// The NextN layer over n rows: tokens tok_dev (device) at cells c0+1.., h rows h_in (device) -> m.hp (h') and
// m.out_tok (argmax), the drafter's cache written at cells c0 ..
bool Glm5Mtp::layer(const int32_t* tok_dev, const float* h_in, int n, int64_t c0, std::string& err) {
    using namespace strata::kernels;
    Impl& m = *m_;
    cudaStream_t cs = m.cs;
    const int64_t N = m.N, QLR = m.QLR, NH = m.NH, QK = m.QK, KVR = m.KVR, VD = m.VD, ID = m.ID, IH = m.IH;
    if (n < 1 || n > m.MT || c0 + n > m.max_cells) { err = "glm5 mtp: rows out of range"; return false; }
    for (int t = 0; t < n; ++t) qsa_step_fill(m.h_steps.data() + (size_t) t * kStepCount, c0 + t, m.s);
    cudaMemcpyAsync(m.steps, m.h_steps.data(), (size_t) n * kStepCount * 4, cudaMemcpyHostToDevice, cs);
    try {
        auto mmvq = [&](const WeightRef* w, const float* in, int n_in, float* y, int n_out) {
            native_quantize_q8_1(in, m.q8, n_in, n, cs);
            native_mmvq(w->native_type, w->native_data, m.q8, y, n_in, n_out, n, cs);
        };
        // u = eh_proj([enorm(e); hnorm(h)])
        native_embed()->gather_dev(tok_dev, n, m.emb, cs);
        native_qsa_rms_norm_weighted(m.emb, (const float*) m.enorm->data, m.eN, (int) N, n, m.rms_eps, cs);
        native_qsa_rms_norm_weighted(h_in, (const float*) m.hnorm->data, m.hN, (int) N, n, m.rms_eps, cs);
        cudaMemcpy2DAsync(m.cat, 2 * N * 4, m.eN, N * 4, N * 4, n, cudaMemcpyDeviceToDevice, cs);
        cudaMemcpy2DAsync(m.cat + N, 2 * N * 4, m.hN, N * 4, N * 4, n, cudaMemcpyDeviceToDevice, cs);
        mmvq(m.eh, m.cat, (int) (2 * N), m.u, (int) N);
        // ---- DSA (verify_glm5.cpp's block)
        native_qsa_rms_norm_weighted(m.u, (const float*) m.an->data, m.x, (int) N, n, m.rms_eps, cs);
        native_quantize_q8_1(m.x, m.q8, (int) N, n, cs);
        {
            const WeightRef* ws[4] = {m.wqa, m.wkv, m.wik, m.wig};
            const void* wp[4];
            float* yp[4] = {m.qr, m.kv, m.rk, m.rg};
            const int no[4] = {(int) QLR, (int) KVR, (int) ID, (int) ID};
            bool q8 = mtp_batch;
            for (int i = 0; i < 4; ++i) { q8 = q8 && ws[i]->native_type == 8; wp[i] = ws[i]->native_data; }
            if (!(q8 && native_mmvq_table_q8_0(wp, no, 4, m.q8, yp, (int) N, cs, n)))
                for (int i = 0; i < 4; ++i)
                    native_mmvq(ws[i]->native_type, ws[i]->native_data, m.q8, yp[i], (int) N, no[i], n, cs);
        }
        native_qsa_rms_norm_weighted(m.qr, (const float*) m.wqan->data, m.qr, (int) QLR, n, m.rms_eps, cs);
        native_qsa_rms_norm_weighted(m.kv, (const float*) m.wkvn->data, m.kv, (int) KVR, n, m.rms_eps, cs);
        glm5_latent_store(m.kv, m.latent, m.page_table, m.steps, kStepCount, 4, (int) KVR, n, cs);
        // direct: the per-head products read / write the [row][head] layouts with strides (row t, head h = t * NH + h)
        const bool direct = mtp_batch && n <= 8 && m.wkb->native_type == 8 && m.wvb->native_type == 8;
        mmvq(m.wqb, m.qr, (int) QLR, m.qb, (int) (NH * QK));
        if (!direct) glm5_transpose_abd(m.qb, m.qT, n, (int) NH, (int) QK, cs);
        native_quantize_q8_1(direct ? m.qb : m.qT, m.q8, (int) (QK * NH * n), 1, cs);
        {
            const size_t row = (size_t) (QK / 32) * 34, qcol = (size_t) (QK / 32) * 36;
            if (direct) {
                if (!native_mmvq_batched_q8_0(m.wkb->native_data, (size_t) KVR * row, m.q8, qcol, m.qa, (size_t) KVR,
                                              (int) QK, (int) KVR, (int) NH, cs, n, NH * qcol, (size_t) (NH * KVR))) {
                    err = "glm5 mtp: the batched wk_b launch was refused";
                    return false;
                }
            } else {
                for (int64_t h = 0; h < NH; ++h)
                    native_mmvq(m.wkb->native_type, (const uint8_t*) m.wkb->native_data + (size_t) h * KVR * row,
                                m.q8 + (size_t) h * n * qcol, m.qaT + (size_t) h * n * KVR, (int) QK, (int) KVR, n, cs);
            }
        }
        if (!direct) glm5_transpose_abd(m.qaT, m.qa, (int) NH, n, (int) KVR, cs);
        const Glm5KpoolBuffers kb{m.tail, m.pooled};
        for (int t = 0; t < n; ++t)
            glm5_kpool_append(m.rk + (size_t) t * ID, m.rg + (size_t) t * ID, m.steps + (size_t) t * kStepCount + kStepPos,
                              (const float*) m.wikn->data, (const float*) m.wikb->data, (const float*) m.wape->data, m.ln_eps,
                              kb, m.max_cells, cs);
        mmvq(m.wiqb, m.qr, (int) QLR, m.iq, (int) (IH * ID));
        glm5_f32_gemv((const float*) m.wip->data, m.x, m.iw, (int) N, (int) IH, n, 1.0f / std::sqrt((float) (ID * IH)), cs);
        glm5_kpool_scores(m.iq, m.iw, m.pooled, m.steps, n, m.max_blocks, m.scores, cs);
        qsa_block_topk(m.scores, m.steps, n, m.max_blocks, m.cap, m.s, m.sel, cs);
        mla_decode_attn_batch(m.qa, m.latent, m.page_table, 4, m.sel, m.steps, m.cap, (int) NH, 1.0f / std::sqrt((float) QK),
                              m.mla_scratch, m.lat, n, cs);
        if (!direct) glm5_transpose_abd(m.lat, m.latT, n, (int) NH, (int) KVR, cs);
        native_quantize_q8_1(direct ? m.lat : m.latT, m.q8, (int) (KVR * NH * n), 1, cs);
        {
            const size_t row = (size_t) (KVR / 32) * 34, qcol = (size_t) (KVR / 32) * 36;
            if (direct) {
                if (!native_mmvq_batched_q8_0(m.wvb->native_data, (size_t) VD * row, m.q8, qcol, m.vb, (size_t) VD,
                                              (int) KVR, (int) VD, (int) NH, cs, n, NH * qcol, (size_t) (NH * VD))) {
                    err = "glm5 mtp: the batched wv_b launch was refused";
                    return false;
                }
            } else {
                for (int64_t h = 0; h < NH; ++h)
                    native_mmvq(m.wvb->native_type, (const uint8_t*) m.wvb->native_data + (size_t) h * VD * row,
                                m.q8 + (size_t) h * n * qcol, m.vbT + (size_t) h * n * VD, (int) KVR, (int) VD, n, cs);
            }
        }
        if (!direct) glm5_transpose_abd(m.vbT, m.vb, (int) NH, n, (int) VD, cs);
        mmvq(m.wo, m.vb, (int) (NH * VD), m.ao, (int) N);
        add_inplace(m.u, m.ao, (int64_t) n * N, cs);
        // ---- FFN: the shared expert (phase A: no routed experts)
        native_qsa_rms_norm_weighted(m.u, (const float*) m.fn->data, m.x, (int) N, n, m.rms_eps, cs);
        NativeSharedWeights nsw;
        nsw.gate_type = m.wsg->native_type; nsw.gate_data = m.wsg->native_data; nsw.up_type = m.wsu->native_type;
        nsw.up_data = m.wsu->native_data; nsw.down_type = m.wsd->native_type; nsw.down_data = m.wsd->native_data; nsw.q8_1 = m.q8;
        shared_expert_multi(n, m.x, nullptr, nsw, nullptr, m.sg, m.su, m.sgs, m.sh, N, m.NFF, cs, nullptr, mtp_batch ? 6 : 0, m.clamp_sh);
        if (m.experts != nullptr) {   // phase B: the routed experts, all on the GPU
            const int64_t K = m.K, cap = (int64_t) n * K;
            glm5_f32_gemv((const float*) m.wr->data, m.x, m.rlog, (int) N, (int) m.NE, n, 1.0f, cs);
            router_sigmoid_topk(m.rlog, (const float*) m.wb->data, n, (int) m.NE, (int) K, m.wscale, m.rids, m.rw, cs);
            cudaMemcpyAsync(m.h_ids.data(), m.rids, (size_t) cap * 4, cudaMemcpyDeviceToHost, cs);
            if (cudaStreamSynchronize(cs) != cudaSuccess) { err = "glm5 mtp: the router failed on the GPU"; return false; }
            // groups: one per distinct expert, its entries (row t*K + j of parts, token t)
            int32_t* start = m.h_plan.data();
            int32_t* dst = start + cap + 1;
            int32_t* tok = dst + cap;
            int groups = 0, entries = 0;
            std::vector<char> done((size_t) cap, 0);
            for (int64_t i = 0; i < cap; ++i) {
                if (done[(size_t) i]) continue;
                const int32_t e = m.h_ids[(size_t) i];
                m.h_ptr[(size_t) groups] = (unsigned long long) (m.experts + (size_t) e * m.L.bytes);
                start[groups] = entries;
                for (int64_t j = i; j < cap; ++j)
                    if (!done[(size_t) j] && m.h_ids[(size_t) j] == e) {
                        done[(size_t) j] = 1;
                        dst[entries] = (int32_t) j;
                        tok[entries] = (int32_t) (j / K);
                        ++entries;
                    }
                ++groups;
            }
            start[groups] = entries;
            int32_t* count = tok + cap;
            count[0] = groups;
            cudaMemcpyAsync(m.plan, m.h_plan.data(), m.h_plan.size() * 4, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync(m.gptr, m.h_ptr.data(), (size_t) groups * 8, cudaMemcpyHostToDevice, cs);
            quantize_q8_1_rows(m.x, n, N, m.xq, cs);
            native_expert_grouped(m.L, m.gptr, m.plan, m.plan + 3 * cap + 1, m.plan + cap + 1, m.plan + 2 * cap + 1, cap, cap,
                                  m.xq, m.gscratch, m.parts, cs);
            native_moe_combine_multi(m.parts, m.rw, m.sh, m.moe, N, K, n, cs);
            add_inplace(m.u, m.moe, (int64_t) n * N, cs);
        } else {
            add_inplace(m.u, m.sh, (int64_t) n * N, cs);
        }
        // ---- h' and the draft
        native_qsa_rms_norm_weighted(m.u, (const float*) m.shn->data, m.hp, (int) N, n, m.rms_eps, cs);
        native_quantize_q8_1(m.hp, m.q8, (int) N, n, cs);
        native_mmvq(m.head->type(), m.head->weights(), m.q8, m.logits, (int) N, (int) m.n_vocab, n, cs);
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(m.logits, n, (int) m.n_vocab, nullptr, 0, sp, m.out_tok, cs);
    } catch (const std::exception& e) {
        err = std::string("glm5 mtp: ") + e.what();
        return false;
    }
    return true;
}

bool Glm5Mtp::catchup(const float* h_rows, const int32_t* tokens, int n, int64_t c0, int draft_from, int32_t* first,
                      std::string& err) {
    Impl& m = *m_;
    const auto t0 = std::chrono::steady_clock::now();
    cudaMemcpyAsync(m.tok, tokens, (size_t) n * 4, cudaMemcpyHostToDevice, m.cs);
    if (!layer(m.tok, h_rows, n, c0, err)) return false;
    if (draft_from >= 0) {
        int32_t d = 0;
        cudaMemcpyAsync(&d, m.out_tok + draft_from, 4, cudaMemcpyDeviceToHost, m.cs);
        // the chain's h: row draft_from's h', moved to row 0 of the h buffer the chain reads
        cudaMemcpyAsync(m.hN, m.hp + (size_t) draft_from * m.N, (size_t) m.N * 4, cudaMemcpyDeviceToDevice, m.cs);
        if (cudaStreamSynchronize(m.cs) != cudaSuccess) { err = "glm5 mtp: catch-up failed on the GPU"; return false; }
        *first = d;
    } else if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = "glm5 mtp: catch-up failed on the GPU";
        return false;
    }
    ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool Glm5Mtp::chain(int32_t first, int64_t c_last, int n_more, int32_t* out, std::string& err) {
    using namespace strata::kernels;
    Impl& m = *m_;
    const auto t0 = std::chrono::steady_clock::now();
    if (n_more <= 0) return true;
    copy_from_mapped(m.tail_snap, m.tail, 3 * 2 * m.ID, m.cs);   // the cells written below are rewritten later
    int32_t tok = first;
    for (int j = 0; j < n_more; ++j) {
        cudaMemcpyAsync(m.tok, &tok, 4, cudaMemcpyHostToDevice, m.cs);
        // h: the previous step's h' (kept in m.hN row 0; layer() overwrites hN with hnorm, so pass a copy)
        cudaMemcpyAsync(m.x, m.hN, (size_t) m.N * 4, cudaMemcpyDeviceToDevice, m.cs);
        if (!layer(m.tok, m.x, 1, c_last + 1 + j, err)) return false;
        int32_t d = 0;
        cudaMemcpyAsync(&d, m.out_tok, 4, cudaMemcpyDeviceToHost, m.cs);
        cudaMemcpyAsync(m.hN, m.hp, (size_t) m.N * 4, cudaMemcpyDeviceToDevice, m.cs);
        if (cudaStreamSynchronize(m.cs) != cudaSuccess) { err = "glm5 mtp: draft failed on the GPU"; return false; }
        out[j] = d;
        tok = d;
    }
    copy_from_mapped(m.tail, m.tail_snap, 3 * 2 * m.ID, m.cs);
    cudaStreamSynchronize(m.cs);
    ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++rounds;
    return true;
}

}  // namespace strata::core
