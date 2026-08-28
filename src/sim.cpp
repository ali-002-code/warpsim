#include "warpsim/warpsim.hpp"
#include "warpsim/cache.hpp"
#include <algorithm>
#include <set>
#include <memory>

namespace warpsim {

static int arith_latency(Op op, const GPUConfig& cfg) {
    switch (op) {
        case Op::IADD: return cfg.int_latency;
        case Op::NOP:  return 0;
        case Op::LOAD: case Op::STORE: return 0;  // handled via mem path
        default:       return cfg.fp_latency;
    }
}

long coalesce_transactions(long base, int stride, int warp_size, int line_bytes) {
    std::set<long> lines;
    for (int lane = 0; lane < warp_size; ++lane)
        lines.insert((base + (long)lane * stride) / line_bytes);
    return (long)lines.size();
}

// v2b: resolve one memory access to a latency via the L1.
// Simplification (documented): the access hits-or-misses as a whole, keyed on
// its base line. Per-line splitting of a coalesced access is deferred to v3,
// where transaction COUNT (bandwidth/MSHR) is what makes that distinction bite.
static int mem_latency(const Instruction& inst, const GPUConfig& cfg,
                       Cache* l1, RunResult& r) {
    long base_line_addr = inst.addr_base;  // byte addr of lane 0
    if (cfg.l1_enabled && l1) {
        bool hit = l1->access(base_line_addr);
        if (hit) { ++r.l1_hits;   return cfg.l1_hit_latency; }
        else     { ++r.l1_misses; return cfg.memory_latency; }
    }
    return cfg.memory_latency;
}

static long mem_txn(const Instruction& inst, const GPUConfig& cfg) {
    if (inst.op != Op::LOAD && inst.op != Op::STORE) return 0;
    return coalesce_transactions(inst.addr_base, inst.addr_stride,
                                 cfg.warp_size, cfg.line_bytes);
}

static bool is_mem(Op op) { return op == Op::LOAD || op == Op::STORE; }

RunResult run_single_warp(const std::vector<Instruction>& program,
                          const GPUConfig& config, int num_registers) {
    std::vector<long> ready((size_t)num_registers, 0);
    std::unique_ptr<Cache> l1 =
        config.l1_enabled
            ? std::make_unique<Cache>(config.l1_bytes, config.line_bytes, config.l1_assoc)
            : nullptr;

    long current_cycle = 0, max_completion = 0;
    size_t pc = 0;
    RunResult r;

    while (pc < program.size()) {
        const Instruction& inst = program[pc];
        bool eligible = true;
        for (int s : inst.src)
            if (s >= 0 && ready[s] > current_cycle) { eligible = false; break; }

        if (eligible) {
            int lat = is_mem(inst.op) ? mem_latency(inst, config, l1.get(), r)
                                      : arith_latency(inst.op, config);
            long completion = current_cycle + lat;
            if (inst.dst >= 0) ready[inst.dst] = completion;
            max_completion = std::max(max_completion, completion);
            r.memory_transactions += mem_txn(inst, config);
            ++pc;
        }
        ++current_cycle;
    }

    r.total_cycles = max_completion;
    r.instructions_issued = (long)program.size();
    return r;
}

RunResult run_sm(const std::vector<Instruction>& program,
                 const GPUConfig& config, int num_warps, int num_registers) {
    struct WarpState { size_t pc = 0; std::vector<long> ready; };
    std::vector<WarpState> warps((size_t)num_warps);
    for (auto& w : warps) w.ready.assign((size_t)num_registers, 0);

    std::unique_ptr<Cache> l1 =
        config.l1_enabled
            ? std::make_unique<Cache>(config.l1_bytes, config.line_bytes, config.l1_assoc)
            : nullptr;

    long current_cycle = 0, max_completion = 0, issued_total = 0;
    const long target = (long)program.size() * num_warps;
    size_t rr = 0;
    RunResult r;

    while (issued_total < target) {
        for (int slot = 0; slot < config.issue_width; ++slot) {
            bool issued_this_slot = false;
            for (size_t scan = 0; scan < warps.size(); ++scan) {
                size_t i = (rr + scan) % warps.size();
                WarpState& w = warps[i];
                if (w.pc >= program.size()) continue;

                const Instruction& inst = program[w.pc];
                bool eligible = true;
                for (int s : inst.src)
                    if (s >= 0 && w.ready[s] > current_cycle) { eligible = false; break; }
                if (!eligible) continue;

                int lat = is_mem(inst.op) ? mem_latency(inst, config, l1.get(), r)
                                          : arith_latency(inst.op, config);
                long completion = current_cycle + lat;
                if (inst.dst >= 0) w.ready[inst.dst] = completion;
                max_completion = std::max(max_completion, completion);
                r.memory_transactions += mem_txn(inst, config);
                ++w.pc; ++issued_total; rr = i + 1;
                issued_this_slot = true;
                break;
            }
            if (!issued_this_slot) break;
        }
        ++current_cycle;
    }

    r.total_cycles = max_completion;
    r.instructions_issued = issued_total;
    return r;
}

}  // namespace warpsim
