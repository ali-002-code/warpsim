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
        case Op::LOAD: case Op::STORE: return 0;  // memory handled separately
        default:       return cfg.fp_latency;
    }
}

long coalesce_transactions(long base, int stride, int warp_size, int line_bytes) {
    std::set<long> lines;
    for (int lane = 0; lane < warp_size; ++lane)
        lines.insert((base + (long)lane * stride) / line_bytes);
    return (long)lines.size();
}

static long mem_txn(const Instruction& inst, const GPUConfig& cfg) {
    if (inst.op != Op::LOAD && inst.op != Op::STORE) return 0;
    return coalesce_transactions(inst.addr_base, inst.addr_stride,
                                 cfg.warp_size, cfg.line_bytes);
}

static bool is_mem(Op op) { return op == Op::LOAD || op == Op::STORE; }

RunResult run_sm(const std::vector<Instruction>& program,
                 const GPUConfig& config, int num_warps, int num_registers) {
    struct WarpState { size_t pc = 0; std::vector<long> ready; };
    std::vector<WarpState> warps((size_t)num_warps);
    for (auto& w : warps) w.ready.assign((size_t)num_registers, 0);

    std::unique_ptr<Cache> l1 =
        config.l1_enabled
            ? std::make_unique<Cache>(config.l1_bytes, config.line_bytes, config.l1_assoc)
            : nullptr;

    // MSHR pool (per-SM, shared across warps): slot i is free when free_at[i] <= cycle.
    std::vector<long> mshr_free_at((size_t)std::max(1, config.max_outstanding), 0);

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

                int lat;
                if (is_mem(inst.op)) {
                    long line = inst.addr_base;
                    bool needs_dram;
                    if (l1) {
                        if (l1->probe(line)) {          // L1 hit: no DRAM, no MSHR
                            l1->access(line);            // commit LRU promote
                            ++r.l1_hits;
                            lat = config.l1_hit_latency;
                            needs_dram = false;
                        } else {
                            needs_dram = true;           // miss: don't commit yet
                        }
                    } else {
                        needs_dram = true;               // no L1: always a DRAM request
                    }

                    if (needs_dram) {
                        int free_slot = -1;
                        for (size_t m = 0; m < mshr_free_at.size(); ++m)
                            if (mshr_free_at[m] <= current_cycle) { free_slot = (int)m; break; }
                        if (free_slot < 0) continue;     // structural stall: no MSHR free
                        mshr_free_at[(size_t)free_slot] = current_cycle + config.memory_latency;
                        if (l1) { l1->access(line); ++r.l1_misses; }  // commit insert now
                        lat = config.memory_latency;
                    }
                } else {
                    lat = arith_latency(inst.op, config);
                }

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

RunResult run_single_warp(const std::vector<Instruction>& program,
                          const GPUConfig& config, int num_registers) {
    return run_sm(program, config, 1, num_registers);   // one warp == single-warp
}

}  // namespace warpsim
