#include "warpsim/warpsim.hpp"
#include <algorithm>
#include <set>

namespace warpsim {

static int latency_of(Op op, const GPUConfig& cfg) {
    switch (op) {
        case Op::IADD:  return cfg.int_latency;
        case Op::LOAD:  return cfg.memory_latency;
        case Op::STORE: return cfg.memory_latency;
        case Op::NOP:   return 0;
        default:        return cfg.fp_latency;   // FADD, FMUL, FMA
    }
}

long coalesce_transactions(long base, int stride, int warp_size, int line_bytes) {
    std::set<long> lines;
    for (int lane = 0; lane < warp_size; ++lane) {
        long addr = base + static_cast<long>(lane) * stride;
        lines.insert(addr / line_bytes);   // which cache line this lane hits
    }
    return static_cast<long>(lines.size());
}

static long mem_txn(const Instruction& inst, const GPUConfig& cfg) {
    if (inst.op != Op::LOAD && inst.op != Op::STORE) return 0;
    return coalesce_transactions(inst.addr_base, inst.addr_stride,
                                 cfg.warp_size, cfg.line_bytes);
}

RunResult run_single_warp(const std::vector<Instruction>& program,
                          const GPUConfig& config, int num_registers) {
    std::vector<long> ready(static_cast<size_t>(num_registers), 0);
    long current_cycle = 0, max_completion = 0, txns = 0;
    size_t pc = 0;

    while (pc < program.size()) {
        const Instruction& inst = program[pc];
        bool eligible = true;
        for (int s : inst.src)
            if (s >= 0 && ready[s] > current_cycle) { eligible = false; break; }

        if (eligible) {
            long completion = current_cycle + latency_of(inst.op, config);
            if (inst.dst >= 0) ready[inst.dst] = completion;
            max_completion = std::max(max_completion, completion);
            txns += mem_txn(inst, config);
            ++pc;
        }
        ++current_cycle;
    }

    RunResult r;
    r.total_cycles = max_completion;
    r.instructions_issued = static_cast<long>(program.size());
    r.memory_transactions = txns;
    return r;
}

RunResult run_sm(const std::vector<Instruction>& program,
                 const GPUConfig& config, int num_warps, int num_registers) {
    struct WarpState { size_t pc = 0; std::vector<long> ready; };
    std::vector<WarpState> warps(static_cast<size_t>(num_warps));
    for (auto& w : warps) w.ready.assign(static_cast<size_t>(num_registers), 0);

    long current_cycle = 0, max_completion = 0, issued_total = 0, txns = 0;
    const long target = static_cast<long>(program.size()) * num_warps;
    size_t rr = 0;

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

                long completion = current_cycle + latency_of(inst.op, config);
                if (inst.dst >= 0) w.ready[inst.dst] = completion;
                max_completion = std::max(max_completion, completion);
                txns += mem_txn(inst, config);
                ++w.pc; ++issued_total; rr = i + 1;
                issued_this_slot = true;
                break;
            }
            if (!issued_this_slot) break;
        }
        ++current_cycle;
    }

    RunResult r;
    r.total_cycles = max_completion;
    r.instructions_issued = issued_total;
    r.memory_transactions = txns;
    return r;
}

}  // namespace warpsim
