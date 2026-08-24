#include "warpsim/warpsim.hpp"
#include <algorithm>

namespace warpsim {

static int latency_of(Op op, const GPUConfig& cfg) {
    switch (op) {
        case Op::IADD: return cfg.int_latency;
        case Op::NOP:  return 0;
        default:       return cfg.fp_latency;  // FADD, FMUL, FMA
    }
}

RunResult run_single_warp(const std::vector<Instruction>& program,
                          const GPUConfig& config,
                          int num_registers) {
    // ready[r] = earliest cycle register r holds a usable value.
    std::vector<long> ready(static_cast<size_t>(num_registers), 0);

    long current_cycle  = 0;
    long max_completion = 0;
    size_t pc = 0;

    while (pc < program.size()) {
        const Instruction& inst = program[pc];

        // Eligible only if every used source is ready by this cycle.
        bool eligible = true;
        for (int s : inst.src) {
            if (s >= 0 && ready[s] > current_cycle) { eligible = false; break; }
        }

        if (eligible) {
            long completion = current_cycle + latency_of(inst.op, config);
            if (inst.dst >= 0) ready[inst.dst] = completion;
            max_completion = std::max(max_completion, completion);
            ++pc;                    // in-order, issue width 1
        }
        ++current_cycle;             // one issue attempt per cycle
    }

    RunResult r;
    r.total_cycles = max_completion;
    r.instructions_issued = static_cast<long>(program.size());
    return r;
}

}  // namespace warpsim

// ---- v1: multi-warp SM ------------------------------------------------

namespace warpsim {

RunResult run_sm(const std::vector<Instruction>& program,
                 const GPUConfig& config,
                 int num_warps,
                 int num_registers) {
    struct WarpState {
        size_t pc = 0;
        std::vector<long> ready;
    };

    std::vector<WarpState> warps(static_cast<size_t>(num_warps));
    for (auto& w : warps)
        w.ready.assign(static_cast<size_t>(num_registers), 0);

    long current_cycle = 0;
    long max_completion = 0;
    long issued_total = 0;
    const long target = static_cast<long>(program.size()) * num_warps;
    size_t rr = 0;  // round-robin cursor: who to start scanning from

    while (issued_total < target) {
        int slots = config.issue_width;

        // Try to fill each issue slot this cycle from an eligible warp.
        for (int slot = 0; slot < slots; ++slot) {
            bool issued_this_slot = false;

            for (size_t scan = 0; scan < warps.size(); ++scan) {
                size_t i = (rr + scan) % warps.size();
                WarpState& w = warps[i];
                if (w.pc >= program.size()) continue;      // warp finished

                const Instruction& inst = program[w.pc];
                bool eligible = true;
                for (int s : inst.src)
                    if (s >= 0 && w.ready[s] > current_cycle) { eligible = false; break; }
                if (!eligible) continue;

                long completion = current_cycle + latency_of(inst.op, config);
                if (inst.dst >= 0) w.ready[inst.dst] = completion;
                max_completion = std::max(max_completion, completion);
                ++w.pc;
                ++issued_total;
                rr = i + 1;                 // next cycle, start after this warp
                issued_this_slot = true;
                break;
            }

            if (!issued_this_slot) break;   // nobody eligible; slot goes idle
        }

        ++current_cycle;
    }

    RunResult r;
    r.total_cycles = max_completion;
    r.instructions_issued = issued_total;
    return r;
}

}  // namespace warpsim
