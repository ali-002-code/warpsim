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
