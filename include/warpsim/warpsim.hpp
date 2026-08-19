#pragma once
#include <vector>
#include <array>
#include <cstddef>

namespace warpsim {

enum class Op { FADD, FMUL, FMA, IADD, NOP };

struct Instruction {
    Op op;
    int dst;                              // -1 = writes nothing
    std::array<int, 3> src{-1, -1, -1};   // -1 = unused source slot
};

struct GPUConfig {
    int fp_latency  = 4;   // FADD, FMUL, FMA
    int int_latency = 1;   // IADD
};

struct RunResult {
    long total_cycles = 0;
    long instructions_issued = 0;
    double ipc() const {
        return total_cycles > 0
            ? static_cast<double>(instructions_issued) / total_cycles
            : 0.0;
    }
};

RunResult run_single_warp(const std::vector<Instruction>& program,
                          const GPUConfig& config,
                          int num_registers);

}  // namespace warpsim
