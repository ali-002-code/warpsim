#pragma once
#include <vector>
#include <array>
#include <cstddef>

namespace warpsim {

enum class Op { FADD, FMUL, FMA, IADD, LOAD, STORE, NOP };

struct Instruction {
    Op op;
    int dst;                              // -1 = writes nothing
    std::array<int, 3> src{-1, -1, -1};   // -1 = unused source slot
    long addr_base = 0;                   // lane-0 byte address (LOAD/STORE)
    int  addr_stride = 0;                 // byte gap between consecutive lanes
};

struct GPUConfig {
    int fp_latency  = 4;      // FADD, FMUL, FMA
    int int_latency = 1;      // IADD
    int issue_width = 1;      // instructions issued per cycle across the SM
    int warp_size   = 32;     // lanes per warp
    int line_bytes  = 128;    // cache line size in bytes
    int memory_latency = 300; // cycles for a LOAD to complete (fixed, for now)
};

struct RunResult {
    long total_cycles = 0;
    long instructions_issued = 0;
    long memory_transactions = 0;   // total coalesced cache-line accesses
    double ipc() const {
        return total_cycles > 0
            ? static_cast<double>(instructions_issued) / total_cycles
            : 0.0;
    }
};

// Distinct cache lines touched by lanes 0..warp_size-1 of one memory access.
long coalesce_transactions(long base, int stride, int warp_size, int line_bytes);

RunResult run_single_warp(const std::vector<Instruction>& program,
                          const GPUConfig& config, int num_registers);

RunResult run_sm(const std::vector<Instruction>& program,
                 const GPUConfig& config, int num_warps, int num_registers);

}  // namespace warpsim
