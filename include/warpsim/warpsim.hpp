#pragma once
#include <vector>
#include <array>
#include <cstddef>

namespace warpsim {

enum class Op { FADD, FMUL, FMA, IADD, LOAD, STORE, NOP };

struct Instruction {
    Op op;
    int dst;
    std::array<int, 3> src{-1, -1, -1};
    long addr_base = 0;
    int  addr_stride = 0;
};

struct GPUConfig {
    int fp_latency  = 4;
    int int_latency = 1;
    int issue_width = 1;
    int warp_size   = 32;
    int line_bytes  = 128;
    int memory_latency = 300;   // L1 miss -> DRAM, in cycles

    // L1 cache (v2b)
    bool l1_enabled   = true;
    int  l1_bytes     = 16 * 1024;
    int  l1_assoc     = 4;
    int  l1_hit_latency = 20;    // cycles on an L1 hit
};

struct RunResult {
    long total_cycles = 0;
    long instructions_issued = 0;
    long memory_transactions = 0;
    long l1_hits = 0;
    long l1_misses = 0;
    double ipc() const {
        return total_cycles > 0
            ? static_cast<double>(instructions_issued) / total_cycles : 0.0;
    }
    double l1_hit_rate() const {
        long t = l1_hits + l1_misses;
        return t > 0 ? static_cast<double>(l1_hits) / t : 0.0;
    }
};

long coalesce_transactions(long base, int stride, int warp_size, int line_bytes);

RunResult run_single_warp(const std::vector<Instruction>& program,
                          const GPUConfig& config, int num_registers);

RunResult run_sm(const std::vector<Instruction>& program,
                 const GPUConfig& config, int num_warps, int num_registers);

}  // namespace warpsim
