#include "warpsim/warpsim.hpp"
#include <cstdio>
#include <vector>

using namespace warpsim;

static int failures = 0;
static void check(const char* name, long got, long expect) {
    bool ok = (got == expect);
    std::printf("[%s] %s: got %ld, expected %ld\n",
                ok ? "PASS" : "FAIL", name, got, expect);
    if (!ok) ++failures;
}

int main() {
    const int WS = 32, LINE = 128;

    // Contiguous: 32 lanes x 4 bytes = 128 bytes = exactly one 128B line.
    check("coalesce contiguous", coalesce_transactions(0, 4, WS, LINE), 1);

    // One line per lane: stride = line size -> every lane in its own line.
    check("coalesce line-per-lane", coalesce_transactions(0, 128, WS, LINE), 32);

    // 32 x 4 = 128 bytes over 64B lines -> spans two lines.
    check("coalesce two-line", coalesce_transactions(0, 4, WS, 64), 2);

    // Memory dependency: LOAD R1, then FMA using R1. Cost = mem_lat + fp_lat.
    GPUConfig cfg;  // memory_latency=300, fp_latency=4
    std::vector<Instruction> prog = {
        Instruction{Op::LOAD, 1, {-1, -1, -1}, 0, 4},
        Instruction{Op::FMA,  2, { 1,  1, -1}},
    };
    RunResult r = run_single_warp(prog, cfg, 4);
    check("mem-dep cycles", r.total_cycles, 300 + 4);
    check("mem-dep transactions", r.memory_transactions, 1);  // contiguous load

    // Transaction plumbing through the SM: 4 warps, each one contiguous LOAD.
    std::vector<Instruction> ld = { Instruction{Op::LOAD, 1, {-1,-1,-1}, 0, 4} };
    RunResult s = run_sm(ld, cfg, 4, 4);
    check("sm transactions", s.memory_transactions, 4);

    std::printf(failures == 0 ? "\nv2a: addresses + coalescing validated.\n"
                              : "\n%d check(s) failed.\n", failures);
    return failures == 0 ? 0 : 1;
}
