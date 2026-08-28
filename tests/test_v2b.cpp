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
    // Two loads to the SAME base line, independent (no dep stall between them).
    // With L1 on: first misses (300+ path), second hits. So 1 hit, 1 miss.
    GPUConfig cfg;
    cfg.l1_enabled = true;
    cfg.l1_hit_latency = 20;
    cfg.memory_latency = 300;

    std::vector<Instruction> prog = {
        Instruction{Op::LOAD, 1, {-1,-1,-1}, 0, 4},   // base line 0
        Instruction{Op::LOAD, 2, {-1,-1,-1}, 0, 4},   // same base line 0
    };
    RunResult r = run_single_warp(prog, cfg, 4);
    check("l1 hits",   r.l1_hits,   1);
    check("l1 misses", r.l1_misses, 1);

    // L1 disabled: no hit/miss accounting, both take full memory latency.
    GPUConfig off = cfg; off.l1_enabled = false;
    RunResult r2 = run_single_warp(prog, off, 4);
    check("l1-off hits",   r2.l1_hits,   0);
    check("l1-off misses", r2.l1_misses, 0);

    std::printf(failures == 0 ? "\nv2b wiring: L1 changes memory latency.\n"
                              : "\n%d check(s) failed.\n", failures);
    return failures == 0 ? 0 : 1;
}
