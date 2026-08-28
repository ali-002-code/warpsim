#include "warpsim/warpsim.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace warpsim;

static int failures = 0;
static void check_close(const char* name, double got, double expect, double tol) {
    bool ok = std::fabs(got - expect) <= tol;
    std::printf("[%s] %s: got %.4f, expected %.4f\n",
                ok ? "PASS" : "FAIL", name, got, expect);
    if (!ok) ++failures;
}

static std::vector<Instruction> loads(int n, int stride_bytes) {
    std::vector<Instruction> p;
    for (int k = 0; k < n; ++k)
        p.push_back(Instruction{Op::LOAD, 1, {-1,-1,-1},
                                (long)k * 4096, stride_bytes});
    return p;
}

int main() {
    const int N = 20000;
    GPUConfig cfg;
    cfg.l1_enabled = false;
    cfg.memory_latency = 100;
    cfg.max_outstanding = 1000000;
    cfg.dram_txns_per_cycle = 2;
    cfg.warp_size = 32;
    cfg.line_bytes = 128;
    cfg.issue_width = 8;

    RunResult coal = run_single_warp(loads(N, 4), cfg, 4);
    check_close("coalesced IPC (bw-bound)", coal.ipc(), 1.0, 0.02);  // serialised channel: 1-txn access still occupies a full cycle

    RunResult uncoal = run_single_warp(loads(N, 128), cfg, 4);
    check_close("uncoalesced IPC (bw-bound)", uncoal.ipc(), 2.0/32.0, 0.01);

    std::printf(failures == 0
        ? "\nv3b: DRAM bandwidth throttles by transaction count; coalescing = 16x under serialised-channel model.\n"
        : "\n%d check(s) failed.\n", failures);
    return failures == 0 ? 0 : 1;
}
