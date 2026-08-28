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

// N independent LOADs to distinct lines, L1 OFF so every one hits DRAM.
static std::vector<Instruction> loads(int n, int stride_lines, int line_bytes) {
    std::vector<Instruction> p;
    for (int k = 0; k < n; ++k)
        p.push_back(Instruction{Op::LOAD, 1, {-1,-1,-1},
                                (long)k * stride_lines * line_bytes, 4});
    return p;
}

int main() {
    const int N = 10000, W = 100, LINE = 128;
    GPUConfig cfg;
    cfg.l1_enabled = false;          // force all DRAM
    cfg.memory_latency = W;

    // Finite MSHRs: IPC capped at M/W by Little's Law, regardless of occupancy.
    cfg.max_outstanding = 10;
    RunResult limited = run_single_warp(loads(N, 1, LINE), cfg, 4);
    check_close("MSHR-limited IPC", limited.ipc(), 10.0 / W, 0.003);  // -> 0.10

    // Effectively infinite MSHRs: loads pipeline freely, IPC -> ~1.0.
    cfg.max_outstanding = 100000;
    RunResult infinite = run_single_warp(loads(N, 1, LINE), cfg, 4);
    check_close("MSHR-infinite IPC", infinite.ipc(),
                (double)N / (N - 1 + W), 0.005);                     // -> ~0.99

    std::printf(failures == 0
        ? "\nv3a: outstanding-request limit throttles MLP to M/W (Little's Law).\n"
        : "\n%d check(s) failed.\n", failures);
    return failures == 0 ? 0 : 1;
}
