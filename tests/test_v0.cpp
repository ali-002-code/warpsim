#include "warpsim/warpsim.hpp"
#include <cstdio>
#include <vector>

using namespace warpsim;

// Independent: op_k reads reg0 (never written), writes reg_(k+1). No stalls.
static std::vector<Instruction> build_independent(int n) {
    std::vector<Instruction> p;
    for (int k = 0; k < n; ++k)
        p.push_back(Instruction{Op::FADD, k + 1, {0, -1, -1}});
    return p;
}

// Dependent: op_k reads reg_k, writes reg_(k+1). Each waits on the previous.
static std::vector<Instruction> build_dependent(int n) {
    std::vector<Instruction> p;
    for (int k = 0; k < n; ++k)
        p.push_back(Instruction{Op::FADD, k + 1, {k, -1, -1}});
    return p;
}

static int failures = 0;
static void check(const char* name, long got, long expect) {
    bool ok = (got == expect);
    std::printf("[%s] %s: got %ld, expected %ld\n",
                ok ? "PASS" : "FAIL", name, got, expect);
    if (!ok) ++failures;
}

int main() {
    const int N = 1000, L = 4;
    GPUConfig cfg; cfg.fp_latency = L;

    RunResult indep = run_single_warp(build_independent(N), cfg, N + 1);
    check("independent cycles", indep.total_cycles, (N - 1) + L);

    RunResult dep = run_single_warp(build_dependent(N), cfg, N + 1);
    check("dependent cycles", dep.total_cycles, (long)N * L);

    std::printf("IPC independent = %.4f (expect ~%.4f)\n",
                indep.ipc(), (double)N / ((N - 1) + L));
    std::printf("IPC dependent   = %.4f (expect %.4f)\n",
                dep.ipc(), 1.0 / L);

    std::printf(failures == 0 ? "\nAll v0 invariants hold.\n"
                              : "\n%d check(s) failed.\n", failures);
    return failures == 0 ? 0 : 1;
}
