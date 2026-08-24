#include "warpsim/warpsim.hpp"
#include <cstdio>
#include <vector>
#include <cmath>

using namespace warpsim;

// Dependent chain: op_k reads reg_k, writes reg_(k+1). IPC of one such warp = 1/L.
static std::vector<Instruction> build_dependent(int n) {
    std::vector<Instruction> p;
    for (int k = 0; k < n; ++k)
        p.push_back(Instruction{Op::FADD, k + 1, {k, -1, -1}});
    return p;
}

static int failures = 0;
static void check_close(const char* name, double got, double expect, double tol) {
    bool ok = std::fabs(got - expect) <= tol;
    std::printf("[%s] %s: got %.4f, expected %.4f\n",
                ok ? "PASS" : "FAIL", name, got, expect);
    if (!ok) ++failures;
}

int main() {
    const int N = 2000, L = 4;
    GPUConfig cfg; cfg.fp_latency = L;

    for (int W : {1, 2, 4, 8}) {
        RunResult r = run_sm(build_dependent(N), cfg, W, N + 1);
        double expect = std::min(1.0, (double)W / L);
        char label[64];
        std::snprintf(label, sizeof(label), "IPC W=%d", W);
        // tolerance loosens slightly for small W where startup/drain matters
        check_close(label, r.ipc(), expect, 0.02);
    }

    std::printf(failures == 0 ? "\nv1: latency hiding matches min(1, W/L).\n"
                              : "\n%d check(s) failed.\n", failures);
    return failures == 0 ? 0 : 1;
}
