#include "warpsim/cache.hpp"
#include <cstdio>

using namespace warpsim;

static int failures = 0;
static void check(const char* name, long got, long expect) {
    bool ok = (got == expect);
    std::printf("[%s] %s: got %ld, expected %ld\n",
                ok ? "PASS" : "FAIL", name, got, expect);
    if (!ok) ++failures;
}

int main() {
    const int LINE = 64;

    // --- Test 1: stride-1 over one line. ---
    // 64 sequential byte accesses, 64B line: first misses (cold), rest hit.
    {
        Cache c(1024, LINE, 4);
        for (int i = 0; i < 64; ++i) c.access(i);
        check("stride1 misses", c.misses(), 1);
        check("stride1 hits",   c.hits(),   63);
    }

    // --- Test 2: line-stride, all cold. ---
    // 100 accesses each one line apart -> every access a new line -> 100 misses.
    {
        Cache c(1 << 20, LINE, 4);   // big enough that nothing evicts
        for (int i = 0; i < 100; ++i) c.access((long)i * LINE);
        check("linestride misses", c.misses(), 100);
        check("linestride hits",   c.hits(),   0);
    }

    // --- Test 3: LRU eviction in one set. ---
    // Cache with exactly 1 set, associativity 2. Lines A,B fill it.
    // Access A,B (2 cold misses), then C evicts A (LRU), then A misses again.
    {
        // 2 lines total, 1 set, assoc 2  -> total = 2*LINE, sets = 1.
        Cache c(2 * LINE, LINE, 2);
        c.access(0 * LINE);   // A: miss
        c.access(1 * LINE);   // B: miss
        c.access(0 * LINE);   // A: hit (now A is MRU, B is LRU)
        c.access(2 * LINE);   // C: miss, evicts B (LRU)
        c.access(1 * LINE);   // B: miss again (was evicted)
        c.access(0 * LINE);   // A: hit (still resident)
        check("lru misses", c.misses(), 5);  // A,B,C,B,A (traced by hand)
        check("lru hits",   c.hits(),   1);  // only the step-3 A hit
    }

    // --- Test 4: set mapping. ---
    // 4 lines, assoc 1 (direct-mapped) -> 4 sets. Lines 0 and 4 collide in set 0.
    {
        Cache c(4 * LINE, LINE, 1);
        c.access(0 * LINE);   // line 0 -> set 0: miss
        c.access(4 * LINE);   // line 4 -> set 0: miss, evicts line 0
        c.access(0 * LINE);   // line 0 -> set 0: miss again (thrashing)
        check("conflict misses", c.misses(), 3);
        check("conflict hits",   c.hits(),   0);
    }

    std::printf(failures == 0 ? "\nv2b: set-associative LRU cache validated.\n"
                              : "\n%d check(s) failed.\n", failures);
    return failures == 0 ? 0 : 1;
}
