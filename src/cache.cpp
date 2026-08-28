#include "warpsim/cache.hpp"
#include <algorithm>

namespace warpsim {

Cache::Cache(int total_bytes, int line_bytes, int associativity)
    : line_bytes_(line_bytes), assoc_(associativity) {
    int num_lines = total_bytes / line_bytes;
    num_sets_ = num_lines / associativity;
    if (num_sets_ < 1) num_sets_ = 1;
    ways_.assign(static_cast<size_t>(num_sets_), {});
}

bool Cache::access(long addr) {
    long line = line_addr(addr);
    int set = set_of(line);
    auto& w = ways_[static_cast<size_t>(set)];

    auto it = std::find(w.begin(), w.end(), line);
    if (it != w.end()) {
        // Hit: promote to most-recently-used (move to back).
        w.erase(it);
        w.push_back(line);
        ++hits_;
        return true;
    }

    // Miss: evict LRU (front) if the set is full, then insert at back.
    if (static_cast<int>(w.size()) >= assoc_)
        w.erase(w.begin());
    w.push_back(line);
    ++misses_;
    return false;
}

}  // namespace warpsim
