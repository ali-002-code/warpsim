#pragma once
#include <vector>
#include <cstddef>

namespace warpsim {

// Set-associative, LRU-replacement cache. Models tag storage only:
// we track which lines are resident, not their data.
class Cache {
public:
    Cache(int total_bytes, int line_bytes, int associativity);

    // Access one byte address. Returns true on hit, false on miss.
    // A miss inserts the line (evicting LRU within the set if full).
    bool access(long addr);

    long hits() const   { return hits_; }
    long misses() const { return misses_; }
    int  num_sets() const { return num_sets_; }

private:
    int line_bytes_;
    int assoc_;
    int num_sets_;
    long hits_ = 0, misses_ = 0;

    // ways_[set] holds resident tags, most-recently-used at the back.
    std::vector<std::vector<long>> ways_;

    long line_addr(long addr) const { return addr / line_bytes_; }
    int  set_of(long line)    const { return static_cast<int>(line % num_sets_); }
};

}  // namespace warpsim
