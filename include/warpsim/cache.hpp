#pragma once
#include <vector>
#include <cstddef>

namespace warpsim {

class Cache {
public:
    Cache(int total_bytes, int line_bytes, int associativity);

    bool probe(long addr) const;   // would this hit? no mutation.
    bool access(long addr);        // perform access, update LRU/insert, return hit.

    long hits() const   { return hits_; }
    long misses() const { return misses_; }
    int  num_sets() const { return num_sets_; }

private:
    int line_bytes_, assoc_, num_sets_;
    long hits_ = 0, misses_ = 0;
    std::vector<std::vector<long>> ways_;

    long line_addr(long addr) const { return addr / line_bytes_; }
    int  set_of(long line)    const { return static_cast<int>(line % num_sets_); }
};

}  // namespace warpsim
