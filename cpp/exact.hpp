#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "distance.hpp"

namespace vs {

// Brute-force top-k inner-product search.
//
// Two ideas make it faster than the textbook loop:
//  1. A size-k min-heap per query: O(N log k) instead of sorting all N scores.
//  2. Query tiling: queries are processed in blocks of kBlock. Each corpus vector is loaded
//     from memory once and reused for the whole block while it is hot in cache, so memory
//     traffic drops ~kBlock-fold. Blocks run in parallel with OpenMP.
class ExactIndex {
public:
    static constexpr size_t kBlock = 8;

    explicit ExactIndex(size_t dim) : dim_(dim) {}

    size_t dim() const { return dim_; }
    size_t size() const { return n_; }

    void add(const float* vecs, size_t n) {
        data_.insert(data_.end(), vecs, vecs + n * dim_);
        n_ += n;
    }

    // ids/scores are (nq, k) row-major, best first. Missing results (k > size) get id -1.
    void search(const float* queries, size_t nq, size_t k, int64_t* ids, float* scores,
                int num_threads) const {
        const int64_t n_blocks = static_cast<int64_t>((nq + kBlock - 1) / kBlock);
#ifdef _OPENMP
        const int threads = num_threads > 0 ? num_threads : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic) num_threads(threads)
#endif
        for (int64_t b = 0; b < n_blocks; ++b) {
            const size_t first = static_cast<size_t>(b) * kBlock;
            const size_t count = std::min(kBlock, nq - first);
            search_block(queries + first * dim_, count, k, ids + first * k, scores + first * k);
        }
        (void)num_threads;
    }

private:
    using Entry = std::pair<float, int64_t>;  // (score, row)
    using MinHeap = std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>>;

    void search_block(const float* qs, size_t count, size_t k, int64_t* ids, float* scores) const {
        const float neg_inf = -std::numeric_limits<float>::infinity();
        std::vector<MinHeap> heaps(count);
        std::vector<float> floor(count, neg_inf);  // score a candidate must beat to enter heap b

        for (size_t i = 0; i < n_; ++i) {
            const float* row = data_.data() + i * dim_;
            for (size_t b = 0; b < count; ++b) {
                const float s = dot(qs + b * dim_, row, dim_);
                if (s > floor[b]) {
                    if (heaps[b].size() >= k) heaps[b].pop();
                    heaps[b].emplace(s, static_cast<int64_t>(i));
                    if (heaps[b].size() >= k) floor[b] = heaps[b].top().first;
                }
            }
        }

        for (size_t b = 0; b < count; ++b) {
            int64_t* out_ids = ids + b * k;
            float* out_scores = scores + b * k;
            std::fill(out_ids, out_ids + k, int64_t{-1});
            std::fill(out_scores, out_scores + k, neg_inf);
            // Heap pops the weakest first, so fill from the back to end up best-first.
            for (size_t j = heaps[b].size(); j > 0; --j) {
                out_scores[j - 1] = heaps[b].top().first;
                out_ids[j - 1] = heaps[b].top().second;
                heaps[b].pop();
            }
        }
    }

    size_t dim_;
    size_t n_ = 0;
    std::vector<float> data_;
};

}  // namespace vs
