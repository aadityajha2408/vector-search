#pragma once
// HNSW (Malkov & Yashunin, 2016) for L2-normalised vectors. distance(a, b) = 1 - dot(a, b).
//
// Layout choices favour readability over memory efficiency: every node owns a
// vector<vector<uint32_t>> of neighbour lists (one per level). A production index would use
// one flat, fixed-stride array for cache locality. That is the first optimisation to try.
//
// Thread safety: add() is single-threaded. search() is read-only and may run in parallel.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "distance.hpp"

namespace vs {

class HNSWIndex {
public:
    HNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 200, uint32_t seed = 42,
              bool use_heuristic = true)
        : dim_(dim), M_(M), M0_(2 * M), efc_(ef_construction), use_heuristic_(use_heuristic),
          level_mult_(1.0 / std::log(static_cast<double>(M))), rng_(seed) {
        if (M < 2) throw std::invalid_argument("M must be >= 2");
        if (dim == 0) throw std::invalid_argument("dim must be >= 1");
    }

    size_t dim() const { return dim_; }
    size_t size() const { return levels_.size(); }
    int max_level() const { return max_level_; }

    double level0_avg_degree() const {
        if (levels_.empty()) return 0.0;
        double total = 0;
        for (const auto& per_level : links_) total += static_cast<double>(per_level[0].size());
        return total / static_cast<double>(levels_.size());
    }

    void add(const float* vecs, size_t n) {
        data_.reserve((size() + n) * dim_);
        for (size_t i = 0; i < n; ++i) add_one(vecs + i * dim_);
    }

    // ids/scores are (nq, k) row-major, best first. score = dot product. Missing results: id -1.
    void search(const float* queries, size_t nq, size_t k, size_t ef, int64_t* ids, float* scores,
                int num_threads) const {
#ifdef _OPENMP
        const int threads = num_threads > 0 ? num_threads : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int64_t q = 0; q < static_cast<int64_t>(nq); ++q) {
            search_one(queries + static_cast<size_t>(q) * dim_, k, ef, ids + static_cast<size_t>(q) * k,
                       scores + static_cast<size_t>(q) * k);
        }
        (void)num_threads;
    }

private:
    using Pair = std::pair<float, uint32_t>;  // (distance, node id)
    using MaxHeap = std::priority_queue<Pair>;
    using MinHeap = std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>>;

    // "Have I already seen this node during this search?" without clearing an N-sized array
    // each time: bump an epoch counter instead.
    struct VisitedList {
        std::vector<uint32_t> mark;
        uint32_t epoch = 0;
        explicit VisitedList(size_t n) : mark(n, 0) {}
        void reset(size_t n) {
            if (mark.size() < n) mark.resize(n, 0);
            if (++epoch == 0) {
                std::fill(mark.begin(), mark.end(), 0);
                epoch = 1;
            }
        }
        void set(uint32_t i) { mark[i] = epoch; }
        bool seen(uint32_t i) const { return mark[i] == epoch; }
    };

    // RAII: borrow a VisitedList from the pool, give it back on scope exit.
    class VisitedHandle {
    public:
        explicit VisitedHandle(const HNSWIndex& idx) : idx_(idx) {
            {
                std::lock_guard<std::mutex> lock(idx_.pool_mutex_);
                if (!idx_.pool_.empty()) {
                    list_ = std::move(idx_.pool_.back());
                    idx_.pool_.pop_back();
                }
            }
            if (!list_) list_ = std::make_unique<VisitedList>(idx_.size());
            list_->reset(idx_.size());
        }
        ~VisitedHandle() {
            std::lock_guard<std::mutex> lock(idx_.pool_mutex_);
            idx_.pool_.push_back(std::move(list_));
        }
        VisitedHandle(const VisitedHandle&) = delete;
        VisitedHandle& operator=(const VisitedHandle&) = delete;
        VisitedList& get() { return *list_; }

    private:
        const HNSWIndex& idx_;
        std::unique_ptr<VisitedList> list_;
    };

    const float* vec(uint32_t id) const { return data_.data() + static_cast<size_t>(id) * dim_; }
    float dist(const float* q, uint32_t id) const { return 1.0f - dot(q, vec(id), dim_); }
    float dist_nodes(uint32_t a, uint32_t b) const { return 1.0f - dot(vec(a), vec(b), dim_); }

    int random_level() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const double r = -std::log(1.0 - u(rng_)) * level_mult_;  // 1-u is in (0, 1]: no log(0)
        return std::min(static_cast<int>(r), 32);
    }

    // Move to the neighbour closest to q until no neighbour improves. Used on the sparse upper layers.
    void greedy(const float* q, uint32_t& ep, float& ep_dist, int level) const {
        bool changed = true;
        while (changed) {
            changed = false;
            for (uint32_t nb : links_[ep][level]) {
                const float d = dist(q, nb);
                if (d < ep_dist) {
                    ep_dist = d;
                    ep = nb;
                    changed = true;
                }
            }
        }
    }

    // Beam search on one layer. Returns the ef closest nodes found (max-heap: farthest on top).
    MaxHeap search_layer(const float* q, uint32_t ep, float ep_dist, size_t ef, int level,
                         VisitedList& visited) const {
        MaxHeap top;   // best ef results so far
        MinHeap cand;  // frontier to expand, closest first
        visited.set(ep);
        top.emplace(ep_dist, ep);
        cand.emplace(ep_dist, ep);

        while (!cand.empty()) {
            const Pair c = cand.top();
            if (top.size() >= ef && c.first > top.top().first) break;  // frontier worse than worst result
            cand.pop();
            for (uint32_t nb : links_[c.second][level]) {
                if (visited.seen(nb)) continue;
                visited.set(nb);
                const float d = dist(q, nb);
                if (top.size() < ef || d < top.top().first) {
                    cand.emplace(d, nb);
                    top.emplace(d, nb);
                    if (top.size() > ef) top.pop();
                }
            }
        }
        return top;
    }

    // Choose up to m neighbours from candidates (distance to the base node, id).
    //  - simple: the m closest.
    //  - heuristic: take a candidate only if it is closer to the base node than to every
    //    neighbour already chosen. This spreads links across directions/clusters, which keeps
    //    graphs navigable on clustered data.
    std::vector<uint32_t> select_neighbors(std::vector<Pair> cands, size_t m) const {
        std::sort(cands.begin(), cands.end());
        std::vector<uint32_t> out;
        out.reserve(m);
        for (const Pair& c : cands) {
            if (out.size() >= m) break;
            bool keep = true;
            if (use_heuristic_) {
                for (uint32_t r : out) {
                    if (dist_nodes(c.second, r) < c.first) {
                        keep = false;
                        break;
                    }
                }
            }
            if (keep) out.push_back(c.second);
        }
        return out;
    }

    void connect(uint32_t node, uint32_t new_id, int level, size_t max_links) {
        auto& list = links_[node][level];
        if (list.size() < max_links) {
            list.push_back(new_id);
            return;
        }
        std::vector<Pair> cands;  // list is full: re-select the best max_links of (old links + new node)
        cands.reserve(list.size() + 1);
        cands.emplace_back(dist_nodes(node, new_id), new_id);
        for (uint32_t x : list) cands.emplace_back(dist_nodes(node, x), x);
        list = select_neighbors(std::move(cands), max_links);
    }

    void add_one(const float* v) {
        const uint32_t id = static_cast<uint32_t>(levels_.size());
        const int level = random_level();
        data_.insert(data_.end(), v, v + dim_);
        levels_.push_back(level);
        links_.emplace_back(static_cast<size_t>(level) + 1);

        if (entry_ < 0) {  // first node
            entry_ = id;
            max_level_ = level;
            return;
        }

        VisitedHandle handle(*this);
        uint32_t ep = static_cast<uint32_t>(entry_);
        float ep_dist = dist(v, ep);

        for (int l = max_level_; l > level; --l) greedy(v, ep, ep_dist, l);

        for (int l = std::min(level, max_level_); l >= 0; --l) {
            if (l < std::min(level, max_level_)) handle.get().reset(size());
            MaxHeap top = search_layer(v, ep, ep_dist, efc_, l, handle.get());

            std::vector<Pair> cands;
            cands.reserve(top.size());
            while (!top.empty()) {
                cands.push_back(top.top());
                top.pop();
            }
            const Pair best = *std::min_element(cands.begin(), cands.end());
            ep = best.second;
            ep_dist = best.first;

            const size_t max_links = (l == 0) ? M0_ : M_;
            links_[id][l] = select_neighbors(cands, M_);
            for (uint32_t nb : links_[id][l]) connect(nb, id, l, max_links);
        }

        if (level > max_level_) {
            max_level_ = level;
            entry_ = id;
        }
    }

    void search_one(const float* q, size_t k, size_t ef, int64_t* ids, float* scores) const {
        std::fill(ids, ids + k, int64_t{-1});
        std::fill(scores, scores + k, -std::numeric_limits<float>::infinity());
        if (entry_ < 0) return;

        VisitedHandle handle(*this);
        uint32_t ep = static_cast<uint32_t>(entry_);
        float ep_dist = dist(q, ep);
        for (int l = max_level_; l > 0; --l) greedy(q, ep, ep_dist, l);

        MaxHeap top = search_layer(q, ep, ep_dist, std::max(ef, k), 0, handle.get());
        while (top.size() > k) top.pop();  // keep the k closest
        for (size_t j = top.size(); j > 0; --j) {
            ids[j - 1] = top.top().second;
            scores[j - 1] = 1.0f - top.top().first;
            top.pop();
        }
    }

    size_t dim_, M_, M0_, efc_;
    bool use_heuristic_;
    double level_mult_;
    std::mt19937 rng_;

    std::vector<float> data_;                                // n * dim, contiguous
    std::vector<int> levels_;                                // top level of each node
    std::vector<std::vector<std::vector<uint32_t>>> links_;  // links_[node][level] -> neighbour ids
    int64_t entry_ = -1;
    int max_level_ = -1;

    mutable std::mutex pool_mutex_;
    mutable std::vector<std::unique_ptr<VisitedList>> pool_;
};

}  // namespace vs
