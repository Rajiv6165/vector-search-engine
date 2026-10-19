#pragma once

#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>

namespace hnsw {

struct Candidate {
    float dist;
    int id;
    bool operator>(const Candidate& other) const {
        return dist > other.dist;
    }
};

struct Result {
    float dist;
    int id;
    bool operator<(const Result& other) const {
        return dist < other.dist;
    }
};

inline float l2_dist(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline float cosine_dist(const float* a, const float* b, int dim) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int d = 0; d < dim; ++d) {
        dot += a[d] * b[d];
        norm_a += a[d] * a[d];
        norm_b += b[d] * b[d];
    }
    if (norm_a <= 0.0f || norm_b <= 0.0f) return 1.0f;
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return 1.0f - (dot / denom);
}

inline float compute_distance(const float* a, const float* b, int dim, int metric) {
    if (metric == 0) {
        return l2_dist(a, b, dim);
    } else {
        return cosine_dist(a, b, dim);
    }
}

inline std::vector<std::pair<int, float>> search_layer(
    const float* query,
    const int* entry_points,
    int num_entry_points,
    int ef,
    const float* vectors,
    int num_vectors,
    int dim,
    const int* edges,
    const int* degrees,
    int max_neighbors,
    int metric,
    int* num_visited = nullptr
) {
    std::vector<uint8_t> visited(num_vectors, 0);
    int visited_cnt = 0;

    // C is min-heap: candidate closest to query is popped first
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> C;
    // W is max-heap: candidate farthest from query is at top
    std::priority_queue<Result, std::vector<Result>, std::less<Result>> W;

    for (int i = 0; i < num_entry_points; ++i) {
        int ep = entry_points[i];
        if (ep < 0 || ep >= num_vectors) continue;
        visited[ep] = 1;
        visited_cnt++;
        float d = compute_distance(query, vectors + (size_t)ep * dim, dim, metric);
        C.push({d, ep});
        W.push({d, ep});
    }

    while (!C.empty()) {
        Candidate curr = C.top();
        C.pop();

        float f_dist = W.top().dist;
        if (curr.dist > f_dist) {
            break;
        }

        int u = curr.id;
        if (u < 0 || u >= num_vectors) continue;
        int deg = degrees[u];
        const int* nbrs = edges + (size_t)u * max_neighbors;

        for (int i = 0; i < deg; ++i) {
            int n = nbrs[i];
            if (n < 0 || n >= num_vectors) continue;
            if (visited[n]) continue;
            visited[n] = 1;
            visited_cnt++;

            float d = compute_distance(query, vectors + (size_t)n * dim, dim, metric);
            f_dist = W.top().dist;

            if (W.size() < (size_t)ef || d < f_dist) {
                C.push({d, n});
                W.push({d, n});
                if (W.size() > (size_t)ef) {
                    W.pop();
                }
            }
        }
    }

    if (num_visited != nullptr) {
        *num_visited = visited_cnt;
    }

    std::vector<std::pair<int, float>> results;
    results.reserve(W.size());
    while (!W.empty()) {
        results.push_back({W.top().id, W.top().dist});
        W.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}

inline std::pair<int, float> greedy_search_layer(
    const float* query,
    int curr_obj,
    float curr_dist,
    const float* vectors,
    int num_vectors,
    int dim,
    const int* edges,
    const int* degrees,
    int max_neighbors,
    int metric
) {
    bool changed = true;
    while (changed) {
        changed = false;
        if (curr_obj < 0 || curr_obj >= num_vectors) break;
        int deg = degrees[curr_obj];
        const int* nbrs = edges + (size_t)curr_obj * max_neighbors;
        for (int i = 0; i < deg; ++i) {
            int n = nbrs[i];
            if (n < 0 || n >= num_vectors) continue;
            float d = compute_distance(query, vectors + (size_t)n * dim, dim, metric);
            if (d < curr_dist) {
                curr_dist = d;
                curr_obj = n;
                changed = true;
            }
        }
    }
    return {curr_obj, curr_dist};
}

inline std::vector<std::pair<int, float>> select_neighbors_heuristic(
    const float* base_vec,
    const std::vector<std::pair<int, float>>& candidates,
    int M,
    const float* vectors,
    int num_vectors,
    int dim,
    int metric,
    bool keep_pruned
) {
    // Sort candidates by distance ascending
    std::vector<std::pair<int, float>> sorted_cands = candidates;
    std::sort(sorted_cands.begin(), sorted_cands.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second < b.second;
    });

    std::vector<std::pair<int, float>> R;
    R.reserve(M);
    std::vector<std::pair<int, float>> W_d;

    for (const auto& cand : sorted_cands) {
        if ((int)R.size() >= M) break;
        int e_id = cand.first;
        float dist_e_q = cand.second;
        if (e_id < 0 || e_id >= num_vectors) continue;
        const float* e_vec = vectors + (size_t)e_id * dim;

        bool is_diverse = true;
        for (const auto& r : R) {
            int r_id = r.first;
            if (r_id < 0 || r_id >= num_vectors) continue;
            const float* r_vec = vectors + (size_t)r_id * dim;
            float dist_e_r = compute_distance(e_vec, r_vec, dim, metric);
            if (dist_e_r < dist_e_q) {
                is_diverse = false;
                break;
            }
        }

        if (is_diverse) {
            R.push_back(cand);
        } else {
            W_d.push_back(cand);
        }
    }

    if (keep_pruned && (int)R.size() < M) {
        for (const auto& cand : W_d) {
            if ((int)R.size() >= M) break;
            R.push_back(cand);
        }
    }

    return R;
}

} // namespace hnsw
