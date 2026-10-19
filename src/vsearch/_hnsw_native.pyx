# distutils: language = c++
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False

from libcpp.vector cimport vector
from libcpp.pair cimport pair
import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "hnsw_core.hpp" namespace "hnsw" nogil:
    vector[pair[int, float]] search_layer(
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
        int metric
    )

    pair[int, float] greedy_search_layer(
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
    )

    vector[pair[int, float]] select_neighbors_heuristic(
        const float* base_vec,
        const vector[pair[int, float]]& candidates,
        int M,
        const float* vectors,
        int num_vectors,
        int dim,
        int metric,
        bint keep_pruned
    )

    float compute_distance(
        const float* a,
        const float* b,
        int dim,
        int metric
    )

def search_layer_native(
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] query,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] entry_points,
    int ef,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] vectors,
    cnp.ndarray[cnp.int32_t, ndim=2, mode="c"] edges,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] degrees,
    int metric
):
    cdef int num_entry_points = entry_points.shape[0]
    cdef int num_vectors = vectors.shape[0]
    cdef int dim = vectors.shape[1]
    cdef int max_neighbors = edges.shape[1]
    cdef const float* query_ptr = <const float*>query.data
    cdef const int* ep_ptr = <const int*>entry_points.data
    cdef const float* vecs_ptr = <const float*>vectors.data
    cdef const int* edges_ptr = <const int*>edges.data
    cdef const int* degrees_ptr = <const int*>degrees.data
    cdef vector[pair[int, float]] c_results

    with nogil:
        c_results = search_layer(
            query_ptr,
            ep_ptr,
            num_entry_points,
            ef,
            vecs_ptr,
            num_vectors,
            dim,
            edges_ptr,
            degrees_ptr,
            max_neighbors,
            metric
        )

    cdef list py_results = []
    cdef size_t i
    for i in range(c_results.size()):
        py_results.append((c_results[i].first, c_results[i].second))
    return py_results

def greedy_search_layer_native(
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] query,
    int curr_obj,
    float curr_dist,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] vectors,
    cnp.ndarray[cnp.int32_t, ndim=2, mode="c"] edges,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] degrees,
    int metric
):
    cdef int num_vectors = vectors.shape[0]
    cdef int dim = vectors.shape[1]
    cdef int max_neighbors = edges.shape[1]
    cdef const float* query_ptr = <const float*>query.data
    cdef const float* vecs_ptr = <const float*>vectors.data
    cdef const int* edges_ptr = <const int*>edges.data
    cdef const int* degrees_ptr = <const int*>degrees.data
    cdef pair[int, float] c_res

    with nogil:
        c_res = greedy_search_layer(
            query_ptr,
            curr_obj,
            curr_dist,
            vecs_ptr,
            num_vectors,
            dim,
            edges_ptr,
            degrees_ptr,
            max_neighbors,
            metric
        )
    return (c_res.first, c_res.second)

def select_neighbors_heuristic_native(
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] base_vec,
    list candidates,
    int M,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] vectors,
    int metric,
    bint keep_pruned
):
    cdef int num_vectors = vectors.shape[0]
    cdef int dim = vectors.shape[1]
    cdef const float* base_ptr = <const float*>base_vec.data
    cdef const float* vecs_ptr = <const float*>vectors.data
    cdef vector[pair[int, float]] c_candidates
    c_candidates.reserve(len(candidates))

    for item in candidates:
        c_candidates.push_back((<int>item[0], <float>item[1]))

    cdef vector[pair[int, float]] c_results
    with nogil:
        c_results = select_neighbors_heuristic(
            base_ptr,
            c_candidates,
            M,
            vecs_ptr,
            num_vectors,
            dim,
            metric,
            keep_pruned
        )

    cdef list py_results = []
    cdef size_t i
    for i in range(c_results.size()):
        py_results.append((c_results[i].first, c_results[i].second))
    return py_results

def compute_distance_native(
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] a,
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] b,
    int metric
):
    cdef int dim = a.shape[0]
    cdef const float* a_ptr = <const float*>a.data
    cdef const float* b_ptr = <const float*>b.data
    cdef float dist
    with nogil:
        dist = compute_distance(a_ptr, b_ptr, dim, metric)
    return dist

from cython.parallel cimport prange

def batch_search_layer_native(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] queries,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] entry_points,
    int ef,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] vectors,
    cnp.ndarray[cnp.int32_t, ndim=2, mode="c"] edges,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] degrees,
    int metric,
    int num_threads = 0
):
    cdef int num_queries = queries.shape[0]
    cdef int num_entry_points = entry_points.shape[0]
    cdef int num_vectors = vectors.shape[0]
    cdef int dim = vectors.shape[1]
    cdef int max_neighbors = edges.shape[1]
    cdef const float* queries_ptr = <const float*>queries.data
    cdef const int* ep_ptr = <const int*>entry_points.data
    cdef const float* vecs_ptr = <const float*>vectors.data
    cdef const int* edges_ptr = <const int*>edges.data
    cdef const int* degrees_ptr = <const int*>degrees.data
    cdef vector[vector[pair[int, float]]] all_results
    all_results.resize(num_queries)

    cdef int q_idx
    with nogil:
        for q_idx in prange(num_queries, num_threads=num_threads):
            all_results[q_idx] = search_layer(
                queries_ptr + <size_t>q_idx * dim,
                ep_ptr,
                num_entry_points,
                ef,
                vecs_ptr,
                num_vectors,
                dim,
                edges_ptr,
                degrees_ptr,
                max_neighbors,
                metric
            )

    cdef list py_batch_results = []
    cdef size_t i, j
    for i in range(num_queries):
        row = []
        for j in range(all_results[i].size()):
            row.append((all_results[i][j].first, all_results[i][j].second))
        py_batch_results.append(row)
    return py_batch_results
