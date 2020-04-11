#ifndef policies_hpp
#define policies_hpp

#include <iostream>

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op.h"

#include "../../third_party/bloomfilter/inc/OrdinaryBloomFilter.hpp"
#include "../../third_party/bloomfilter/inc/MurmurHash.hpp"
#include "../../third_party/bloomfilter/inc/FnvHash.hpp"

using namespace tensorflow;
#include <random>


class Policies {

public:

    explicit
    Policies(){}

    static int find(const Tensor& indices, int x) {
        auto indices_flat = indices.flat<int>();
        for (int i=0; i<indices_flat.size(); ++i) {   // Dummy lookup
            if (indices_flat(i) == x)
                return 1;
        }
        return 0;
    }

    static int get_policy_errors(const int K, const Tensor& indices, const std::vector<int>& selected_indices) {
        int policy_errors = 0;
        for (int i=0; i<K; i++) {
            int chosen_index = selected_indices[i];
            if (!find(indices, chosen_index)) {
                policy_errors++;
            }
        }
        return policy_errors;
    }

    static void build_conflict_sets(int N, bloom::OrdinaryBloomFilter<uint32_t>& bloom, std::map<int, std::vector<int>>& conflict_sets) {
        // Iterating over the universe and collecting the conflict sets
        uint8_t hash_num = bloom.Get_numHashes();
        for (size_t i=0; i<N; i++) {
            if (bloom.Query(i)) {  // If it is positive
                for (uint8_t j=0; j<hash_num; j++) {
                    int hash = bloom.Get_Hash(i,j);
                    std::vector<int>& cs = conflict_sets[hash];
                    if (std::find(cs.begin(), cs.end(), i) == cs.end()) {
                        conflict_sets[hash].push_back(i);
                    }
                }
            }
        }
    }

    static void transform_and_sort(std::map<int, std::vector<int>>& conflict_sets, std::vector<std::vector<int>>& conflict_sets_ordered) {

        typedef std::function<std::vector<int>(std::pair<int, std::vector<int>>)> Transformator;
        Transformator transformator = [](std::pair<int, std::vector<int>> i) {
            return i.second;
        };
        std::transform(conflict_sets.begin(), conflict_sets.end(), conflict_sets_ordered.begin(), transformator);
        std::sort(conflict_sets_ordered.begin(), conflict_sets_ordered.end(), [](const std::vector<int> l, const std::vector<int> r) {
            return l.size() < r.size();
        });
    }

    static void choose_indices_from_conflict_sets(int K, int seed, std::vector<std::vector<int>>& conflict_sets_ordered, std::vector<int>& selected_indices) {
//        srand(seed);
        std::default_random_engine generator;
        generator.seed(seed);

        int random, idx, left = K;
        while (left > 0) {                                      // Don't stop until you have selected K positives
            std::vector<int>& cset = conflict_sets_ordered[0];
            std::uniform_int_distribution<int> distribution(0, cset.size()-1);
            random = distribution(generator);
//            random = rand() % cset.size();    // choose randomly an element from the set
            idx = cset[random];
            selected_indices.push_back(idx);
            left--;
            // Search the item in all the other conflicts sets and erase it
            // Then move those conflicts sets at the end of the vector
            int size = conflict_sets_ordered.size();
            for (int i=0; i<size; i++) {
                std::vector<int>& cs = conflict_sets_ordered[i];
                auto it = std::find(cs.begin(), cs.end(), idx);
                if (it != cs.end()) {
                    cs.erase(it);
                    if (cs.size()) {
                        conflict_sets_ordered.push_back(cs);
                    }
                    conflict_sets_ordered.erase(conflict_sets_ordered.begin()+i);
                    i--;
                }
            }
        }
        std::sort(selected_indices.begin(), selected_indices.end());
    }

    static void conflict_sets_policy(int N, int K, int seed, bloom::OrdinaryBloomFilter<uint32_t>& bloom,
                                    std::vector<int>& selected_indices) {
        std::map<int, std::vector<int>> conflict_sets;
        build_conflict_sets(N, bloom, conflict_sets);

        // Sort the conflict sets by their size
        std::vector<std::vector<int>> conflict_sets_ordered;
        conflict_sets_ordered.resize(conflict_sets.size());
        transform_and_sort(conflict_sets, conflict_sets_ordered);

        // Collect selected indices
        choose_indices_from_conflict_sets(K, seed, conflict_sets_ordered, selected_indices);
     }

    static void leftmostK(int N, int K, bloom::OrdinaryBloomFilter<uint32_t>& bloom,
                                    std::vector<int>& selected_indices) {
        // Iterating over the universe and collecting the first K positives
        for (size_t i=0; i<N; i++) {
            if (bloom.Query(i)) {  // If it is positive
                selected_indices.push_back(i);
            }
        }
     }

     static void select_indices(std::string policy, int N, int K, int64 step,
                                bloom::OrdinaryBloomFilter<uint32_t>& bloom,
                                std::vector<int>& selected_indices) {
        if (policy == "conflict_sets") {
            conflict_sets_policy(N, K, step, bloom, selected_indices);
        } else if (policy == "leftmostK") {
            leftmostK(N, K, bloom, selected_indices);
        }
    }

};

#endif
