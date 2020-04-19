#ifndef compression_utils_hpp
#define compression_utils_hpp

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op.h"

#include "../../third_party/bloomfilter/inc/OrdinaryBloomFilter.hpp"
#include "../../third_party/bloomfilter/inc/MurmurHash.hpp"
#include "../../third_party/bloomfilter/inc/FnvHash.hpp"
#include "policies.hpp"

#include <iostream>

class CompressionUtilities {

public:

    explicit
    CompressionUtilities(){}

    static void fprint(FILE* f, int size, std::vector<uint8_t> ptr) {
        unsigned int bit_pos, byte_pos, value, byte;
        fprintf(f, "Bitstream Array: \n [ ");

        for (byte_pos=0; byte_pos<size; byte_pos++) {
            for (bit_pos=0; bit_pos<8; bit_pos++) {
                byte = ptr[byte_pos];
                value = 1;
                value = value << bit_pos;
                value = value | byte;
                if ((value^byte) != 0)
                    fprintf(f, "0 ");
                else fprintf(f, "1 ");
            }
        }
        fprintf(f, "]\n\n");
    }

    static void fprint(FILE* f, int size, std::vector<int8_t> ptr) {
        unsigned int bit_pos, byte_pos, value, byte;
        fprintf(f, "Bitstream Array: \n [ ");

        for (byte_pos=0; byte_pos<size; byte_pos++) {
            for (bit_pos=0; bit_pos<8; bit_pos++) {
                byte = ptr[byte_pos];
                value = 1;
                value = value << bit_pos;
                value = value | byte;
                if ((value^byte) != 0)
                    fprintf(f, "0 ");
                else fprintf(f, "1 ");
            }
        }
        fprintf(f, "]\n\n");
    }

    static void print_vector(int* vec, int size, FILE* f) {
        fprintf(f, "\n[");
        int i=0;
        for (i = 0; i < size-1; i++) {
            fprintf(f, "%d, ", (int) vec[i]);
        }
        fprintf(f, "%d]\n\n", (int) vec[i]);
    }

    static void print_vector(int* vec, int size) {
        printf("\n[");
        int i=0;
        for (i = 0; i < size-1; i++) {
            printf("%d, ", (int) vec[i]);
        }
        printf("%d]\n\n", (int) vec[i]);
    }

    static void print_2d_vector(std::vector<std::vector<int>>& vec, FILE* f) {
        for (auto& it: vec) {
             fprintf(f, "{");
             for (auto& itt : it)
                fprintf(f, "%d, ", itt);
             fprintf(f, "}\n");
        }
    }

    static void print_map(std::map<int, std::vector<int>>& map, FILE* f) {
        for (auto& it: map) {
            fprintf(f, "Key: %d, Values: ", it.first);
             for (auto& itt : it.second)
                fprintf(f, "%d, ", itt);
             fprintf(f, "\n");
        }
    }

    static void logging_compressor(bloom::OrdinaryBloomFilter<uint32_t>& bloom, int N, int K, int output_concat_dim,
    const Tensor& initial_tensor, const Tensor& indices, const Tensor& values, std::vector<int>& new_values, std::vector<int>& selected_indices,
    int logfile_suffix, int logs_path_suffix, int64 step, std::string policy, int rank, int verbosity) {

        FILE* f;
        std::string str;
        int false_positives = bloom.Compute_False_Positives(N, indices);
        int policy_errors = Policies::get_policy_errors(K, indices, selected_indices);
        std::string suffix = std::to_string(logfile_suffix);
        std::string logs_suffix = std::to_string(logs_path_suffix);
        std::string str_step = std::to_string(step);
        std::string str_rank = std::to_string(rank);

        std::string path = "logs" + logs_suffix + "/" + str_rank + "/step_" + str_step + "/" + suffix + "/";
        std::string cmd = "mkdir -p " + path;

        if (verbosity > 1) {
            str = path + "compressor_logs_" + policy + ".txt";
            f = fopen(str.c_str(),"w");
            fprintf(f, "\nInitial Tensor: %s\n\n", initial_tensor.DebugString(N).c_str());
            fprintf(f, "Values: %s\n", values.DebugString(K).c_str());
            fprintf(f, "\nIndices: %s\n\n", indices.DebugString(K).c_str());
            fprintf(f, "Step: = %d\n\n", step);
/*
        if (policy == "conflict_sets") {
            std::map<int, std::vector<int>> conflict_sets;
            Policies::build_conflict_sets(N, bloom, conflict_sets);
            fprintf(f, "\nConflict Sets Map:\n");
            print_map(conflict_sets, f);
            std::vector<std::vector<int>> conflict_sets_ordered;
            conflict_sets_ordered.resize(conflict_sets.size());
            Policies::transform_and_sort(conflict_sets, conflict_sets_ordered);
            fprintf(f, "\nConflict Sets Ordered:\n");
            print_2d_vector(conflict_sets_ordered, f);
            std::vector<int> tmp;
//            Policies::choose_indices_from_conflict_sets(K, step, conflict_sets_ordered, tmp);
//std::default_random_engine generator;
//generator.seed(step);
//int random, idx, left = K;
//while (left > 0) {                                      // Don't stop until you have selected K positives
//    std::vector<int>& cset = conflict_sets_ordered[0];
////    random = rand() % cset.size();    // choose randomly an element from the set
//    std::uniform_real_distribution<float> distribution(0, cset.size()-1);
//    random = distribution(generator);
//    fprintf(f, "Random: %d\n", random);
//
//    idx = cset[random];
//    tmp.push_back(idx);
//    left--;
//    // Search the item in all the other conflicts sets and erase it
//    // Then move those conflicts sets at the end of the vector
//    int size = conflict_sets_ordered.size();
//    for (int i=0; i<size; i++) {
//        std::vector<int>& cs = conflict_sets_ordered[i];
//        auto it = std::find(cs.begin(), cs.end(), idx);
//        if (it != cs.end()) {
//            cs.erase(it);
//            if (cs.size()) {
//                conflict_sets_ordered.push_back(cs);
//            }
//            conflict_sets_ordered.erase(conflict_sets_ordered.begin()+i);
//            i--;
//        }
//    }
//}
//            fprintf(f, "\nConflict Sets Ordered After Selection:\n");
            print_2d_vector(conflict_sets_ordered, f);
        }
        print_vector(selected_indices.data(), K, f);
*/
            int bloom_size = bloom.Get_numBytes();
            fprintf(f, "Bloom size: = %d\n", bloom_size);
            bloom.fprint(f);
            fprintf(f, "\nIndices Chosen:");
            print_vector(selected_indices.data(), K, f);
            if (new_values.size() > 0) {
                fprintf(f, "\nValues-Sent:");
                print_vector(new_values.data(), K, f);
            }
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            fprintf(f, "FalsePositives: %d\n", false_positives);
            fprintf(f, "Total: %d\n", N);
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);
        }

        str = path + "fpr.txt";
        f = fopen(str.c_str(),"w");
        fprintf(f, "FalsePositives: %d  Total: %d\n", false_positives,  N);
        fclose(f);

        str = path + "policy_errors.txt";
        f = fopen(str.c_str(),"w");
        fprintf(f, "PolicyErrors: %d  Total: %d\n", policy_errors,  K);
        fclose(f);

        str = path + "stats.txt";
        f = fopen(str.c_str(),"w");
        fprintf(f, "Initial_Size: %d  Final_Size: %d\n", N /*in bits*/,  bloom_size*8 /*in bits*/);
        fclose(f);
    }

    static void logging_decompressor(bloom::OrdinaryBloomFilter<uint32_t>& bloom, int N, int K,
    int* values_vec, std::vector<int>& selected_indices, int logfile_suffix, int logs_path_suffix,
    int suffix, int64 step, Tensor* decompressed_tensor, std::string policy, , int rank, int verbosity) {

        FILE* f;
        std::string str_suffix = std::to_string(logfile_suffix);
        std::string logs_suffix = std::to_string(logs_path_suffix);
        std::string str_step = std::to_string(step);
        std::string str_rank = std::to_string(rank);

        std::string path = "logs" + logs_suffix + "/" + str_rank + "/step_" + str_step + "/" + str_suffix;
        std::string str = path + "/decompressor_logs_" + policy + "_" + std::to_string(suffix) + ".txt";
        f = fopen(str.c_str(),"w");
        fprintf(f, "decompressed size: %d\n\n", N);
        fprintf(f, "Step: = %d\n\n", step);

        if (verbosity > 1) {
            int bloom_size = bloom.Get_numBytes();
            fprintf(f, "Bloom size: = %d\n", bloom_size);
            bloom.fprint(f);
            fprintf(f, "\nIndices Chosen:");
            print_vector(selected_indices.data(), K, f);
            fprintf(f, "Values Received:"); print_vector(values_vec, K, f);
            fprintf(f, "Decompressed_tensor: %s\n", decompressed_tensor->DebugString(N).c_str());
            fprintf(f, "########################################################################################\n\n");
        }
        fclose (f);
    }

};

#endif
