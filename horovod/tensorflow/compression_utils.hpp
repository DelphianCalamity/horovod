#ifndef compression_utils_hpp
#define compression_utils_hpp

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "./policies.hpp"

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

    static void logging_compressor(bloom::OrdinaryBloomFilter<uint32_t>& bloom, int N, int K, int output_concat_dim,
    const Tensor& initial_tensor, const Tensor& indices, const Tensor& values, std::vector<int>& selected_indices,
    int logfile_suffix, int logs_path_suffix, int64 step, std::string policy) {

         // Compute number of false positives
        int false_positives = bloom.Compute_False_Positives(N, indices);
std::cout << "DD" << std::endl;
        int policy_errors = Policies::get_policy_errors(K, indices, selected_indices);
        std::string suffix = std::to_string(logfile_suffix);
        std::string logs_suffix = std::to_string(logs_path_suffix);
        std::string str_step = std::to_string(step);

        std::string cmd = "mkdir -p logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/";
        int systemRet = system(cmd.c_str());
        if(systemRet == -1){
            perror("mkdir failed");
        }
        std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/compressor_logs_"
        + policy + "_" + suffix + ".txt";
        FILE* f = fopen(str.c_str(),"w");
        if (f==NULL) {
            perror ("Can't open file");
        }
        fprintf(f, "\nInitial Tensor: %s\n\n", initial_tensor.DebugString(N).c_str());
        fprintf(f, "Values: %s\n", values.DebugString(K).c_str());
        fprintf(f, "Indices: %s\n\n", indices.DebugString(K).c_str());

        int bloom_size = bloom.Get_numBytes();
        fprintf(f, "Bloom size: = %d\n", bloom_size);
        bloom.fprint(f);
        fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
        fprintf(f, "FalsePositives: %d\n", false_positives);
        fprintf(f, "Total: %d\n", N);
        fprintf(f, "\n\n########################################################################################\n\n");
        fclose(f);

        std::string str1 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/fpr_" + suffix + ".txt";
        f = fopen(str1.c_str(),"w");
        fprintf(f, "FalsePositives: %d  Total: %d\n", false_positives,  N);
        fclose(f);

        std::string str4 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/policy_errors_" + suffix + ".txt";
        f = fopen(str4.c_str(),"w");
        fprintf(f, "PolicyErrors: %d  Total: %d\n", policy_errors,  K);
        fclose(f);

        std::string str3 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/stats" + suffix + ".txt";
        f = fopen(str3.c_str(),"w");
        fprintf(f, "Initial_Size: %d  Final_Size: %d\n", N /*in bits*/,  bloom_size*8 /*in bits*/);
        fclose(f);

        std::string str2 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/hashes_" + suffix + ".txt";
        f = fopen(str2.c_str(),"w");
        for (int i=0; i<N; ++i) {
            fprintf(f, "i=%lu\n", i);
            for (uint8_t j=0; j<bloom.Get_numHashes(); j++){
                fprintf(f, "%lu, ", bloom.ComputeHash(i, j));
            }
            fprintf(f, "\n");
        }
        fclose(f);


    }

    static void logging_decompressor(bloom::OrdinaryBloomFilter<uint32_t>& bloom, int N, int K,
    int* values_vec, std::vector<int>& selected_indices, int logfile_suffix, int logs_path_suffix,
    int suffix,  int64 step, Tensor* decompressed_tensor, std::string policy) {

        std::string str_suffix = std::to_string(logfile_suffix);
        std::string logs_suffix = std::to_string(logs_path_suffix);
        std::string str_step = std::to_string(step);
        std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + str_suffix + "/decompressor_logs_"  +
        policy + "_" + str_suffix + "_" + std::to_string(suffix) + ".txt";
        FILE* f = fopen(str.c_str(),"w");
        if (f==NULL)
            perror ("Can't open file");
        fprintf(f, "decompressed size: %d\n\n", N);
        int bloom_size = bloom.Get_numBytes();
        fprintf(f, "Bloom size: = %d\n", bloom_size);
        bloom.fprint(f);
        fprintf(f, "\nIndices Chosen:");
        print_vector(selected_indices.data(), K, f);
        fprintf(f, "Values Vector:"); print_vector(values_vec, K, f);
        fprintf(f, "Decompressed_tensor: %s\n", decompressed_tensor->DebugString(N).c_str());
        fprintf(f, "########################################################################################\n\n");
        fclose (f);
    }


};

#endif
