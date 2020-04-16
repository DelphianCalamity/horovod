#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "../../third_party/bloomfilter/inc/OrdinaryBloomFilter.hpp"
#include "../../third_party/bloomfilter/inc/FnvHash.hpp"
#include "../../third_party/bloomfilter/inc/MurmurHash.hpp"
#include "./compression_utils.hpp"
#include <cmath>

#include <string>
#include <cstdlib>

using namespace tensorflow;

REGISTER_OP("StackedBloomCompressorConflictSets")
.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("hash_num: int")
.Attr("bloom_size: int")
.Attr("hash_num2: int")
.Attr("bloom_size2: int")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("verbosity: int")            // For debugging
.Input("values: T")
.Input("indices: int32")
.Input("initial_tensor: int32")    // For debugging
.Input("step: int64")              // For debugging
.Output("compressed_tensor: int8")
.Doc(R"doc()doc");


namespace std {
    template<>
    struct hash<bloom::HashParams<uint32_t>> {
    size_t operator()(bloom::HashParams<uint32_t> const &s) const {
    uint32_t out;
    bloom::MurmurHash3::murmur_hash3_x86_32((uint32_t*) &s.a, sizeof(s.a), s.b, (uint32_t*) &out);
    return out;
}
};
}

class StackedBloomCompressorConflictSetsOp : public OpKernel {

public:

    explicit StackedBloomCompressorConflictSetsOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("hash_num", &hash_num));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size", &bloom_size));
        OP_REQUIRES_OK(context, context->GetAttr("hash_num2", &hash_num2));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size2", &bloom_size2));
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
    }


    int find(const Tensor& indices, int x) {
        auto indices_flat = indices.flat<int>();
        for (int i=0; i<indices_flat.size(); ++i) {   // Dummy lookup
            if (indices_flat(i) == x)
                return 1;
        }
        return 0;
    }


    void Compute(OpKernelContext *context) override {

         // Retrieving Inputs
        const Tensor &values = context->input(0);  auto values_flat = values.flat<int>();
        const Tensor &indices = context->input(1);  auto indices_flat = indices.flat<int>();
        int values_size = values_flat.size();
        const Tensor &initial_tensor = context->input(2);
        auto initial_flat = initial_tensor.flat<int>();
        int N = initial_flat.size();
        int K = values_size;

        int output_concat_dim = values_size*sizeof(int) + bloom_size + bloom_size2;

/*****************************************************************************/
        // Building Bloom Filter
        bloom::OrdinaryBloomFilter<uint32_t> bloom(hash_num, bloom_size);

        // Build the first bloom on the indices
        for (int i=0; i<indices_flat.size(); ++i) {
            bloom.Insert(indices_flat(i));
        }

        bloom::OrdinaryBloomFilter<uint32_t> bloom2(hash_num2, bloom_size2);
        // Retrieve false positive items
        std::vector<int> false_positive_items;
        for (int i=0; i<N; ++i) {
            if (bloom.Query(i) && !find(indices, i)) {
                false_positive_items.push_back(i);
            }
        }
        // Build second bloom on the false positive items
        for (int i=0; i<false_positive_items.size(); ++i) {
            bloom2.Insert(false_positive_items[i]);
        }
/*****************************************************************************/

        // Create an output tensor
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        auto output_flat = output->template flat<int8>();
        int8* out_ptr = output_flat.data();
        const void* values_ptr = values_flat.data();

        memcpy(out_ptr, values_ptr, values_size*sizeof(int));
        std::vector<unsigned char> &bloom_vec = bloom.Get_bloom();
        std::copy(bloom_vec.begin(), bloom_vec.end(), out_ptr+values_size*sizeof(int));
        std::vector<unsigned char> &bloom_vec2 = bloom2.Get_bloom();
        std::copy(bloom_vec2.begin(), bloom_vec2.end(), out_ptr+values_size*sizeof(int)+bloom_size);

        // *********************** For Debugging ********************** //
        const Tensor &step_tensor = context->input(3);
        auto step = step_tensor.flat<int64>();

        if (verbosity != 0 && step(0) % verbosity == 0 ) {

            // Compute number of false positives
            int false_positives = 0;
            for (int i=0; i<N; ++i) {
                if (bloom.Query(i) && bloom2.Query(i) && !find(indices, i)) {
                    false_positives++;
                }
            }
    /*******************************************************************************************/
            int random, idx, left = K;
            std::vector<int> selected_indices;
            std::unordered_map<string, std::vector<int>> conflict_sets;
            // Iterating over the universe and collecting the conflict sets of bloom1
            for (int i=0; i<N; i++) {
                if (bloom.Query(i)) {  // If it is positive
//                    std::cout << "positive: " << i << std::endl;
                    string hash_string = bloom.Hash(i);
//                    std::cout << "hash_string: " << hash_string << std::endl;
                    conflict_sets[hash_string].push_back(i);
                }
            }

            for (auto& cset: conflict_sets) {
                if (cset.second.size() == 1) {
                    selected_indices.push_back(cset.second[0]);
                    left--;
                    cset.second.erase(cset.second.begin());
                }
            }
            for (auto& cset: conflict_sets) {
                for (int i=0; i < cset.second.size(); i++) {
                    if (!bloom2.Query(cset.second[i])) {  // If it is negative then add to selected items and remove from conflict sets
                        selected_indices.push_back(cset.second[i]);
                        left--;
                        cset.second.erase(cset.second.begin()+i);
                    }
                }
            }
            // Conflict sets are now pruned

            srand(step(0));
            while (left > 0) {                          // Don't stop until you have selected K positives
                for (auto& cset: conflict_sets) {       // Choose a positive out of every conflict set - remove it from set
                    if (left == 0)
                        break;
                    random = rand() % cset.second.size();    // choose randomly an element from the set
                    idx = cset.second[random];
                    cset.second.erase(cset.second.begin()+random);
                    selected_indices.push_back(idx);
                    left--;
                }
            }
            std::sort(selected_indices.begin(), selected_indices.end());

            // Selected-indices contains the indices that will be selected from the decompressor examine how many of those are false
            int policy_errors = 0;
            for (int i=0; i<K; i++) {
                int chosen_index = selected_indices[i];
                if (!find(indices, chosen_index)) {
                    policy_errors++;
                }
            }
    /*******************************************************************************************/

            std::string suffix = std::to_string(logfile_suffix);
            std::string logs_suffix = std::to_string(logs_path_suffix);
            std::string str_step = std::to_string(step(0));

            std::string cmd = "mkdir -p logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/";
            int systemRet = system(cmd.c_str());
            if(systemRet == -1){
                perror("mkdir failed");
            }
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/stacked_compressor_conflict_sets_logs_" + suffix + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            if (f==NULL) {
                perror ("Can't open file");
            }
            fprintf(f, "\nInitial Tensor: %s\n\n", initial_tensor.DebugString(initial_flat.size()).c_str());
            fprintf(f, "Values: %s\n", values.DebugString(values_flat.size()).c_str());
            fprintf(f, "Indices: %s\n\n", indices.DebugString(indices_flat.size()).c_str());
            fprintf(f, "Bloom size 1: = %d\n", bloom_size);
            bloom.fprint(f);
            fprintf(f, "Bloom size 2: = %d\n", bloom_size2);
            bloom2.fprint(f);
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            fprintf(f, "FalsePositives: %d\n", false_positives);
            fprintf(f, "Total: %d\n", initial_flat.size());
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);

            std::string str1 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/fpr_" + suffix + ".txt";
            f = fopen(str1.c_str(),"w");
            fprintf(f, "FalsePositives: %d  Total: %d\n", false_positives,  initial_flat.size());
            fclose(f);

            std::string str4 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/policy_errors_" + suffix + ".txt";
            f = fopen(str4.c_str(),"w");
            fprintf(f, "PolicyErrors: %d  Total: %d\n", policy_errors,  K);
            fclose(f);

            std::string str3 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/stats" + suffix + ".txt";
            f = fopen(str3.c_str(),"w");
            fprintf(f, "Initial_Size: %d  Final_Size: %d\n", initial_flat.size() /*in bits*/,  bloom_size*8 /*in bits*/);
            fclose(f);

            std::string str2 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/hashes_" + suffix + ".txt";
            f = fopen(str2.c_str(),"w");
            for (int i=0; i<initial_flat.size(); ++i) {
                fprintf(f, "i=%lu\n", i);
                for (uint8_t j=0; j<hash_num; j++){
                    fprintf(f, "%lu, ", bloom.ComputeHash(i, j));
                }
                fprintf(f, "\n");
            }
            fclose(f);

        }
        // *********************** For Debugging ********************** //

    }

private:
    int hash_num;
    int bloom_size;
    int hash_num2;
    int bloom_size2;
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int verbosity;          // For debugging
};


REGISTER_KERNEL_BUILDER(Name("StackedBloomCompressorConflictSets").Device(DEVICE_CPU), StackedBloomCompressorConflictSetsOp);
