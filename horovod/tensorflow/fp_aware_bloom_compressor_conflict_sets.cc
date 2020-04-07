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

#include <string>
#include <cstdlib>

using namespace tensorflow;

REGISTER_OP("FpAwareBloomCompressorConflictSets")
.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("hash_num: int")
.Attr("bloom_size: int")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("verbosity: int")            // For debugging
.Input("initial_tensor: T")
.Input("indices: int32")
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

class FpAwareBloomCompressorConflictSetsOp : public OpKernel {

public:

    explicit FpAwareBloomCompressorConflictSetsOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("hash_num", &hash_num));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size", &bloom_size));
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
        const Tensor &initial_tensor = context->input(0);  auto initial_tensor_flat = initial_tensor.flat<int>();
        const Tensor &indices = context->input(1);  auto indices_flat = indices.flat<int>();
        const Tensor &step_tensor = context->input(2);
        auto step = step_tensor.flat<int64>();

        int N = initial_tensor_flat.size();
        int K = indices_flat.size();
        int output_concat_dim =  K*sizeof(int) + bloom_size;

        // Building Bloom Filter
        bloom::OrdinaryBloomFilter<uint32_t> bloom(hash_num, bloom_size);
        for (int i=0; i<K; ++i) {
            bloom.Insert(indices_flat(i));
        }

        // Create an output tensor
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        auto output_flat = output->template flat<int8>();
        int8* out_ptr = output_flat.data();
        std::vector<unsigned char> &bloom_vec = bloom.Get_bloom();
        std::copy(bloom_vec.begin(), bloom_vec.end(), out_ptr+K*sizeof(int));


   /*******************************************************************************************/
    // Select Indices
        std::unordered_map<string, std::vector<int>> conflict_sets;

        // Iterating over the universe and collecting the conflict sets
        for (int i=0; i<N; i++) {
            if (bloom.Query(i)) {  // If it is positive
                string hash_string = bloom.Hash(i);
                conflict_sets[hash_string].push_back(i);
            }
        }

        srand(step(0));

        int random, idx, left = K;
        std::vector<int> selected_indices;
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

        // FP_aware compressor: for every selected index select to send its value
        std::vector<int> values;
        for (int i=0; i<K; i++) {
            int chosen_index = selected_indices[i];
            values.push_back(initial_tensor_flat(chosen_index));
        }
        memcpy(out_ptr, values.data(), K*sizeof(int));

        // *********************** For Debugging ********************** //
        if (verbosity != 0 && step(0) % verbosity == 0 ) {

            // Compute number of false positives
            int false_positives = 0;
            for (int i=0; i<N; ++i) {
                if (bloom.Query(i) && !find(indices, i)) {
                    false_positives++;
                }
            }

            std::string suffix = std::to_string(logfile_suffix);
            std::string logs_suffix = std::to_string(logs_path_suffix);
            std::string str_step = std::to_string(step(0));

            std::string cmd = "mkdir -p logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/";
            int systemRet = system(cmd.c_str());
            if(systemRet == -1){
                perror("mkdir failed");
            }
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/FP_aware_compressor_conflict_sets_logs_" + suffix + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            if (f==NULL) {
                perror ("Can't open file");
            }
            fprintf(f, "\nInitial Tensor: %s\n\n", initial_tensor.DebugString(N).c_str());
//            fprintf(f, "Values: %s\n", values.DebugString(values_flat.size()).c_str());
            fprintf(f, "Indices: %s\n\n", indices.DebugString(K).c_str());
            fprintf(f, "Bloom size: = %d\n", bloom_size);
            bloom.fprint(f);
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);

            fprintf(f, "Values:");
            CompressionUtilities::print_vector(values.data(), K, f);
            fprintf(f, "Policy errors: %d\n\n", policy_errors);
            fprintf(f, "\nIndices Chosen:");
            CompressionUtilities::print_vector(selected_indices.data(), K, f);

            fprintf(f, "FalsePositives: %d\n", false_positives);
            fprintf(f, "Total: %d\n", N);
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);

            std::string str1 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/fpr_" + suffix + ".txt";
            f = fopen(str1.c_str(),"w");
            fprintf(f, "FalsePositives: %d  Total: %d\n", false_positives,  N);
            fclose(f);

            std::string str3 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/stats" + suffix + ".txt";
            f = fopen(str3.c_str(),"w");
            fprintf(f, "Initial_Size: %d  Final_Size: %d\n", N /*in bits*/,  bloom_size*8 /*in bits*/);
            fclose(f);

            std::string str4 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/policy_errors_" + suffix + ".txt";
            f = fopen(str4.c_str(),"w");
            fprintf(f, "PolicyErrors: %d  Total: %d\n", policy_errors,  K);
            fclose(f);

            std::string str2 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/hashes_" + suffix + ".txt";
            f = fopen(str2.c_str(),"w");
            for (int i=0; i<N; ++i) {
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
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int verbosity;          // For debugging
};


REGISTER_KERNEL_BUILDER(Name("FpAwareBloomCompressorConflictSets").Device(DEVICE_CPU), FpAwareBloomCompressorConflictSetsOp);