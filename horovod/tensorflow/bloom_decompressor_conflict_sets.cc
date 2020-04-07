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
#include<cstdlib>

using namespace tensorflow;

REGISTER_OP("BloomDecompressorConflictSets")
.Attr("hash_num: int")
.Attr("bloom_size: int")
.Attr("logfile_suffix: int")            // For debugging
.Attr("logs_path_suffix: int")          // For debugging
.Attr("suffix: int")                    // For debugging
.Attr("verbosity: int")                 // For debugging
.Input("compressed_tensor: int8")
.Input("decompressed_size: int32")
.Input("step: int64")                   // For debugging
.Input("k: int32")
.Output("decompressed_tensor: int32")
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


class BloomDecompressorConflictSetsOp : public OpKernel {

public:

    explicit BloomDecompressorConflictSetsOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("hash_num", &h));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size", &m));
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("suffix", &suffix));                       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
    }

    void Compute(OpKernelContext *context) override {

        // Retrieving Inputs
        const Tensor &compressed_tensor = context->input(0);
        const Tensor &decompressed_size_tensor = context->input(1);

        auto compressed_tensor_flat = compressed_tensor.flat<int8>();
        auto decompressed_size_flat = decompressed_size_tensor.flat<int>();

        const Tensor &step_tensor = context->input(2);
        auto step = step_tensor.flat<int64>();

        bloom_size = m;
        hash_num = h;

        int values_size = (compressed_tensor_flat.size()-bloom_size)/sizeof(int);
        int decompressed_size = *decompressed_size_flat.data();
        int N = decompressed_size;
        int K = values_size;

        // Reconstruct the bloom filter
        const int8 *ptr = compressed_tensor_flat.data();           // Note: int8 is 1 byte
        int values_bytes = values_size*sizeof(int);
        int *values_vec = (int*) malloc(values_bytes);
        memcpy(values_vec, ptr, values_bytes);
        ptr += values_bytes;
        bloom::OrdinaryBloomFilter<uint32_t> bloom_filter(hash_num, bloom_size, ptr);

        // Create an output tensor
        TensorShape decompressed_tensor_shape;
        decompressed_tensor_shape.AddDim(decompressed_size);
        Tensor *decompressed_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, decompressed_tensor_shape, &decompressed_tensor));
        auto decompressed_tensor_flat = decompressed_tensor->template flat<int>();
        memset(decompressed_tensor_flat.data(), 0, decompressed_size*sizeof(int));


    /*******************************************************************************************/
        // use a hashmap <hash_string, vector<int>>
        std::unordered_map<string, std::vector<int>> conflict_sets;

        // Iterating over the universe and collecting the conflict sets
        for (int i=0; i<N; i++) {
            if (bloom_filter.Query(i)) {  // If it is positive
                string hash_string = bloom_filter.Hash(i);
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

    /*******************************************************************************************/
        // Map values to the selected indices
        for (int i=0; i<K; i++) {
            decompressed_tensor_flat(selected_indices[i]) = values_vec[i];
        }

        // *********************** For Debugging ********************** //

        if (verbosity != 0 && step(0) % verbosity == 0 ) {
            std::string str_suffix = std::to_string(logfile_suffix);
            std::string logs_suffix = std::to_string(logs_path_suffix);
            std::string str_step = std::to_string(step(0));
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + str_suffix + "/decompressor_conflict_sets_logs_" + str_suffix + "_" + std::to_string(suffix) + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            if (f==NULL)
                perror ("Can't open file");
            fprintf(f, "decompressed size: %d\n\n", decompressed_size);
            fprintf(f, "Bloom size: = %d\n", bloom_size);
            bloom_filter.fprint(f);
            fprintf(f, "\nIndices Chosen:");
            CompressionUtilities::print_vector(selected_indices.data(), K, f);
            fprintf(f, "Values Vector:"); CompressionUtilities::print_vector(values_vec, values_size, f);
            fprintf(f, "Decompressed_tensor: %s\n", decompressed_tensor->DebugString(decompressed_tensor_flat.size()).c_str());
            fprintf(f, "########################################################################################\n\n");
            fclose (f);
        }
        // *********************** For Debugging ********************** //

        free(values_vec);
    }

private:
    int hash_num;
    int bloom_size;
    int m;
    int h;
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int suffix;             // For debugging
    int verbosity;          // For debugging
};


REGISTER_KERNEL_BUILDER(Name("BloomDecompressorConflictSets").Device(DEVICE_CPU), BloomDecompressorConflictSetsOp);