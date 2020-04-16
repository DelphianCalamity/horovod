#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "../../third_party/bloomfilter/inc/OrdinaryBloomFilter.hpp"
#include "../../third_party/bloomfilter/inc/FnvHash.hpp"
#include "../../third_party/bloomfilter/inc/MurmurHash.hpp"
#include "./policies.hpp"
#include "./compression_utils.hpp"

#include <cmath>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <functional>

using namespace tensorflow;

REGISTER_OP("BloomCompressor")
.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("stacked: bool")
.Attr("false_positives_aware: bool")
.Attr("policy: string")
.Attr("hash_num: int")
.Attr("bloom_size: int")
.Attr("second_hash_num: int")
.Attr("second_bloom_size: int")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("verbosity: int")            // For debugging
.Input("values: T")
.Input("indices: int32")
.Input("initial_tensor: int32")    // For debugging
.Input("step: int64")              // For debugging
.Output("compressed_tensor: int8")
.Doc(R"doc()doc");

REGISTER_OP("BloomDecompressor")
.Attr("stacked: bool")
.Attr("policy: string")
.Attr("mem_mode: int")
.Attr("hash_num: int")
.Attr("bloom_size: int")
.Attr("second_hash_num: int")
.Attr("second_bloom_size: int")
.Attr("logfile_suffix: int")            // For debugging
.Attr("logs_path_suffix: int")          // For debugging
.Attr("suffix: int")                    // For debugging
.Attr("verbosity: int")                 // For debugging
.Input("compressed_tensor: int8")
.Input("decompressed_size: int32")
.Input("step: int64")                   // For debugging
.Output("decompressed_tensor: int32")
.Doc(R"doc()doc");

class BloomCompressorOp : public OpKernel {

public:

    explicit BloomCompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("stacked", &stacked));
        OP_REQUIRES_OK(context, context->GetAttr("false_positives_aware", &false_positives_aware));
        OP_REQUIRES_OK(context, context->GetAttr("policy", &policy));
        OP_REQUIRES_OK(context, context->GetAttr("hash_num", &hash_num));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size", &bloom_size));
        OP_REQUIRES_OK(context, context->GetAttr("second_hash_num", &second_hash_num));
        OP_REQUIRES_OK(context, context->GetAttr("second_bloom_size", &second_bloom_size));
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
    }

    void Compute(OpKernelContext *context) override {

         // Retrieving Inputs
        const Tensor &values = context->input(0);  auto values_flat = values.flat<int>();
        const Tensor &indices = context->input(1);  auto indices_flat = indices.flat<int>();
        const Tensor &initial_tensor = context->input(2); auto initial_flat = initial_tensor.flat<int>();
        int64 step = context->input(3).flat<int64>()(0);

        int N = initial_flat.size();
        int K = values_flat.size();
        int output_concat_dim = K*sizeof(int) + bloom_size;

        // Create an output tensor
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->template flat<int8>();
        int8* out_ptr = output_flat.data();

        // Building Bloom Filter
        bloom::OrdinaryBloomFilter<uint32_t> bloom(hash_num, bloom_size);
        for (int i=0; i<K; ++i) {
            bloom.Insert(indices_flat(i));
        }
        // Copy the bloom filter in the output tensor
        std::copy(bloom.Get_bloom().begin(), bloom.Get_bloom().end(), out_ptr+K*sizeof(int));
bloom.print();
        bloom::OrdinaryBloomFilter<uint32_t>* second_bloom_ptr = NULL;
        if (stacked) {
            output_concat_dim += second_bloom_size;
            second_bloom_ptr = new bloom::OrdinaryBloomFilter<uint32_t>(second_hash_num, second_bloom_size);
            // Retrieve false positive items
            std::vector<int> false_positive_items;
            for (int i=0; i<N; ++i) {
                if (bloom.Query(i) && !Policies::find(indices, i)) {
                    false_positive_items.push_back(i);
                }
            }
            // Build second bloom on the false positive items
            for (int i=0; i<false_positive_items.size(); ++i) {
                second_bloom_ptr->Insert(false_positive_items[i]);
            }
            // Copy the second bloom filter in the output tensor
            std::copy(second_bloom_ptr->Get_bloom().begin(), second_bloom_ptr->Get_bloom().end(), out_ptr+K*sizeof(int)+bloom_size);
second_bloom_ptr->print();
        }

        std::vector<int> selected_indices;
        // Copy the values in the output tensor
        std::vector<int> new_values;
        if (false_positives_aware) {
            // Select Indices using a Policy
            Policies::select_indices(policy, N, K, step, bloom, second_bloom_ptr, selected_indices);
            for (int i=0; i<K; i++) {
                int chosen_index = selected_indices[i];
                new_values.push_back(initial_flat(chosen_index));
            }
            memcpy(out_ptr, new_values.data(), K*sizeof(int));
        } else {
            const void* values_ptr = values_flat.data();
            memcpy(out_ptr, values_ptr, K*sizeof(int));
        }

        // *********************** For Debugging ********************** //
        if (verbosity != 0 && step % verbosity == 0 ) {
            if (!false_positives_aware) {
                // Select Indices using a Policy
                Policies::select_indices(policy, N, K, step, bloom, second_bloom_ptr, selected_indices);
            }
            CompressionUtilities::logging_compressor(bloom, N, K, output_concat_dim, initial_tensor, indices, values,
                                            new_values, selected_indices, logfile_suffix, logs_path_suffix, step, policy);
        }
        // *********************** For Debugging ********************** //
        if (second_bloom_ptr != NULL) {
            free(second_bloom_ptr);
        }
    }

private:
    bool stacked;
    bool false_positives_aware;
    string policy;
    int hash_num;
    int bloom_size;
    int second_hash_num;
    int second_bloom_size;
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int verbosity;          // For debugging
};

class BloomDecompressorOp : public OpKernel {

public:

    explicit BloomDecompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("stacked", &stacked));
        OP_REQUIRES_OK(context, context->GetAttr("policy", &policy));
        OP_REQUIRES_OK(context, context->GetAttr("mem_mode", &mem_mode));
        OP_REQUIRES_OK(context, context->GetAttr("hash_num", &hash_num));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size", &bloom_size));
        OP_REQUIRES_OK(context, context->GetAttr("second_hash_num", &second_hash_num));
        OP_REQUIRES_OK(context, context->GetAttr("second_bloom_size", &second_bloom_size));
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("suffix", &suffix));                       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
    }

    void Compute(OpKernelContext *context) override {

        // Retrieving Inputs
        const Tensor &compressed_tensor = context->input(0);
        auto compressed_tensor_flat = compressed_tensor.flat<int8>();
        int N = *context->input(1).flat<int>().data();
        int K = (compressed_tensor_flat.size()-bloom_size-second_bloom_size)/sizeof(int);
        int64 step = context->input(2).flat<int64>()(0);

        // Reconstruct the bloom filter
        const int8 *ptr = compressed_tensor_flat.data();           // Note: int8 is 1 byte
        int values_bytes = K*sizeof(int);
        int *values_vec = (int*) malloc(values_bytes);
        memcpy(values_vec, ptr, values_bytes);
        ptr += values_bytes;
        bloom::OrdinaryBloomFilter<uint32_t> bloom(hash_num, bloom_size, ptr);


        bloom::OrdinaryBloomFilter<uint32_t>* second_bloom_ptr = NULL;
        if (stacked) {
            ptr += bloom_size;
            second_bloom_ptr = new bloom::OrdinaryBloomFilter<uint32_t>(second_hash_num, second_bloom_size, ptr);
        }

        // Create an output tensor
        TensorShape decompressed_tensor_shape;
        decompressed_tensor_shape.AddDim(N);
        Tensor *decompressed_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, decompressed_tensor_shape, &decompressed_tensor));
        auto decompressed_tensor_flat = decompressed_tensor->template flat<int>();
        memset(decompressed_tensor_flat.data(), 0, N*sizeof(int));

        // Select Indices using a Policy
        std::vector<int> selected_indices;
        Policies::select_indices(policy, N, K, step, bloom, second_bloom_ptr, selected_indices);

        // Map values to the selected indices
        for (int i=0; i<K; i++) {
            decompressed_tensor_flat(selected_indices[i]) = values_vec[i];
        }

        // *********************** For Debugging ********************** //
        if (verbosity != 0 && step % verbosity == 0 && mem_mode == 0) {
            CompressionUtilities::logging_decompressor(bloom, N, K, values_vec, selected_indices, logfile_suffix,
                                logs_path_suffix, suffix, step, decompressed_tensor, policy);
        }
        // *********************** For Debugging ********************** //
        free(values_vec);
        if (second_bloom_ptr != NULL) {
            free(second_bloom_ptr);
        }
    }

private:
    bool stacked;
    string policy;
    int mem_mode;
    int hash_num;
    int bloom_size;
    int second_hash_num;
    int second_bloom_size;
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int suffix;             // For debugging
    int verbosity;          // For debugging
};

REGISTER_KERNEL_BUILDER(Name("BloomCompressor").Device(DEVICE_CPU), BloomCompressorOp);
REGISTER_KERNEL_BUILDER(Name("BloomDecompressor").Device(DEVICE_CPU), BloomDecompressorOp);