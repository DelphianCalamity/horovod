
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "../../third_party/bloomfilter/inc/OrdinaryBloomFilter.hpp"
#include "../../third_party/bloomfilter/inc/FnvHash.hpp"
#include <string>

using namespace tensorflow;

REGISTER_OP("BloomDecompressor")
.Attr("T: {int32, int64, float16, float32, float64}")                // Todo: Should be set to bits
.Attr("hash_num: int")
.Attr("bloom_size: int")
.Attr("logfile_suffix: int")            // For debugging
.Attr("suffix: int")                    // For debugging
.Attr("verbosity: int")                 // For debugging
.Input("compressed_tensor: T")
.Input("decompressed_size: int32")
.Input("step: int64")                   // For debugging
.Output("decompressed_tensor: int32")
/*
//Todo: Fix the segfault error below to enable shape inference
// https://github.com/tensorflow/tensorflow/issues/31335
//  https://github.com/tensorflow/tensorflow/issues/30494

/// .SetShapeFn([](shape_inference::InferenceContext* c) {
//   return shape_inference::ConcatV2Shape(c);
//   return shape_inference::ConcatShape(c, c->num_inputs()-1);
// })
*/
.Doc(R"doc(
        Receives a compressed tensor which is a concatenation of values and bloom filter,
        splits it to those two tensors using the bloom_size info and uses both of them
         to re-construct the initial tensor. Also receives the decompressed tensor's size.
        Arguments
            compressed_tensor: concatenation of values and bloom filter
        Output
            decompressed_tensor: values decoded
    )doc");


namespace std {
    template<>
    struct hash<bloom::HashParams<uint32_t>> {
        size_t operator()(bloom::HashParams<uint32_t> const &s) const {
            bloom::FnvHash32 h;
            h.Update(s.b);      // casting uint8_t to int
            h.Update(s.a);
            return h.Digest();
        }
    };
}


void print_vector(int* vec, int size, FILE* f) {
    fprintf(f, "\n[");
    int i=0;
    for (i = 0; i < size-1; i++) {
        fprintf(f, "%d, ", (int) vec[i]);
    }
    fprintf(f, "%d]\n\n", (int) vec[i]);
}


class BloomDecompressorOp : public OpKernel {

public:

    explicit BloomDecompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("hash_num", &hash_num));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size", &bloom_size));
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("suffix", &suffix));                       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                       // For debugging
    }

    void Compute(OpKernelContext *context) override {

        // Retrieving Inputs
        const Tensor &compressed_tensor = context->input(0);
        const Tensor &decompressed_size_tensor = context->input(1);

        auto compressed_tensor_flat = compressed_tensor.flat<int>();   // Todo: Expect bits
        auto decompressed_size_flat = decompressed_size_tensor.flat<int>();

        int values_size = compressed_tensor_flat.size()-bloom_size;
        int decompressed_size = *decompressed_size_flat.data();

        // Reconstruct the bloom filter
        int *bloom_vec = (int*) malloc(bloom_size*sizeof(int));         // Todo: to bits
        memcpy(bloom_vec, compressed_tensor_flat.data()+values_size, bloom_size*sizeof(int));

        int *values_vec = (int*) malloc(values_size*sizeof(int));       // Todo: to bits
        memcpy(values_vec, compressed_tensor_flat.data(), values_size*sizeof(int));

        // std::copy_n(compressed_tensor_flat.data(), bloom_size, );

        bloom::OrdinaryBloomFilter<uint32_t> bloom_filter(hash_num, bloom_size, bloom_vec);

        TensorShape decompressed_tensor_shape;
        decompressed_tensor_shape.AddDim(decompressed_size);

        // Create an output tensor
        Tensor *decompressed_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, decompressed_tensor_shape, &decompressed_tensor));
        auto decompressed_tensor_flat = decompressed_tensor->template flat<int>();

        // Decode the compressed tensor
        int i,j;
        for (i=0,j=0; j<values_size; ++i) {
            if (bloom_filter.Query(i)) {
                decompressed_tensor_flat(i) = values_vec[j];
                j++;
            } else {
                decompressed_tensor_flat(i) = 0;
            }
        }
        for (; i<decompressed_size; ++i) {
            decompressed_tensor_flat(i) = 0;
        }

        // *********************** For Debugging ********************** //

        const Tensor &step_tensor = context->input(2);
        auto step = step_tensor.flat<int64>();
        if (verbosity != 0 && step(0) % verbosity == 0 ) {        // Log every 100 iterations
            std::string str_suffix = std::to_string(logfile_suffix);
            std::string str_step = std::to_string(step(0));
            std::string str = "logs/step_" + str_step + "/" + str_suffix + "/decompressor_logs_" + str_suffix + "_" + std::to_string(suffix) + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            if (f==NULL) {
                perror ("Can't open file");
            }
            fprintf(f, "compressed_tensor: %s\n", compressed_tensor.DebugString(compressed_tensor_flat.size()).c_str());
            fprintf(f, "decompressed size: %d\n\n", decompressed_size);
            fprintf(f, "Bloom size: = %d\n", bloom_size);
             fprintf(f, "Bloom Filter:"); print_vector(bloom_vec, bloom_size, f);
            fprintf(f, "Values Vector:"); print_vector(values_vec, values_size, f);
            fprintf(f, "Decompressed_tensor: %s\n", decompressed_tensor->DebugString(decompressed_tensor_flat.size()).c_str());
            fprintf(f, "########################################################################################\n\n");
            fclose (f);
        }
        // *********************** For Debugging ********************** //

        free(values_vec);
        free(bloom_vec);
    }

private:
    int hash_num;
    int bloom_size;
    int logfile_suffix; // For debugging
    int suffix;         // For debugging
    int verbosity;      // For debugging
};


REGISTER_KERNEL_BUILDER(Name("BloomDecompressor").Device(DEVICE_CPU), BloomDecompressorOp);

//ToDo: GPU implementation

// #if HOROVOD_GPU_ALLREDUCE
// REGISTER_KERNEL_BUILDER(Name("BloomDecompressor").Device(DEVICE_GPU),BloomDecompressorOp);
// #endif