
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
.Attr("logfile_suffix: int")
.Attr("suffix: int")
.Input("compressed_tensor: T")
.Input("decompressed_size: int32")
.Output("decompressed_tensor: int32")
/*
//Todo: Fix the segfault error below to enable shape inference
// https://github.com/tensorflow/tensorflow/issues/31335
//  https://github.com/tensorflow/tensorflow/issues/30494


/// .SetShapeFn([](shape_inference::InferenceContext* c) {
//   return shape_inference::ConcatV2Shape(c);
//   return shape_inference::ConcatShape(c, c->num_inputs()-1);
// })

//.SetShapeFn([](shape_inference::InferenceContext* c) {
//    c->set_output(0, c->input(0));
//    return Status::OK();
//})
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
    h.Update(&s.b, sizeof(uint8_t));
    void *buff = malloc(sizeof(uint32_t));
    memcpy(buff, &s.a, sizeof(uint32_t));
    h.Update((const uint8_t *) buff, sizeof(uint32_t));
    free(buff);
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
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));
        OP_REQUIRES_OK(context, context->GetAttr("suffix", &suffix));
    }

    void Compute(OpKernelContext *context) override {
        std::string str_suffix = std::to_string(logfile_suffix);
        std::string str = "logs/" + str_suffix + "/decompressor_logs_" + str_suffix + "_" + std::to_string(suffix) + ".txt";

        FILE* f = fopen(str.c_str(),"w");
        // Retrieving Inputs
        const Tensor &compressed_tensor = context->input(0);
        auto compressed_tensor_flat = compressed_tensor.flat<int>();   // Todo: Expect bits
        const Tensor &decompressed_size = context->input(1);
        auto decompressed_size_flat = decompressed_size.flat<int>();

        int values_size = compressed_tensor_flat.size()-bloom_size;

        fprintf(f, "compressed_tensor: %s\n", compressed_tensor.DebugString(compressed_tensor_flat.size()).c_str());
        fprintf(f, "decompressed size: %s\n\n\n", decompressed_size.DebugString(decompressed_size_flat.size()).c_str());

        // Reconstruct the bloom filter
        int *bloom_vec = (int*) malloc(bloom_size*sizeof(int));         // Todo: to bits
        memcpy(bloom_vec, compressed_tensor_flat.data()+values_size, bloom_size*sizeof(int));

        int *values_vec = (int*) malloc(values_size*sizeof(int));       // Todo: to bits
        memcpy(values_vec, compressed_tensor_flat.data(), values_size*sizeof(int));
//        std::copy_n(compressed_tensor_flat.data(), bloom_size, );
        fprintf(f, "Bloom size: = %d\n", bloom_size);
        fprintf(f, "Bloom Filter:"); print_vector(bloom_vec, bloom_size, f);
        fprintf(f, "Values Vector:"); print_vector(values_vec, values_size, f);

        bloom::OrdinaryBloomFilter<uint32_t> bloom_filter(hash_num, bloom_size, bloom_vec);
        free(bloom_vec);

        TensorShape decompressed_tensor_shape;
        decompressed_tensor_shape.AddDim(*decompressed_size_flat.data());

        // Create an output tensor
        Tensor *decompressed_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, decompressed_tensor_shape, &decompressed_tensor));
        auto decompressed_tensor_flat = decompressed_tensor->template flat<int>();

        // Decode the compressed tensor
        for (int i=0,j=0; j<values_size; ++i) {
            if (bloom_filter.Query(i)) {
                decompressed_tensor_flat(i) = values_vec[j];
                j++;
            } else {
                decompressed_tensor_flat(i) = 0;
            }
        }
        free(values_vec);

        fprintf(f, "\n\n########################################################################################\n\n");

    }

private:
    int hash_num;
    int bloom_size;
    int logfile_suffix;
    int suffix;
};


REGISTER_KERNEL_BUILDER(Name("BloomDecompressor").Device(DEVICE_CPU), BloomDecompressorOp);

//ToDo: GPU implementation

// #if HOROVOD_GPU_ALLREDUCE
// REGISTER_KERNEL_BUILDER(Name("BloomDecompressor").Device(DEVICE_GPU),BloomDecompressorOp);
// #endif