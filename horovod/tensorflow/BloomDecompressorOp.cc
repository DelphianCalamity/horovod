
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "bloomfilter/inc/OrdinaryBloomFilter.hpp"
#include "bloomfilter/inc/FnvHash.hpp"

using namespace tensorflow;

// Todo: pass bloom size parameter as node argument
// Todo: need to have the initial size of the tensor.. how?

REGISTER_OP("BloomDecompressor")
.Attr("T: {int32}")                 // Todo: Should be set to bytes
.Input("compressed_tensor: T")
.Input("decompressed_size: int32")
.Output("decompressed_tensor: T")
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
         to re-construct the initial tensor.
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


class BloomDecompressorOp : public OpKernel {

public:

    explicit BloomDecompressorOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {

        // Retrieving Inputs
        const Tensor &compressed_tensor = context->input(0);
        auto compressed_tensor_flat = compressed_tensor.flat<int>();   // Todo: Expect bytes
        const Tensor &decompressed_size = context->input(1);
        auto decompressed_size_flat = decompressed_size.flat<int>();

        printf("\n");
        printf("compressed_tensor dims: %d\n", compressed_tensor.shape().dims());

        printf("\n\n");
        printf("compressed_tensor: %s\n", compressed_tensor.DebugString(compressed_tensor_flat.size()).c_str());
        printf("decompressed size: %s\n", decompressed_size.DebugString(decompressed_size_flat.size()).c_str());
        printf("\n\n");

        // Todo: Important: pass those as node arguments - not hardcoded
        // Reconstruct the bloom filter
        int hash_num = 2;
        uint16_t bloom_size = 10;
        printf("Bloom_Size: = %d\n\n", bloom_size);

        int *bloom_vec = (int*) malloc(bloom_size*sizeof(int));     // Todo: to bytes
        // move the bloom filter from compressed_tensor to bloom
//        std::copy_n(compressed_tensor_flat.data(), bloom_size, );
        memcpy(bloom_vec, compressed_tensor_flat.data()+compressed_tensor_flat.size()-bloom_size, bloom_size*sizeof(int));

        for (int i = 0; i < bloom_size; i++) {
            printf("B: %d\n", (int) bloom_vec[i]);
        }

        bloom::OrdinaryBloomFilter<uint32_t> bloom_filter(hash_num, bloom_size, bloom_vec);

        TensorShape decompressed_tensor_shape;
        decompressed_tensor_shape.AddDim(*decompressed_size_flat.data());
        printf("%d\n", decompressed_tensor_shape.dims());
        printf("%d\n", decompressed_tensor_shape.dim_size(0));

        printf("tensor_initial_size: = %d\n\n", *decompressed_size_flat.data());

        // Create an output tensor
        Tensor *decompressed_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, decompressed_tensor_shape, &decompressed_tensor));
        auto decompressed_tensor_flat = decompressed_tensor->template flat<int>();


        // Reconstruct Initial Tensor
        // write the values directly to the output tensor
        // while ....
        //    if (bloom.Query(indices_flat(0))) {
        //      std::cout << "Error: Query for first inserted element was false." << std::endl;
        //    }
    }
};


REGISTER_KERNEL_BUILDER(Name("BloomDecompressor").Device(DEVICE_CPU), BloomDecompressorOp);

//ToDo: GPU implementation

// #if HOROVOD_GPU_ALLREDUCE
// REGISTER_KERNEL_BUILDER(Name("BloomDecompressor").Device(DEVICE_GPU),BloomDecompressorOp);
// #endif
