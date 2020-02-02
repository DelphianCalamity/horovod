
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

REGISTER_OP("BloomCompressor")
.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("hash_num: int")
.Attr("bloom_size: int")
.Attr("logfile_suffix: int")
.Input("values: T")
.Input("indices: int32")
.Output("compressed_tensor: T")

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
.Doc(R"doc(
        Receives 'values' and 'indices' as inputs and builds a bloom filter on the indices.
        Returns a tensor wich is the concatenation of the values and the bloom filter
        Arguments
            values: a tensor of values to concatenate with bloom filter
            indices: a tensor of indices for building the bloom filter
        Output
            compressed_tensor: a concatenation of the values and the bloom filter
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


class BloomCompressorOp : public OpKernel {

public:

    explicit BloomCompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("hash_num", &hash_num));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size", &bloom_size));
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));
    }

    void Compute(OpKernelContext *context) override {

        std::string str = "logs/compressor_logs_" + std::to_string(logfile_suffix) + ".txt";
        FILE* f = fopen(str.c_str(),"w");
        // Retrieving Inputs
        const Tensor &values = context->input(0);
        const Tensor &indices = context->input(1);

        auto values_flat = values.flat<int>();
        auto indices_flat = indices.flat<int>();

        fprintf(f, "Values: %s\n", values.DebugString(values_flat.size()).c_str());
        fprintf(f, "Indices: %s\n\n", indices.DebugString(indices_flat.size()).c_str());

        // Building Bloom Filter
        bloom::OrdinaryBloomFilter<uint32_t> bloom(hash_num, bloom_size);
        for (int i = 0; i < indices_flat.size(); ++i) {
            bloom.Insert(indices_flat(i));
        }

        const std::vector<bool> &bloom_vec = bloom.Get_bloom();

        int output_concat_dim = values_flat.size() + bloom_size;
        fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);

        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
//        fprintf(f, "%d\n", output_shape.dims());
//        fprintf(f, "%d\n", output_shape.dim_size(0));

        // Create an output tensor
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->template flat<int>();

        // Todo: Important!! Copy values and bloom in a more efficient way; \
        // use bytes datatype for output tensor and memcopy the integer values. \
        //  https://github.com/tensorflow/tensorflow/blob/dcc414587f50673271a31ab767909ec89c956324/tensorflow/core/framework/tensor_testutil.h#L57

        for (int i = 0; i < values_flat.size(); ++i) {
            output_flat(i) = values_flat(i);
        }
        for (int i = values_flat.size(), j = 0; j < bloom_size; ++i, ++j) {
            output_flat(i) = bloom_vec[j];
        }

        fprintf(f, "\n\n########################################################################################\n\n");
    }

private:
    int hash_num;
    int bloom_size;
    int logfile_suffix;
};


REGISTER_KERNEL_BUILDER(Name("BloomCompressor").Device(DEVICE_CPU), BloomCompressorOp);

//ToDo: GPU implementation

// #if HOROVOD_GPU_ALLREDUCE
// REGISTER_KERNEL_BUILDER(Name("BloomCompressor").Device(DEVICE_GPU),BloomCompressorOp);
// #endif

