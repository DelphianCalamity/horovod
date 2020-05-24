#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "../compression_utils.hpp"

#include <assert.h>
#include <string>
#include <cstdlib>

using namespace tensorflow;


REGISTER_OP("RleCompressorV1Code8")
.Attr("logs_path: string")
.Attr("gradient_id: int")
.Attr("rank: int")
.Attr("verbosity_frequency: int")
.Attr("verbosity: int")
.Input("indices: int32")
.Input("initial_tensor_size: int32")
.Input("step: int64")
.Output("encoding: uint8")
.Doc(R"doc( Run Length Encoding )doc");

REGISTER_OP("RleDecompressorV1Code8")
.Attr("logs_path: string")
.Attr("gradient_id: int")
.Attr("rank: int")
.Attr("suffix: int")
.Attr("verbosity_frequency: int")
.Attr("verbosity: int")
.Input("encoding: uint8")
.Input("initial_indices_size: int32")
.Input("step: int64")
.Output("decompressed_indices: int32")
.Doc(R"doc( Run Length Decoding )doc");

class RleCompressorV1Code8_Op : public OpKernel {

public:

    explicit RleCompressorV1Code8_Op(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("logs_path", &logs_path));
        OP_REQUIRES_OK(context, context->GetAttr("gradient_id", &gradient_id));
        OP_REQUIRES_OK(context, context->GetAttr("rank", &rank));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity_frequency", &verbosity_frequency));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &indices_tensor = context->input(0);
        auto indices_tensor_flat = indices_tensor.flat<int32_t>();
        size_t indices_tensor_size = indices_tensor_flat.size();

        size_t initial_tensor_size = *context->input(1).flat<int32_t>().data();
        int tensor_size_bytes = initial_tensor_size/8;
        if (initial_tensor_size%8 != 0) {
            tensor_size_bytes++;
        }

        /////////////////////////////////////////////////////////////////
        // Build the bitstream vector that is to be encoded
        std::vector<uint8_t> bitstream(tensor_size_bytes, 0);
        int index;
        unsigned int bit_pos, byte_pos, byte, value;
        for (int i=0; i<indices_tensor_size; i++) {
                index = indices_tensor_flat(i);
                byte_pos = index/8;
                bit_pos = index%8;
                byte = bitstream[byte_pos];
                value = 1;
                value = value << bit_pos;
                value = value | byte;
                bitstream[byte_pos] = value;
        }

        /////////////////////////////////////////////////////////////////
        std::vector<int> lengths;
        int count=0, current=0;

        // Binary Run Length Encoding
        for (int i=0; i<initial_tensor_size; i++) {
            byte_pos = i/8;
            bit_pos = i%8;
            byte = bitstream[byte_pos];
            value=1;
            value = value << bit_pos;
            value = value | byte;
            if ((value^byte) != 0) {    // It's 0
                if (current == 0) {
                    count++;
                } else {
                    lengths.push_back(count);
                    count=1; current=0;
                }
            } else {                    // It's 1
                if (current == 1) {
                    count++;
                } else {
                    lengths.push_back(count);
                    count=1; current=1;
                }
            }
        }
        lengths.push_back(count);

        /////////////////////////////////////////////////////////////////
        // Make lengths representation smaller
        int len, blocks, code = 8;
        int m = 2<<(code-1);
        uint8_t rem;
        std::vector<uint8_t> encoded_lengths;

        for (int i=0; i<lengths.size(); i++) {
            len = lengths[i];
            blocks = len/(m-1);
            rem = len%(m-1);
            blocks++;
            for (int j=0; j<blocks-1; j++) {
                encoded_lengths.push_back(m-1);
            }
            encoded_lengths.push_back(rem);
        }
        /////////////////////////////////////////////////////////////////

        // Create an output tensor
        int output_concat_dim = encoded_lengths.size() ;
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->template flat<uint8>();
        uint8* out_ptr = output_flat.data();
        std::copy(encoded_lengths.begin(), encoded_lengths.end(), out_ptr);

        // *********************** For Debugging ********************** //
        const Tensor &step_tensor = context->input(2);
        int step = step_tensor.flat<int64>()(0);

        if (verbosity_frequency != 0 && step % verbosity_frequency == 0 ) {
            CompressionUtilities::logging_bitstream_compressor(indices_tensor, output_concat_dim, tensor_size_bytes,
                            bitstream, &lengths, output, initial_tensor_size, logs_path, gradient_id, step, rank, verbosity);
        }
        // *********************** For Debugging ********************** //
    }

private:
    string logs_path;
    int gradient_id;
    int rank;
    int verbosity_frequency;
    int verbosity;
};

class RleDecompressorV1Code8_Op : public OpKernel {

public:

    explicit RleDecompressorV1Code8_Op(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("logs_path", &logs_path));
        OP_REQUIRES_OK(context, context->GetAttr("gradient_id", &gradient_id));
        OP_REQUIRES_OK(context, context->GetAttr("rank", &rank));
        OP_REQUIRES_OK(context, context->GetAttr("suffix", &suffix));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity_frequency", &verbosity_frequency));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &encoding = context->input(0);
        auto encoding_flat = encoding.flat<uint8_t>();
        const size_t encoding_tensor_size = encoding_flat.size();

        int initial_indices_size = *context->input(1).flat<int>().data();

        // Create an output tensor
        int output_concat_dim = initial_indices_size;
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->template flat<int32>();

        /////////////////////////////////////////////////////////////////
        // Build the lengths
        uint8_t byte;
        int ones_blocks=0, len, code = 8;
        int m = 2<<(code-1);
        std::vector<int> lengths;
        ones_blocks = 0;

        for (int i=0; i<encoding_tensor_size; i++) {
            byte = encoding_flat(i);
            if (byte == m-1) {
                ones_blocks++;      // encoding can never end in all ones
            } else {
                len = ones_blocks*(m-1) + byte;
                lengths.push_back(len);
                ones_blocks = 0;
            }
        }
        /////////////////////////////////////////////////////////////////
        // Derive the initial indices using the lengths
        int length, current=0, offset=0, x=0;
        for (int i=0; i<lengths.size(); i++) {
            length = lengths[i];
            if (current == 0) {
                offset += length;
                current=1;
            } else {
                for (int j=0; j<length; j++) {
                    output_flat(x) = offset;
                    offset++;
                    x++;
                }
                current=0;
            }
        }

        // *********************** For Debugging ********************** //
        const Tensor &step_tensor = context->input(2);
        int step = step_tensor.flat<int64>()(0);
        if (verbosity_frequency != 0 && step % verbosity_frequency == 0 ) {
            CompressionUtilities::logging_bitstream_decompressor(encoding, output_concat_dim, &lengths,
            output, logs_path, gradient_id, step, rank, suffix, verbosity);
        }
        // *********************** For Debugging ********************** //
    }

private:
    string logs_path;
    int gradient_id;
    int rank;
    int suffix;
    int verbosity_frequency;
    int verbosity;
};

REGISTER_KERNEL_BUILDER(Name("RleCompressorV1Code8").Device(DEVICE_CPU), RleCompressorV1Code8_Op);

REGISTER_KERNEL_BUILDER(Name("RleDecompressorV1Code8").Device(DEVICE_CPU), RleDecompressorV1Code8_Op);

