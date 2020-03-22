#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "../compression_utils.h"

#include <assert.h>
#include <string>
#include <cstdlib>

using namespace tensorflow;


REGISTER_OP("RleCompressorV0Code8")
//.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("verbosity: int")            // For debugging
.Input("indices: int32")             // Indices
.Input("initial_tensor_size: int32")
.Input("step: int64")              // For debugging
.Output("encoding: uint8")
.Doc(R"doc( Run Length Encoding )doc");

REGISTER_OP("RleDecompressorV0Code8")
//.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("suffix: int")               // For debugging
.Attr("verbosity: int")            // For debugging
.Input("encoding: uint8")
.Input("initial_indices_size: int32")
.Input("step: int64")              // For debugging
.Output("decompressed_indices: int32")
.Doc(R"doc( Run Length Decoding )doc");

class RleCompressorV0Code8_Op : public OpKernel {

public:

    explicit RleCompressorV0Code8_Op(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &indices_tensor = context->input(0);
        auto indices_tensor_flat = indices_tensor.flat<int32_t>();
        size_t indices_tensor_size = indices_tensor_flat.size();

        size_t initial_tensor_size = *context->input(1).flat<int32_t>().data();
        printf("size = %d\n", initial_tensor_size);
        int tensor_size_bytes = initial_tensor_size/8;
        if (initial_tensor_size%8 != 0) {
            tensor_size_bytes++;
        }
        printf("tensor bytes = %d\n", tensor_size_bytes);

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
        int zeros_count=0, ones_count=0;

        // Binary Run Length Encoding
        for (int i=0; i<initial_tensor_size; i++) {
            byte_pos = i/8;
            bit_pos = i%8;
            byte = bitstream[byte_pos];
            value=1;
            value = value << bit_pos;
            value = value | byte;

            if ((value^byte) != 0) {    // It's 0
                zeros_count++;
                if (ones_count > 0) {
                    while (ones_count > 1) {
                        ones_count--;
                        lengths.push_back(0);
                    }
                    ones_count=0;
                }
            } else {                    // It's 1
                ones_count++;
                if (zeros_count > 0) {
                    lengths.push_back(zeros_count);
                    zeros_count = 0;
                }
            }
        }
        while (ones_count > 1) {
            ones_count--;
            lengths.push_back(0);
        }

        /////////////////////////////////////////////////////////////////
        // Make lengths representation smaller
        int len, blocks, code = 8;
        int m = 2<<(code-1); printf("m=%d\n", m);
        uint8 rem;
        std::vector<uint8> encoded_lengths;

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
        printf("output_concat_dim %d\n", output_concat_dim);
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->template flat<uint8>();
        uint8* out_ptr = output_flat.data();
        std::copy(encoded_lengths.begin(), encoded_lengths.end(), out_ptr);


        // *********************** For Debugging ********************** //
        const Tensor &step_tensor = context->input(2);
        auto step = step_tensor.flat<int64>();

        if (verbosity != 0 && step(0) % verbosity == 0 ) {
            std::string suffix = std::to_string(logfile_suffix);
            std::string logs_suffix = std::to_string(logs_path_suffix);
            std::string str_step = std::to_string(step(0));

            std::string cmd = "mkdir -p logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/";
            int systemRet = system(cmd.c_str());
            if(systemRet == -1){
                perror("mkdir failed");
            }
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/RleCompressorV0Code8_logs_" + suffix + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            fprintf(f, "indices_tensor: %s\n", indices_tensor.DebugString(indices_tensor_flat.size()).c_str());
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            fprint(f, tensor_size_bytes, bitstream);
            fprintf(f, "Lengths:\n");
            print_vector(lengths.data(), lengths.size(), f);
            fprintf(f, "Encoded lengths: %s\n", output->DebugString(output_flat.size()).c_str());
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);

            std::string str1 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/stats" + suffix + ".txt";
            f = fopen(str1.c_str(),"w");
            fprintf(f, "Initial_Size: %d  Final_Size: %d\n", initial_tensor_size,  output_concat_dim*8 + 32); // in bits // worker sends the size of the encoded tensor
            fclose(f);
        }
        // *********************** For Debugging ********************** //
    }

private:
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int verbosity;          // For debugging
};

class RleDecompressorV0Code8_Op : public OpKernel {

public:

    explicit RleDecompressorV0Code8_Op(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("suffix", &suffix));                       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
//        OP_REQUIRES_OK(context, context->GetAttr("code", &code));

    }

    void Compute(OpKernelContext *context) override {

        const Tensor &encoding = context->input(0);
        auto encoding_flat = encoding.flat<uint8_t>();
        const size_t encoding_tensor_size = encoding_flat.size();

        int initial_indices_size = *context->input(1).flat<int>().data();

        // Create an output tensor
        int output_concat_dim = initial_indices_size;
        printf("output_concat_dim %d\n", output_concat_dim);
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->template flat<int32>();


        /////////////////////////////////////////////////////////////////
        // Build the lengths
        uint8 byte;
        int ones_blocks=0, len, code = 8;
        int m = 2<<(code-1); printf("must be 256,  m=%d\n", m);
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
        int offset=0, x=0;
        for (int i=0; i<lengths.size(); i++) {
            offset += lengths[i];
            output_flat(x) = offset;
            offset++;
            x++;
        }

        // *********************** For Debugging ********************** //
        const Tensor &step_tensor = context->input(2);
        auto step = step_tensor.flat<int64>();
        if (verbosity != 0 && step(0) % verbosity == 0 ) {
            std::string str_suffix = std::to_string(logfile_suffix);
            std::string logs_suffix = std::to_string(logs_path_suffix);
            std::string str_step = std::to_string(step(0));
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + str_suffix + "/RleDecompressorV0Code8_logs_" + str_suffix + "_" + std::to_string(suffix) + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            fprintf(f, "encoding_flat: %s\n", encoding.DebugString(encoding_flat.size()).c_str());
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            fprintf(f, "Lenghts:\n");
            print_vector(lengths.data(), lengths.size(), f);
            fprintf(f, "Indices: %s\n", output->DebugString(output_flat.size()).c_str());
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);
        }
        // *********************** For Debugging ********************** //

    }

private:
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int suffix;             // For debugging
    int verbosity;          // For debugging
};

REGISTER_KERNEL_BUILDER(Name("RleCompressorV0Code8").Device(DEVICE_CPU), RleCompressorV0Code8_Op);

REGISTER_KERNEL_BUILDER(Name("RleDecompressorV0Code8").Device(DEVICE_CPU), RleDecompressorV0Code8_Op);

