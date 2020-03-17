
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

#include "../../third_party/FastPFor/headers/codecfactory.h"
#include "../../third_party/FastPFor/headers/deltautil.h"

#include <assert.h>
#include <string>
#include <cstdlib>

using namespace tensorflow;


REGISTER_OP("BitstreamCompressor")
//.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("verbosity: int")            // For debugging
//.Attr("code: string")
.Input("indices: int32")             // Indices
.Input("initial_tensor_size: int32")
.Input("step: int64")              // For debugging
.Output("bitcompressed_tensor: int8")
.Doc(R"doc( bitstream compression )doc");

REGISTER_OP("BitstreamDecompressor")
//.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("suffix: int")                    // For debugging
.Attr("verbosity: int")            // For debugging
//.Attr("code: string")
.Input("input: int8")
.Input("decompressed_size: int32")
.Input("step: int64")              // For debugging
.Output("decompressed_tensor: uint32")
.Doc(R"doc( bitstream decompression )doc");


class BitstreamCompressorOp : public OpKernel {

public:

    explicit BitstreamCompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
//        OP_REQUIRES_OK(context, context->GetAttr("code", &code));
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &indices_tensor = context->input(0);
        auto indices_tensor_flat = indices_tensor.flat<uint32_t>();
        size_t indices_tensor_size = indices_tensor_flat.size();

        size_t initial_tensor_size = context->input(1).flat<int32_t>().size();

        // Build the bitsream vector that is to be encoded
        std::vector<int8_t> bitstream(initial_tensor_size, 0);
        unsigned int bit_pos, byte_pos, byte, value;
        for (int i=0; i<indices_tensor_size; i++) {
                byte_pos = hash/8;
                bit_pos = hash%8;
                byte = bitstream[byte_pos];
                value = 1;
                value = value << bit_pos;
                value = value | byte;
                bitstream[byte_pos] = value;
        }


        // Binary Run Length Encoding
        // Let s ← 0.
        // While there are bits to encode:
        // Read the next n consecutive bits equal to s.
        // Write n.
        // s ← (s + 1)modulus2.




        // Create an output tensor
        int output_concat_dim = bitcompressed_output.size() ;
        printf("output_concat_dim %d\n", output_concat_dim);
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        auto output_flat = output->template flat<uint32>();
        uint32* out_ptr = output_flat.data();

        std::copy(bitcompressed_output.begin(), bitcompressed_output.end(), out_ptr);


        //////
        std::vector<uint32> init(input_tensor_size);
        const uint32_t *ptr = input_tensor_flat.data();
        memcpy(init.data(), ptr, input_tensor_size*sizeof(int));
        std::vector<uint32_t> decompressed_output(input_tensor_size);
        codec.decodeArray(bitcompressed_output.data(), output_concat_dim, decompressed_output.data(), input_tensor_size);
        decompressed_output.resize(input_tensor_size);

        assert(std::equal(init.begin(), init.end(), decompressed_output.begin()) == 1);
        /////



        // *********************** For Debugging ********************** //
        const Tensor &step_tensor = context->input(1);
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
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/bitcompressor_logs_" + suffix + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            fprintf(f, "input_tensor: %s\n", input_tensor.DebugString(input_tensor_flat.size()).c_str());
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            fprintf(f, "Bitcompressed_tensor: %s\n", output->DebugString(output_flat.size()).c_str());
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);

            std::string str1 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/stats" + suffix + ".txt";
            f = fopen(str1.c_str(),"w");
            fprintf(f, "Initial_Size: %d  Final_Size: %d\n", input_tensor_flat.size(),  output_concat_dim);
            fclose(f);
        }
        // *********************** For Debugging ********************** //

    }

private:
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int verbosity;          // For debugging
    string code;
};

class BitstreamDecompressorOp : public OpKernel {

public:

    explicit BitstreamDecompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("suffix", &suffix));                       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("code", &code));

    }

    void Compute(OpKernelContext *context) override {

        const Tensor &input_tensor = context->input(0);
        auto input_tensor_flat = input_tensor.flat<uint32_t>();
        const size_t input_tensor_size = input_tensor_flat.size();

        const Tensor &decompressed_size_tensor = context->input(1);
        auto decompressed_size_flat = decompressed_size_tensor.flat<int>();
        int decompressed_size = *decompressed_size_flat.data();

        IntegerCODEC &codec = *CODECFactory::getFromName(code);   // Pick a CODEC
        std::vector<uint32_t> decompressed_output(decompressed_size);
        size_t recoveredsize = decompressed_output.size();

        codec.decodeArray(input_tensor_flat.data(), input_tensor_size, decompressed_output.data(), recoveredsize);
        decompressed_output.resize(recoveredsize);

        // Create an output tensor
        int output_concat_dim = recoveredsize;
        printf("output_concat_dim %d\n", output_concat_dim);
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        auto output_flat = output->template flat<uint32>();
        uint32* out_ptr = output_flat.data();

        std::copy(decompressed_output.begin(), decompressed_output.end(), out_ptr);

        // *********************** For Debugging ********************** //
        const Tensor &step_tensor = context->input(2);
        auto step = step_tensor.flat<int64>();
        if (verbosity != 0 && step(0) % verbosity == 0 ) {
            std::string str_suffix = std::to_string(logfile_suffix);
            std::string logs_suffix = std::to_string(logs_path_suffix);
            std::string str_step = std::to_string(step(0));
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + str_suffix + "/bitdecompressor_logs_" + str_suffix + "_" + std::to_string(suffix) + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            fprintf(f, "input_tensor: %s\n", input_tensor.DebugString(input_tensor_flat.size()).c_str());
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            fprintf(f, "recoveredsize: = %d\n\n", recoveredsize);
            fprintf(f, "Bitdecompressed_tensor: %s\n", output->DebugString(output_flat.size()).c_str());
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
    string code;
};

REGISTER_KERNEL_BUILDER(Name("BitstreamCompressor").Device(DEVICE_CPU), BitstreamCompressorOp);

REGISTER_KERNEL_BUILDER(Name("BitstreamDecompressor").Device(DEVICE_CPU), BitstreamDecompressorOp);

