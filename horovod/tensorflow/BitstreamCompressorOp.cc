
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

#include "headers/codecfactory.h"
#include "headers/deltautil.h"

#include <string>
#include <cstdlib>

using namespace tensorflow;

using namespace FastPForLib;

REGISTER_OP("BitstreamCompressor")
//.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("verbosity: int")            // For debugging
.Attr("code: string")
.Input("input: uint32")
.Input("step: int64")              // For debugging
.Output("bitcompressed_tensor: uint32")
.Doc(R"doc( bitstream compression )doc");

class BitstreamCompressorOp : public OpKernel {

public:

    explicit BitstreamCompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("code", &code));
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &input_tensor = context->input(0);
        auto input_tensor_flat = input_tensor.flat<uint32_t>();
        const size_t input_tensor_size = input_tensor_flat.size();

        IntegerCODEC &codec = *CODECFactory::getFromName(code);   // Pick a CODEC

        std::vector<uint32> bitcompressed_output(input_tensor_size + 1024);
        size_t bitcompressed_size = bitcompressed_output.size();
        codec.encodeArray(input_tensor_flat.data(), input_tensor_size, bitcompressed_output.data(), bitcompressed_size);
        // Shrink back the array:
        bitcompressed_output.resize(bitcompressed_size);
        bitcompressed_output.shrink_to_fit();

        // display compression rate:
        std::cout << std::setprecision(3);
        std::cout << "You are using " << 32.0 * static_cast<double>(bitcompressed_output.size()) /
                     static_cast<double>(input_tensor_flat.size()) << " bits per integer. " << std::endl;

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
        }
        // *********************** For Debugging ********************** //

    }

private:
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int verbosity;          // For debugging
    string code;
};


REGISTER_KERNEL_BUILDER(Name("BitstreamCompressor").Device(DEVICE_CPU), BitstreamCompressorOp);

