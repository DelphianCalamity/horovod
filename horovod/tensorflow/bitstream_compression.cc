
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

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
.Output("bitcompressed_tensor: int32")
.Doc(R"doc( bitstream compression )doc");

REGISTER_OP("BitstreamDecompressor")
//.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("suffix: int")               // For debugging
.Attr("verbosity: int")            // For debugging
//.Attr("code: string")
.Input("encoding: int32")
.Input("initial_indices_size: int32")
.Input("step: int64")              // For debugging
.Output("decompressed_indices: int32")
.Doc(R"doc( bitstream decompression )doc");


void fprint(FILE* f, int size, std::vector<int8_t> ptr) {
    unsigned int bit_pos, byte_pos, value, byte;
    fprintf(f, "Bitstream Array: \n [ ");

    for (byte_pos=0; byte_pos<size; byte_pos++) {
        for (bit_pos=0; bit_pos<8; bit_pos++) {
            byte = ptr[byte_pos];
            value = 1;
            value = value << bit_pos;
            value = value | byte;
            if ((value^byte) != 0)
                fprintf(f, "0 ");
            else fprintf(f, "1 ");
        }
    }
    fprintf(f, "]\n\n");

}

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
        auto indices_tensor_flat = indices_tensor.flat<int32_t>();
        size_t indices_tensor_size = indices_tensor_flat.size();

        size_t initial_tensor_size = *context->input(1).flat<int32_t>().data();
        printf("size = %d\n", initial_tensor_size);
        int tensor_size_bytes = initial_tensor_size/8;
        if (initial_tensor_size%8 != 0) {
            tensor_size_bytes++;
        }
        printf("tensor bytes = %d\n", tensor_size_bytes);
        // Build the bitstream vector that is to be encoded
        std::vector<int8_t> bitstream(tensor_size_bytes, 0);
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

        std::vector<int> encode;
        int count=0, current=0;

        // Binary Run Length Encoding
        for (int i=0; i<initial_tensor_size; i++) {
        printf("i:%d\n", i);
            byte_pos = i/8;
            bit_pos = i%8;
            byte = bitstream[byte_pos];
            value=1;
            value = value << bit_pos;
            value = value | byte;
            if ((value^byte) != 0) {    // It's 0
                if (current == 0) {
                    count++;
                    printf("0_cur_0 count=%d\n", count);

                } else {
                    encode.push_back(count);
                    count=1; current=0;
                    printf("0_cur_1 count=%d\n", count);
                }
            } else {                    // It's 1
                if (current == 1) {
                    count++;
                    printf("1_cur_1 count=%d\n", count);

                } else {
                    encode.push_back(count);
                    count=1; current=1;
                    printf("1_cur_0 count=%d\n", count);
                }
            }
        }
        encode.push_back(count);

        // Create an output tensor
        int output_concat_dim = encode.size() ;
        printf("output_concat_dim %d\n", output_concat_dim);
        TensorShape output_shape;
        output_shape.AddDim(output_concat_dim);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        auto output_flat = output->template flat<int32>();
        int32* out_ptr = output_flat.data();

        std::copy(encode.begin(), encode.end(), out_ptr);


//        //////
//        std::vector<uint32> init(input_tensor_size);
//        const uint32_t *ptr = input_tensor_flat.data();
//        memcpy(init.data(), ptr, input_tensor_size*sizeof(int));
//        std::vector<uint32_t> decompressed_output(input_tensor_size);
//        codec.decodeArray(bitcompressed_output.data(), output_concat_dim, decompressed_output.data(), input_tensor_size);
//        decompressed_output.resize(input_tensor_size);
//
//        assert(std::equal(init.begin(), init.end(), decompressed_output.begin()) == 1);
//        /////



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
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/bitcompressor_logs_" + suffix + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            fprintf(f, "indices_tensor: %s\n", indices_tensor.DebugString(indices_tensor_flat.size()).c_str());
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            fprint(f, tensor_size_bytes, bitstream);
            fprintf(f, "Output: %s\n", output->DebugString(output_flat.size()).c_str());
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);

//            std::string str1 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/stats" + suffix + ".txt";
//            f = fopen(str1.c_str(),"w");
//            fprintf(f, "Initial_Size: %d  Final_Size: %d\n", input_tensor_flat.size(),  output_concat_dim);
//            fclose(f);
        }
        // *********************** For Debugging ********************** //

    }

private:
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int verbosity;          // For debugging
//    string code;
};

class BitstreamDecompressorOp : public OpKernel {

public:

    explicit BitstreamDecompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("suffix", &suffix));                       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
//        OP_REQUIRES_OK(context, context->GetAttr("code", &code));

    }

    void Compute(OpKernelContext *context) override {

        const Tensor &encoding = context->input(0);
        auto encoding_flat = encoding.flat<int32_t>();
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

        int length, current=0, offset=0, x=0;
        for (int i=0; i<encoding_flat.size(); i++) {
            length = encoding_flat(i);

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
        auto step = step_tensor.flat<int64>();
        if (verbosity != 0 && step(0) % verbosity == 0 ) {
            std::string str_suffix = std::to_string(logfile_suffix);
            std::string logs_suffix = std::to_string(logs_path_suffix);
            std::string str_step = std::to_string(step(0));
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + str_suffix + "/bitdecompressor_logs_" + str_suffix + "_" + std::to_string(suffix) + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            fprintf(f, "encoding_flat: %s\n", encoding.DebugString(encoding_flat.size()).c_str());
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            fprintf(f, "Bitdecompressed_indices: %s\n", output->DebugString(output_flat.size()).c_str());
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
//    string code;
};

REGISTER_KERNEL_BUILDER(Name("BitstreamCompressor").Device(DEVICE_CPU), BitstreamCompressorOp);

REGISTER_KERNEL_BUILDER(Name("BitstreamDecompressor").Device(DEVICE_CPU), BitstreamDecompressorOp);

