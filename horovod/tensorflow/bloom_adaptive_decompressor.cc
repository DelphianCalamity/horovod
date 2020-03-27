#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "../../third_party/bloomfilter/inc/OrdinaryBloomFilter.hpp"
#include "../../third_party/bloomfilter/inc/FnvHash.hpp"
#include "../../third_party/bloomfilter/inc/MurmurHash.hpp"
#include "./compression_utils.hpp"

#include <string>
#include <cmath>


using namespace tensorflow;

REGISTER_OP("BloomAdaptiveDecompressor")
.Attr("bloom_size: int")
.Attr("logfile_suffix: int")            // For debugging
.Attr("logs_path_suffix: int")          // For debugging
.Attr("suffix: int")                    // For debugging
.Attr("verbosity: int")                 // For debugging
.Attr("partitioning: float")
.Input("compressed_tensor: int8")
.Input("decompressed_size: int32")
.Input("step: int64")                   // For debugging
.Output("decompressed_tensor: int32")
.Doc(R"doc()doc");


namespace std {
    template<>
    struct hash<bloom::HashParams<uint32_t>> {
    size_t operator()(bloom::HashParams<uint32_t> const &s) const {
    uint32_t out;
    bloom::MurmurHash3::murmur_hash3_x86_32((uint32_t*) &s.a, sizeof(s.a), s.b, (uint32_t*) &out);
    return out;
}
};
}


class BloomAdaptiveDecompressorOp : public OpKernel {

public:

    explicit BloomAdaptiveDecompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size", &bloom_size));
        OP_REQUIRES_OK(context, context->GetAttr("partitioning", &partitioning));
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));   // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("suffix", &suffix));                       // For debugging
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));                 // For debugging
    }

    int infer_k(float k, int items_partitions) {
        // Assume that the topk values are uniformly distributed inside the tensor
        int x = ceil(k/items_partitions);
        return x;
    }

    void infer_blooms_sizes(int bloom_size, float partitioning, std::vector<int>* bloom_sizes) {
        int m;
        float left_space = bloom_size;
        while (left_space != 0) {
            m = ceil(left_space/partitioning);
            bloom_sizes->push_back(m);
            left_space -= m;
        }
    }

    void Compute(OpKernelContext *context) override {

        // Retrieving Inputs
        const Tensor &compressed_tensor = context->input(0);
        const Tensor &decompressed_size_tensor = context->input(1);

        auto compressed_tensor_flat = compressed_tensor.flat<int8>();
        auto decompressed_size_flat = decompressed_size_tensor.flat<int>();

        int values_size = (compressed_tensor_flat.size()-bloom_size)/sizeof(int);
        int decompressed_size = *decompressed_size_flat.data();
        int items_space_size = decompressed_size;

        // Create an output tensor
        TensorShape decompressed_tensor_shape;
        decompressed_tensor_shape.AddDim(decompressed_size);
        Tensor *decompressed_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, decompressed_tensor_shape, &decompressed_tensor));
        auto decompressed_tensor_flat = decompressed_tensor->template flat<int>();
        memset(decompressed_tensor_flat.data(), 0, decompressed_size*sizeof(int));


        const int8 *ptr = compressed_tensor_flat.data();           // Note: int8 is 1 byte
        int values_bytes = values_size*sizeof(int);
        int *values_vec = (int*) malloc(values_bytes);
        memcpy(values_vec, ptr, values_bytes);
        ptr += values_bytes;

        std::vector<int> bloom_sizes;
        infer_blooms_sizes(bloom_size, partitioning, &bloom_sizes);

        int j=0, start=0, end, hash_num, partition_size, partitions_num = bloom_sizes.size();
        float m, k, items=items_space_size;
        k = infer_k((float) values_size, partitions_num);

        assert(items_space_size >= partitions_num);

        std::vector<bloom::OrdinaryBloomFilter<uint32_t>*> blooms;
        bloom::OrdinaryBloomFilter<uint32_t>* bloom;

        for (int i=0; i<bloom_sizes.size() && j<values_size; i++) {

            m = bloom_sizes[i];
            hash_num = int(ceil((m/k)*log(2)));

            partition_size = ceil(items/partitions_num);
            partitions_num--;
            items -= partition_size;
            end = start + partition_size;
            bloom = new bloom::OrdinaryBloomFilter<uint32_t>(hash_num, m, ptr);
//            bloom->print();
            blooms.push_back(bloom);
            ptr += (int) m;

            while (start != end && j < values_size) {
                if (bloom->Query(start)) {
                    decompressed_tensor_flat(start) = values_vec[j];
                    j++;
                }
                start++;
            }
        }

        // *********************** For Debugging ********************** //
        const Tensor &step_tensor = context->input(2);
        auto step = step_tensor.flat<int64>();
        if (verbosity != 0 && step(0) % verbosity == 0 ) {
            std::string str_suffix = std::to_string(logfile_suffix);
            std::string logs_suffix = std::to_string(logs_path_suffix);
            std::string str_step = std::to_string(step(0));
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + str_suffix + "/bloom_adaptive_decompressor_logs_" + str_suffix + "_" + std::to_string(suffix) + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            if (f==NULL)
                perror ("Can't open file");
            fprintf(f, "decompressed size: %d\n\n", decompressed_size);
            fprintf(f, "Bloom Sizes:\n");
            CompressionUtilities::print_vector(bloom_sizes.data(), bloom_sizes.size(), f);
            for (int i=0; i<blooms.size(); i++) {
                blooms[i]->fprint(f);
            }
            fprintf(f, "Values Vector:"); CompressionUtilities::print_vector(values_vec, values_size, f);
            fprintf(f, "Decompressed_tensor: %s\n", decompressed_tensor->DebugString(decompressed_tensor_flat.size()).c_str());
            fprintf(f, "########################################################################################\n\n");
            fclose (f);
        }
        // *********************** For Debugging ********************** //

        free(values_vec);
        for (int i=0; i<blooms.size(); i++) {
            free(blooms[i]);
        }
    }

private:
    int bloom_size;
    float partitioning;
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int suffix;             // For debugging
    int verbosity;          // For debugging
};


REGISTER_KERNEL_BUILDER(Name("BloomAdaptiveDecompressor").Device(DEVICE_CPU), BloomAdaptiveDecompressorOp);
