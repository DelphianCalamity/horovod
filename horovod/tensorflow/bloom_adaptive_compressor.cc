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

#include <assert.h>
#include <cmath>
#include <string>
#include <cstdlib>

using namespace tensorflow;

REGISTER_OP("BloomAdaptiveCompressor")
.Attr("T: {int32, int64, float16, float32, float64}")
.Attr("partitioning: float")
.Attr("bloom_size: int")
.Attr("logfile_suffix: int")       // For debugging
.Attr("logs_path_suffix: int")     // For debugging
.Attr("verbosity: int")            // For debugging
.Input("values: T")
.Input("indices: int32")
.Input("initial_tensor: int32")    // For debugging
.Input("step: int64")              // For debugging
.Output("compressed_tensor: int8")
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

class BloomAdaptiveCompressorOp : public OpKernel {

public:

    explicit BloomAdaptiveCompressorOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("partitioning", &partitioning));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_size", &bloom_size));
        OP_REQUIRES_OK(context, context->GetAttr("logfile_suffix", &logfile_suffix));
        OP_REQUIRES_OK(context, context->GetAttr("logs_path_suffix", &logs_path_suffix));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));
    }


    int find(const Tensor& indices, int x) {
        auto indices_flat = indices.flat<int>();
        for (int i=0; i<indices_flat.size(); ++i) {   // Dummy lookup
            if (indices_flat(i) == x)
                return 1;
        }
        return 0;
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
        const Tensor &values = context->input(0);  auto values_flat = values.flat<int>();
        const Tensor &indices = context->input(1);  auto indices_flat = indices.flat<int>();
        const Tensor &initial_tensor = context->input(2); auto initial_flat = initial_tensor.flat<int>();

        int values_size = values_flat.size();
        int items_space_size = initial_flat.size();
        int output_concat_dim = values_size*sizeof(int) + bloom_size;

        // Create an output tensor
        TensorShape output_shape; output_shape.AddDim(output_concat_dim); Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->template flat<int8>();
        int8* out_ptr = output_flat.data(); const void* values_ptr = values_flat.data();
        memcpy(out_ptr, values_ptr, values_size*sizeof(int));
        int offset = values_size*sizeof(int);
        //////////////////////////////////////////////////////////////////////////////////

        std::vector<int> bloom_sizes;
        infer_blooms_sizes(bloom_size, partitioning, &bloom_sizes);

        int start=0, end, hash_num, partition_size, partitions_num = bloom_sizes.size();
        float m, k, items=items_space_size;
        k = infer_k((float) values_size, partitions_num);

        assert(items_space_size >= partitions_num);

        std::vector<bloom::OrdinaryBloomFilter<uint32_t>*> blooms;
        bloom::OrdinaryBloomFilter<uint32_t>* bloom;
        for (int i=0; i<bloom_sizes.size(); i++) {

            m = bloom_sizes[i];
            hash_num = int(ceil((m/k)*log(2)));
            assert(hash_num > 0);

            partition_size = ceil(items/partitions_num);
            partitions_num--;
            items -= partition_size;

            end = start + partition_size;
            bloom = new bloom::OrdinaryBloomFilter<uint32_t>(hash_num, m);
            for (int x=0; x<indices_flat.size(); ++x) {
                if (indices_flat(x) >= start && indices_flat(x) < end) {    // If it is within the partition -- optimize this
                    bloom->Insert(indices_flat(x));
                }
            }
//            bloom->print();
            blooms.push_back(bloom);
            std::vector<unsigned char> &bloom_vec = bloom->Get_bloom();
            std::copy(bloom_vec.begin(), bloom_vec.end(), out_ptr+offset);
            offset += m;
            start = end;
        }

        //////////////////////////////////////////////////////////////////////////////////


        // *********************** For Debugging ********************** //
        const Tensor &step_tensor = context->input(3);
        auto step = step_tensor.flat<int64>();

        if (verbosity != 0 && step(0) % verbosity == 0 ) {

            // Compute False Positives per bloom filter
            start=0;
            items=items_space_size;
            partitions_num=bloom_sizes.size();
            int false_positives;
            std::vector<int> false_positives_vec;
            std::vector<int> partition_size_vec;

            for (int i=0; i<bloom_sizes.size(); i++) {
                partition_size = ceil(items/partitions_num);
                partition_size_vec.push_back(partition_size);
                partitions_num--;
                items -= partition_size;
                false_positives = 0;
                end = start + partition_size;
                while (start != end) {
                    if (blooms[i]->Query(start) && !find(indices, start)) {
                        false_positives++;
                    }
                    start++;
                }
                false_positives_vec.push_back(false_positives);
            }

            std::string suffix = std::to_string(logfile_suffix);
            std::string logs_suffix = std::to_string(logs_path_suffix);
            std::string str_step = std::to_string(step(0));

            std::string cmd = "mkdir -p logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/";
            int systemRet = system(cmd.c_str());
            if(systemRet == -1){
                perror("mkdir failed");
            }
            std::string str = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/bloom_adaptive_compressor_logs_" + suffix + ".txt";
            FILE* f = fopen(str.c_str(),"w");
            if (f==NULL) {
                perror ("Can't open file");
            }
            fprintf(f, "\nInitial Tensor: %s\n\n", initial_tensor.DebugString(initial_flat.size()).c_str());
            fprintf(f, "Values: %s\n", values.DebugString(values_flat.size()).c_str());
            fprintf(f, "Indices: %s\n\n", indices.DebugString(indices_flat.size()).c_str());
            fprintf(f, "Partitioning: %f\n", partitioning);
            fprintf(f, "Item partitions: %d\n", bloom_sizes.size());
            fprintf(f, "Inferred K: %f\n\n", k);

            fprintf(f, "Bloom Sizes:");
            CompressionUtilities::print_vector(bloom_sizes.data(), bloom_sizes.size(), f);
            for (int i=0; i<blooms.size(); i++) {
                blooms[i]->fprint(f);
            }
            fprintf(f, "Output_concat_size: = %d\n\n", output_concat_dim);
            for (int i=0; i<false_positives_vec.size(); i++) {
                fprintf(f, "FalsePositives: %d Total: %d\n", false_positives_vec[i], partition_size_vec[i]);
            }
            fprintf(f, "\n\n########################################################################################\n\n");
            fclose(f);

            std::string str1 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/fpr_" + suffix + ".txt";
            f = fopen(str1.c_str(),"w");
            for (int i=0; i<false_positives_vec.size(); i++) {
                fprintf(f, "FalsePositives: %d Total: %d\n", false_positives_vec[i], partition_size_vec[i]);
            }
            fclose(f);

            std::string str3 = "logs" + logs_suffix + "/step_" + str_step + "/" + suffix + "/stats" + suffix + ".txt";
            f = fopen(str3.c_str(),"w");
            fprintf(f, "Initial_Size: %d  Final_Size: %d\n", initial_flat.size() /*in bits*/,  bloom_size*8 /*in bits*/);
            fclose(f);
        }
        // *********************** For Debugging ********************** //

        for (int i=0; i<blooms.size(); i++) {
            free(blooms[i]);
        }
    }

private:
    float partitioning;
    int bloom_size;
    int logfile_suffix;     // For debugging
    int logs_path_suffix;   // For debugging
    int verbosity;          // For debugging
};


REGISTER_KERNEL_BUILDER(Name("BloomAdaptiveCompressor").Device(DEVICE_CPU), BloomAdaptiveCompressorOp);
