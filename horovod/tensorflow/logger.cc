#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "./compression_utils.hpp"

#include <cmath>
#include <string>

using namespace tensorflow;

REGISTER_OP("Logger")
.Attr("num_of_coefficients: int")
.Attr("K: int")
.Attr("bloom_logs_path: string")
.Attr("gradient_id: int")
.Attr("rank: int")
.Attr("verbosity_frequency: int")
.Attr("verbosity: int")
.Input("initial_tensor: float32")
.Input("estimated_tensor: float32")
.Input("theta_estimates: float32")
.Input("step: int64")
.Doc(R"doc()doc");

class LoggerOp : public OpKernel {

public:

    explicit LoggerOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("K", &K));
        OP_REQUIRES_OK(context, context->GetAttr("num_of_coefficients", &num_of_coefficients));
        OP_REQUIRES_OK(context, context->GetAttr("bloom_logs_path", &bloom_logs_path));
        OP_REQUIRES_OK(context, context->GetAttr("gradient_id", &gradient_id));
        OP_REQUIRES_OK(context, context->GetAttr("rank", &rank));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity", &verbosity));
        OP_REQUIRES_OK(context, context->GetAttr("verbosity_frequency", &verbosity_frequency));
    }

    void Compute(OpKernelContext *context) override {

        // Retrieving Inputs
        const Tensor &initial_tensor = context->input(0); auto initial_flat = initial_tensor.flat<float>();
        const Tensor &estimated_tensor = context->input(1);  auto estimated_tensor_flat = estimated_tensor.flat<float>();
        const Tensor &theta_estimates_tensor = context->input(2);
        int64 step = context->input(3).flat<int64>()(0);
        int N = initial_flat.size();

        // *********************** Logging ********************** //
        if (verbosity_frequency != 0 && step % verbosity_frequency == 0 ) {
            CompressionUtilities::logging(N, K, num_of_coefficients, initial_tensor, estimated_tensor, theta_estimates_tensor, bloom_logs_path, gradient_id,
                                        step, rank, verbosity);
        }
        // *********************** Logging ********************** //
    }

private:
    // Logging
    int num_of_coefficients;
    int K;
    string bloom_logs_path;
    int gradient_id;
    int rank;
    int verbosity_frequency;
    int verbosity;
};
REGISTER_KERNEL_BUILDER(Name("Logger").Device(DEVICE_CPU), LoggerOp);
