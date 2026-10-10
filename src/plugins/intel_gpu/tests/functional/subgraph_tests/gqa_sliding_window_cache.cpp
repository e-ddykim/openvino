// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Reproducer for the sliding_window_cache decode regression introduced by the native-SWA
// mask elision in ov::intel_gpu::GroupQueryAttentionDecomposition.
//
// For a windowed (rolling) KV cache with a statically-known single-token query, the common
// decomposition assembles the attention K/V as the full capacity-C buffer: rows [0, kept) hold
// the resident tokens, rows [kept, C) are zero padding (ScatterUpdate into a zeroed buffer).
// The GPU override then elides the explicit attention mask and relies on the kernel-side
// is_causal + sliding_window_size. But LOWER_RIGHT causal + window is anchored to the *buffer*
// length C, not to the kept valid rows, so the kernel attends over the zero-padding tail instead
// of the window of valid rows. The pre-elision explicit mask was anchored to the valid rows
// (mask_past_seqlen = kept) and produced the correct window.
//
// The reference below is the common (CPU) decomposition of the same GQA node - i.e. exactly the
// graph the GPU used before the elision - so the test is fix-agnostic: any fix that restores the
// correct attention span (keep the mask, slice K/V to the valid rows, pass the valid length to the
// kernel, ...) makes it pass.

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/identity.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/op_conversions/group_query_attention_decomposition.hpp"

namespace {

using QuantType = ov::op::internal::GroupQueryAttentionQuantType;

struct GQASWAParams {
    ov::element::Type dtype;
    int32_t seqlens_k;  // ONNX convention: total (past + current) == seqlens_k + 1
};

class GQASlidingWindowCacheDecode : public testing::WithParamInterface<GQASWAParams>,
                                    virtual public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        // f16 exercises the sdpa_micro path on micro-capable devices, f32 the sdpa_opt stage
        // kernels; both must give the correct attention span.
        configuration.insert(ov::hint::inference_precision(GetParam().dtype));

        const int64_t num_heads = 4;
        const int64_t kv_num_heads = 2;
        const int64_t head_size = 64;
        const int64_t capacity = 512;
        const int64_t local_window_size = 128;
        const auto dtype = GetParam().dtype;
        const int32_t seqlens_k = GetParam().seqlens_k;

        auto query = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{1, num_heads, 1, head_size});
        auto key = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{1, kv_num_heads, 1, head_size});
        auto value = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{1, kv_num_heads, 1, head_size});
        auto past_key = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{1, kv_num_heads, capacity, head_size});
        auto past_value = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{1, kv_num_heads, capacity, head_size});

        const auto seqlens = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {seqlens_k});
        const auto total_sequence_length = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {seqlens_k});

        ov::OutputVector inputs(14);
        inputs[0] = query;
        inputs[1] = key;
        inputs[2] = value;
        inputs[3] = past_key;
        inputs[4] = past_value;
        inputs[5] = seqlens;
        inputs[6] = total_sequence_length;
        for (size_t i = 7; i <= 13; ++i) {
            inputs[i] = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
        }

        auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(inputs,
                                                                          num_heads,
                                                                          kv_num_heads,
                                                                          0.0f,   // scale: use 1/sqrt(head_size)
                                                                          false,  // do_rotary
                                                                          false,  // rotary_interleaved
                                                                          0,      // kv_cache_bit_width
                                                                          QuantType::NONE,
                                                                          QuantType::NONE,
                                                                          local_window_size,
                                                                          true,   // sliding_window_cache
                                                                          false,  // smooth_softmax
                                                                          true);  // causal

        ov::ResultVector results;
        for (const auto& output : gqa->outputs()) {
            // Identity keeps the Result's input node type in the framework's compare map
            // (GroupQueryAttention is an internal op with no registered comparator).
            results.push_back(std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v16::Identity>(output)));
        }
        ov::ParameterVector parameters{query, key, value, past_key, past_value};
        function = std::make_shared<ov::Model>(results, parameters, "gqa_sliding_window_cache_decode");

        // Reference: the common decomposition (explicit mask anchored to the valid rows),
        // which is what the GPU pipeline produced before the mask elision.
        functionRefs = function->clone();
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::GroupQueryAttentionDecomposition>();
        manager.run_passes(functionRefs);

        abs_threshold = 0.01;
        rel_threshold = 0.01;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        for (const auto& param : function->get_parameters()) {
            const auto tensor = ov::test::utils::create_and_fill_tensor(
                param->get_element_type(), param->get_shape(), ov::test::utils::InputGenerateData(0, 1, 8, 1));
            inputs.insert({param, tensor});
        }
    }
};

TEST_P(GQASlidingWindowCacheDecode, Inference) {
    run();
}

// seqlens_k + 1 = total tokens. 300: cache below capacity (valid rows 0..300 of 512) - the
// failing scenario. 511: cache exactly full - control, correct on both old and new behaviour.
INSTANTIATE_TEST_SUITE_P(smoke_GQASlidingWindowCacheDecode,
                         GQASlidingWindowCacheDecode,
                         ::testing::Values(GQASWAParams{ov::element::f32, 300},
                                           GQASWAParams{ov::element::f32, 511},
                                           GQASWAParams{ov::element::f16, 300},
                                           GQASWAParams{ov::element::f16, 511}),
                         [](const testing::TestParamInfo<GQASWAParams>& info) {
                             return std::string(info.param.dtype.get_type_name()) +
                                    "_seqlens_k_" + std::to_string(info.param.seqlens_k);
                         });

}  // namespace
