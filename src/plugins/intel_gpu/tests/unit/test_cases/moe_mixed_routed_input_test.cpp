// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Reproducer for the unresolved review thread discussion_r4409492342 on PR #38663 (CVS-195660).
//
// `prepare_moe_otd_params` (src/plugins/intel_gpu/src/plugin/ops/moe.cpp) disables OTD for a
// MOECompressed node when any routed input (3..11) is not a pure `OffloadableRoutedExpert`
// constant, e.g. when one expert weight is provided as an input tensor. The node is then built
// with the fully resident weight provider (lru_expert_num == 0).
//
// However, `try_prepare_partial_upload` (src/plugins/intel_gpu/src/plugin/ops/moe_offload_constant.cpp)
// decides per constant and still allocates the *other* routed constants of the same node with only
// `max(1, num_expert * (100 - OFFLOAD_RATIO) / 100)` expert slots and marks their data nodes with
// skip_device_transfer, so the full data is never uploaded. The resident kernels read the routed
// constants using the full [num_expert, ...] layout and therefore read uninitialized/out-of-bounds
// memory, producing wrong results.
//
// Tests:
// - ratio75_tensor_weight_matches_ratio0: reproduces the mismatch through the "routed input is
//   a tensor, siblings are pure constants" trigger of the same fallback branch.
// - mixed_consumer_constant_ratio75_matches_ratio0: the literal case from discussion
//   _r4409492342 - one routed constant additionally feeds a non-MoE consumer (role
//   NotOffloadableRoutedExpert).
// Both compile the same model with OFFLOAD_RATIO=0 (reference) and OFFLOAD_RATIO=75; the
// fallback to the fully resident path must make the results identical. Both tests FAIL on the
// reviewed revision (all MoE output elements differ).
// - all_constant_routed_ratio75_requires_weights_path: control. With all nine routed inputs as
//   pure constants, OTD is genuinely enabled and compilation fails on the missing
//   ov::weights_path, which proves the mismatches above are caused by the fallback branch.
//
// After the whole-MoE eligibility decision is shared between the per-node OTD gate,
// the per-constant partial upload, and the resident-bytes estimator, both tests must pass.

#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "ov_ops/moe_compressed.hpp"

using namespace tests;

namespace {

constexpr size_t NUM_EXPERTS = 4;
constexpr size_t HIDDEN_SIZE = 128;
constexpr size_t INTER_SIZE = 256;
constexpr size_t GROUP_SIZE = 128;
constexpr size_t TOP_K = 2;

// Builds a GEMM3_SWIGLU MOECompressed model without shared experts. When w0_as_input is true,
// routed input 3 (weight_0) is a model input instead of a Constant, which makes the node fall
// back to resident mode while the remaining eight routed constants stay eligible for partial
// upload.
std::shared_ptr<ov::Model> build_mixed_moe_model(bool w0_as_input) {
    const size_t group_num = HIDDEN_SIZE / GROUP_SIZE;
    const size_t group_num2 = INTER_SIZE / GROUP_SIZE;

    ov::op::internal::MOECompressed::Config config{};
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    config.hidden_size = HIDDEN_SIZE;
    config.inter_size = INTER_SIZE;
    config.num_expert = NUM_EXPERTS;
    config.num_shared_expert = 0;
    config.top_k = TOP_K;
    config.group_size = GROUP_SIZE;
    config.has_zp = true;
    config.out_type = ov::element::f16;

    ov::ParameterVector parameters;
    ov::OutputVector inputs;

    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, HIDDEN_SIZE});
    auto routing = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, TOP_K});
    auto topk_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, TOP_K});
    parameters = {hidden, routing, topk_idx};
    inputs = {hidden->output(0), routing->output(0), topk_idx->output(0)};

    auto make_const = [](ov::element::Type type, const ov::Shape& shape, float value) {
        return std::make_shared<ov::op::v0::Constant>(type, shape, static_cast<double>(value));
    };

    const ov::Shape w_shape{NUM_EXPERTS, INTER_SIZE, GROUP_SIZE, group_num};
    if (w0_as_input) {
        auto w0 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, w_shape);
        parameters.push_back(w0);
        inputs.push_back(w0->output(0));
    } else {
        inputs.push_back(make_const(ov::element::u8, w_shape, 3.0f)->output(0));  // 3: w0_weight
    }
    inputs.push_back(make_const(ov::element::f16, {NUM_EXPERTS, INTER_SIZE, group_num, 1}, 0.01f)->output(0));  // 4: s0
    inputs.push_back(make_const(ov::element::u8, {NUM_EXPERTS, INTER_SIZE, group_num, 1}, 1.0f)->output(0));    // 5: z0
    inputs.push_back(make_const(ov::element::u8, w_shape, 4.0f)->output(0));                                     // 6: w1
    inputs.push_back(make_const(ov::element::f16, {NUM_EXPERTS, INTER_SIZE, group_num, 1}, 0.01f)->output(0));  // 7: s1
    inputs.push_back(make_const(ov::element::u8, {NUM_EXPERTS, INTER_SIZE, group_num, 1}, 1.0f)->output(0));    // 8: z1
    inputs.push_back(make_const(ov::element::u8, {NUM_EXPERTS, HIDDEN_SIZE, GROUP_SIZE, group_num2}, 5.0f)->output(0));  // 9: w2
    inputs.push_back(make_const(ov::element::f16, {NUM_EXPERTS, HIDDEN_SIZE, group_num2, 1}, 0.01f)->output(0));         // 10: s2
    inputs.push_back(make_const(ov::element::u8, {NUM_EXPERTS, HIDDEN_SIZE, group_num2, 1}, 1.0f)->output(0));           // 11: z2

    auto moe = std::make_shared<ov::op::internal::MOECompressed>(inputs, config);
    auto result = std::make_shared<ov::op::v0::Result>(moe);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, parameters);
}

// Copilot discussion_r4409492342 literal case: one routed *constant* (weight_0) additionally
// feeds a non-MoE consumer that cannot be constant-folded away because it also consumes a
// runtime input. Role becomes NotOffloadableRoutedExpert -> the node falls back to resident
// mode, while the sibling routed constants are still partially uploaded.
std::shared_ptr<ov::Model> build_mixed_consumer_moe_model() {
    const size_t group_num = HIDDEN_SIZE / GROUP_SIZE;
    const size_t group_num2 = INTER_SIZE / GROUP_SIZE;

    ov::op::internal::MOECompressed::Config config{};
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    config.hidden_size = HIDDEN_SIZE;
    config.inter_size = INTER_SIZE;
    config.num_expert = NUM_EXPERTS;
    config.num_shared_expert = 0;
    config.top_k = TOP_K;
    config.group_size = GROUP_SIZE;
    config.has_zp = true;
    config.out_type = ov::element::f16;

    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, HIDDEN_SIZE});
    auto routing = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, TOP_K});
    auto topk_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, TOP_K});
    ov::ParameterVector parameters = {hidden, routing, topk_idx};
    ov::OutputVector inputs = {hidden->output(0), routing->output(0), topk_idx->output(0)};

    auto make_const = [](ov::element::Type type, const ov::Shape& shape, float value) {
        return std::make_shared<ov::op::v0::Constant>(type, shape, static_cast<double>(value));
    };

    const ov::Shape w_shape{NUM_EXPERTS, INTER_SIZE, GROUP_SIZE, group_num};
    auto w0 = make_const(ov::element::u8, w_shape, 3.0f);
    inputs.push_back(w0->output(0));  // 3: w0_weight (shared with a non-MoE consumer below)
    inputs.push_back(make_const(ov::element::f16, {NUM_EXPERTS, INTER_SIZE, group_num, 1}, 0.01f)->output(0));  // 4: s0
    inputs.push_back(make_const(ov::element::u8, {NUM_EXPERTS, INTER_SIZE, group_num, 1}, 1.0f)->output(0));    // 5: z0
    inputs.push_back(make_const(ov::element::u8, w_shape, 4.0f)->output(0));                                     // 6: w1
    inputs.push_back(make_const(ov::element::f16, {NUM_EXPERTS, INTER_SIZE, group_num, 1}, 0.01f)->output(0));  // 7: s1
    inputs.push_back(make_const(ov::element::u8, {NUM_EXPERTS, INTER_SIZE, group_num, 1}, 1.0f)->output(0));    // 8: z1
    inputs.push_back(make_const(ov::element::u8, {NUM_EXPERTS, HIDDEN_SIZE, GROUP_SIZE, group_num2}, 5.0f)->output(0));  // 9: w2
    inputs.push_back(make_const(ov::element::f16, {NUM_EXPERTS, HIDDEN_SIZE, group_num2, 1}, 0.01f)->output(0));         // 10: s2
    inputs.push_back(make_const(ov::element::u8, {NUM_EXPERTS, HIDDEN_SIZE, group_num2, 1}, 1.0f)->output(0));           // 11: z2

    auto moe = std::make_shared<ov::op::internal::MOECompressed>(inputs, config);

    // Non-foldable second consumer of w0: Multiply(w0, broadcast runtime input).
    auto hidden_u8 = std::make_shared<ov::op::v0::Convert>(hidden, ov::element::u8);
    auto hidden_bc = std::make_shared<ov::op::v1::Reshape>(hidden_u8,
                                                           ov::op::v0::Constant::create(ov::element::i64,
                                                                                        {4},
                                                                                        std::vector<int64_t>{1, 1, 1, static_cast<int64_t>(HIDDEN_SIZE)}),
                                                           true);
    auto extra = std::make_shared<ov::op::v1::Multiply>(w0, hidden_bc);

    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(moe),
                                                        std::make_shared<ov::op::v0::Result>(extra)},
                                       parameters);
}

// Returns the first GPU-usable device name known to ov::Core, or an empty string.
std::string pick_gpu_device(ov::Core& core) {
    try {
        const auto devices = core.get_available_devices();
        for (const auto* name : {"GPU", "GPU.0", "GPU.1"}) {
            if (std::find(devices.begin(), devices.end(), std::string(name)) != devices.end()) {
                return name;
            }
        }
    } catch (const std::exception& ex) {
        printf("ov::Core device query failed: %s\n", ex.what());
    }
    return {};
}

std::vector<ov::float16> run_model(ov::CompiledModel& compiled, const std::shared_ptr<ov::Model>& model) {
    auto req = compiled.create_infer_request();
    const auto inputs = model->inputs();

    auto hidden_t = ov::Tensor(ov::element::f16, {1, 1, HIDDEN_SIZE});
    std::fill_n(hidden_t.data<ov::float16>(), HIDDEN_SIZE, ov::float16(0.1f));
    auto routing_t = ov::Tensor(ov::element::f16, {1, 1, TOP_K});
    routing_t.data<ov::float16>()[0] = ov::float16(0.5f);
    routing_t.data<ov::float16>()[1] = ov::float16(0.5f);
    auto indices_t = ov::Tensor(ov::element::i32, {1, 1, TOP_K});
    indices_t.data<int32_t>()[0] = 0;
    indices_t.data<int32_t>()[1] = 1;

    req.set_tensor(inputs[0], hidden_t);
    req.set_tensor(inputs[1], routing_t);
    req.set_tensor(inputs[2], indices_t);
    if (inputs.size() > 3) {
        auto w0_t = ov::Tensor(ov::element::u8, {NUM_EXPERTS, INTER_SIZE, GROUP_SIZE, HIDDEN_SIZE / GROUP_SIZE});
        std::fill_n(w0_t.data<uint8_t>(), w0_t.get_size(), static_cast<uint8_t>(3));
        req.set_tensor(inputs[3], w0_t);
    }
    req.infer();

    const auto out = req.get_output_tensor(0);
    return std::vector<ov::float16>(out.data<ov::float16>(), out.data<ov::float16>() + out.get_size());
}

}  // namespace

// Reproducer: with one routed weight provided as a tensor input, the MoE node must behave as a
// fully resident node. Compiling the same model with OFFLOAD_RATIO=75 instead of 0 must not
// change the results. On the reviewed revision the sibling routed constants are partially
// uploaded for the (disabled) OTD path, so the resident path reads undersized data.
TEST(moe_mixed_routed_input, ratio75_tensor_weight_matches_ratio0) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "The MoE 3GEMM oneDNN path requires supports_immad";
    }
    ov::Core core;
    const std::string device = pick_gpu_device(core);
    if (device.empty()) {
        GTEST_SKIP() << "GPU plugin is not available to ov::Core";
    }

    auto model = build_mixed_moe_model(true /* w0_as_input */);

    auto compiled_ref = core.compile_model(model, device, {ov::intel_gpu::offload_ratio(0)});
    const auto out_ref = run_model(compiled_ref, model);

    auto compiled_ratio = core.compile_model(model, device, {ov::intel_gpu::offload_ratio(75)});
    const auto out_ratio = run_model(compiled_ratio, model);

    ASSERT_EQ(out_ref.size(), out_ratio.size());
    size_t mismatch = 0;
    float max_abs_diff = 0.f;
    for (size_t i = 0; i < out_ref.size(); ++i) {
        const float a = static_cast<float>(out_ref[i]);
        const float b = static_cast<float>(out_ratio[i]);
        if (std::fabs(a - b) > 1e-3f) {
            ++mismatch;
            max_abs_diff = std::max(max_abs_diff, std::fabs(a - b));
        }
    }
    EXPECT_EQ(mismatch, 0U) << "OFFLOAD_RATIO=75 with mixed routed inputs must match OFFLOAD_RATIO=0 "
                            << "(the node is expected to fall back to the fully resident path). "
                            << "max_abs_diff=" << max_abs_diff;
}

// Copilot discussion_r4409492342 literal case: a routed constant shared with a non-MoE
// consumer (role NotOffloadableRoutedExpert) must make the node fully resident, so
// OFFLOAD_RATIO=75 and OFFLOAD_RATIO=0 must agree. If the pipeline constant-folds the extra
// consumer away the node becomes purely offloadable; then ratio-75 compilation demands
// ov::weights_path and the test skips with that diagnostic instead of failing.
TEST(moe_mixed_routed_input, mixed_consumer_constant_ratio75_matches_ratio0) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "The MoE 3GEMM oneDNN path requires supports_immad";
    }
    ov::Core core;
    const std::string device = pick_gpu_device(core);
    if (device.empty()) {
        GTEST_SKIP() << "GPU plugin is not available to ov::Core";
    }

    auto model = build_mixed_consumer_moe_model();
    auto compiled_ref = core.compile_model(model, device, {ov::intel_gpu::offload_ratio(0)});
    const auto out_ref = run_model(compiled_ref, model);

    ov::CompiledModel compiled_ratio;
    try {
        compiled_ratio = core.compile_model(model, device, {ov::intel_gpu::offload_ratio(75)});
    } catch (const std::exception& ex) {
        const std::string msg = ex.what();
        if (msg.find("weights_path") != std::string::npos) {
            GTEST_SKIP() << "Mixed routed consumer was eliminated before program build "
                            "(constant folding); node took the OTD-enabled branch: " << msg;
        }
        FAIL() << "unexpected compile failure: " << msg;
    }

    const auto out_ratio = run_model(compiled_ratio, model);
    ASSERT_EQ(out_ref.size(), out_ratio.size());
    size_t mismatch = 0;
    float max_abs_diff = 0.f;
    for (size_t i = 0; i < out_ref.size(); ++i) {
        const float a = static_cast<float>(out_ref[i]);
        const float b = static_cast<float>(out_ratio[i]);
        if (std::fabs(a - b) > 1e-3f) {
            ++mismatch;
            max_abs_diff = std::max(max_abs_diff, std::fabs(a - b));
        }
    }
    EXPECT_EQ(mismatch, 0U) << "OFFLOAD_RATIO=75 with a routed constant shared with a non-MoE consumer "
                            << "must match OFFLOAD_RATIO=0. max_abs_diff=" << max_abs_diff;
}

// Control: with all nine routed inputs as pure constants the node really enables OTD, so
// compilation must fail on the missing ov::weights_path instead of running with shrunk data.
TEST(moe_mixed_routed_input, all_constant_routed_ratio75_requires_weights_path) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "The MoE 3GEMM oneDNN path requires supports_immad";
    }
    ov::Core core;
    const std::string device = pick_gpu_device(core);
    if (device.empty()) {
        GTEST_SKIP() << "GPU plugin is not available to ov::Core";
    }

    auto model = build_mixed_moe_model(false /* w0_as_input */);
    try {
        auto compiled = core.compile_model(model, device, {ov::intel_gpu::offload_ratio(75)});
        FAIL() << "Expected OTD to be enabled and ov::weights_path to be required for all-constant routed inputs";
    } catch (const std::exception& ex) {
        const std::string msg = ex.what();
        EXPECT_NE(msg.find("weights_path"), std::string::npos) << "unexpected failure: " << msg;
    }
}
