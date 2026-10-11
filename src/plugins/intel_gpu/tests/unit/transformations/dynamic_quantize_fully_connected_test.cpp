// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/dynamic_quantize_fully_connected.hpp"

#include <memory>
#include <string>
#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "transformations/init_node_info.hpp"

namespace ov::test::intel_gpu {
namespace {

using DynamicQuantize = ov::op::internal::DynamicQuantize;
using FullyConnectedCompressed = ov::intel_gpu::op::FullyConnectedCompressed;
using Placeholder = ov::intel_gpu::op::Placeholder;
using TestParams = std::tuple<size_t, bool>;

constexpr size_t ifm = 128;
constexpr size_t groups = 2;
constexpr uint64_t activation_group_size = 32;

std::shared_ptr<ov::Model> getModel(size_t ofm, bool transpose_parameters) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, ifm});
    auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ofm, ifm}, {3});
    const auto param_shape = transpose_parameters ? ov::Shape{ofm, groups} : ov::Shape{groups, ofm};
    auto scale = ov::op::v0::Constant::create(ov::element::f16, param_shape, {2});
    auto zp = ov::op::v0::Constant::create(ov::element::u8, param_shape, {1});
    auto bias = std::make_shared<Placeholder>();
    auto fc = std::make_shared<FullyConnectedCompressed>(input, weights, bias, scale, zp, ov::element::f16, true, transpose_parameters, transpose_parameters);
    return std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> getModelRef(size_t ofm, bool transpose_parameters) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, ifm});
    auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ofm, ifm}, {3});
    const auto param_shape = transpose_parameters ? ov::Shape{ofm, groups} : ov::Shape{groups, ofm};
    auto scale = ov::op::v0::Constant::create(ov::element::f16, param_shape, {2});
    auto zp = ov::op::v0::Constant::create(ov::element::u8, param_shape, {1});
    auto bias = std::make_shared<Placeholder>();

    DynamicQuantize::Attributes attrs;
    attrs.quantization_type = DynamicQuantize::QuantizationType::Symmetric;
    attrs.quantization_dt = ov::element::i8;
    attrs.scale_dt = ov::element::f16;
    attrs.group_sizes = {1, 1, activation_group_size};
    attrs.precomputed_reduction = true;
    attrs.precomputed_reduction_dt = ov::element::i32;
    auto dq = std::make_shared<DynamicQuantize>(input, attrs);
    auto activation_zp = std::make_shared<Placeholder>();
    auto fc = std::make_shared<FullyConnectedCompressed>(dq->output(0),
                                                         weights,
                                                         bias,
                                                         scale,
                                                         zp,
                                                         dq->output(1),
                                                         activation_zp,
                                                         dq->output(2),
                                                         ov::element::f16,
                                                         true,
                                                         transpose_parameters,
                                                         transpose_parameters);
    return std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
}

}  // namespace

class DynamicQuantizeFullyConnectedTest : public TransformationTestsF, public ::testing::WithParamInterface<TestParams> {
public:
    DynamicQuantizeFullyConnectedTest() {
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }

    static std::string getTestCaseName(const ::testing::TestParamInfo<TestParams>& info) {
        const auto& [ofm, transpose_parameters] = info.param;
        return "N" + std::to_string(ofm) + (transpose_parameters ? "_TransposedParameters" : "_NonTransposedParameters");
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [ofm, transpose_parameters] = GetParam();
        model = getModel(ofm, transpose_parameters);
        model_ref = getModelRef(ofm, transpose_parameters);
        manager.register_pass<ov::intel_gpu::DynamicQuantizeFullyConnected>(activation_group_size);
        // DQ does not initialize metadata on its new nodes; compare quantization independently of that behavior.
        manager.register_pass<ov::pass::InitNodeInfo>();
    }
};

TEST_P(DynamicQuantizeFullyConnectedTest, PreservesGroupedQuantizationAndPrecomputedReduction) {}

INSTANTIATE_TEST_SUITE_P(smoke,
                         DynamicQuantizeFullyConnectedTest,
                         ::testing::Combine(::testing::Values(size_t{64}, size_t{256}), ::testing::Bool()),
                         DynamicQuantizeFullyConnectedTest::getTestCaseName);

}  // namespace ov::test::intel_gpu
