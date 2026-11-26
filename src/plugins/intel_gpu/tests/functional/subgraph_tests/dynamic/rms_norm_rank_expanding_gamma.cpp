// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Reproducer for the rank-expanding (NUMPY-broadcast) RMSNorm gamma issue.
//
// Contract under test (plain OV IR semantics, no GPU-specific behaviour):
//   x     : f32 [1, 2, 2, C]     (normalization over the last axis)
//   gamma : f32 [1, 1, 1, 1, C]  -> right-aligned NUMPY broadcast
//   y     = x / sqrt(ReduceMean(x^2, -1, keep_dims) + eps) * gamma
//   expected output shape = right-aligned broadcast of data and gamma
//
// The reference is produced by the template (CPU) plugin from the same model, so the
// assertion is pure broadcast semantics and independent of the GPU RMS kernels.
//
// Why this test exists: a gamma whose rank exceeds the data rank is a valid NUMPY
// broadcast that expands the output rank (here [1,2,2,C] * [1,1,1,1,C] -> [1,1,2,2,C]).
// The GPU RMS lowering does not preserve that expansion: the fused RMS derives its
// output layout from its rank-4 input, so the executed tensor is [1,2,2,C] while the
// model publishes [1,1,2,2,C].
//
// Measured behaviour of this reproducer (Arc A770 + UHD 770, f32).
// "shape" is the tensor rank/shape published by the compiled model versus the tensor
// actually returned; "values" are compared on the flat buffer against an f64 reference.
//
//   gamma prefix all-singleton, e.g. [1,1,1,1,C]:
//     pre-PR  : shape mismatch only  - the returned rank-4 buffer is a flat reshape of
//               the expected rank-5 data, so per-element values are correct (~1e-3).
//     post-PR : shape mismatch PLUS wrong values (~2.56), because the rank rejection
//               leaves a rank-5 Multiply that gets folded into the rank-4 RMS primitive.
//               This part is a regression.
//
//   gamma varying along the expanded axis, e.g. [1,1,1,2,C]:
//     pre-PR  : shape mismatch AND wrong values (~2.10) - the original grouped-gamma
//               defect, surfacing in the rank-expanded form.
//     post-PR : shape mismatch AND wrong values (~2.56) - the rank rejection cannot fix
//               the grouped defect because it drops the whole broadcast instead.
//
// So the shape half of this contract is a pre-existing defect for both variants, while
// the value half is a regression only for the all-singleton prefix variant. Both are
// asserted here because a caller cannot tell the difference from the published shape.
// The test is expected to fail until the rank-expanding broadcast is either preserved
// end-to-end or such gamma is kept fully outside the fused RMS (see the review notes
// for the trade-offs).
//
// Shapes are intentionally small: the intent is to pin the contract, not to benchmark.

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

using RMSNormRankExpandingGammaParams = std::tuple<std::vector<InputShape>,  // input shapes
                                                   ov::element::Type,        // input precision
                                                   ov::Shape>;               // gamma shape

class RMSNormRankExpandingGamma : public testing::WithParamInterface<RMSNormRankExpandingGammaParams>,
                                  virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RMSNormRankExpandingGammaParams>& obj) {
        const auto& [input_shapes, input_precision, gamma_shape] = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (const auto& shape : input_shapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "input_precision=" << input_precision;
        result << "_gamma_shape=" << ov::test::utils::vec2str(gamma_shape);
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(const ov::PartialShape& input_shape,
                                             const ov::element::Type input_precision,
                                             const ov::Shape& gamma_shape) {
        auto data = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shape);

        auto square = std::make_shared<ov::op::v1::Multiply>(data, data);
        auto mean_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(square, mean_axes, true);

        auto eps = ov::op::v0::Constant::create(input_precision, ov::Shape{}, {1e-5f});
        auto add_eps = std::make_shared<ov::op::v1::Add>(mean, eps);

        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add_eps);
        auto div_const = ov::op::v0::Constant::create(input_precision, ov::Shape{}, {1});
        auto div = std::make_shared<ov::op::v1::Divide>(div_const, sqrt);
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(data, div);

        auto tensor = ov::test::utils::create_and_fill_tensor(input_precision, gamma_shape);
        auto gamma = std::make_shared<ov::op::v0::Constant>(tensor);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(gamma, mul1);

        return std::make_shared<ov::Model>(ov::OutputVector{mul2},
                                           ov::ParameterVector{data},
                                           "RMSNormRankExpandingGamma");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [input_shapes, input_precision, gamma_shape] = GetParam();
        init_input_shapes(input_shapes);
        inType = outType = input_precision;

        function = init_subgraph(inputDynamicShapes.front(), input_precision, gamma_shape);
    }
};

TEST_P(RMSNormRankExpandingGamma, Inference) {
    run();
}

// rank-4 data with a rank-5 all-singleton-leading gamma: pure rank expansion.
const std::vector<std::vector<InputShape>> input_shapes_rank_expanding = {
    {{{1, 2, 2, 8}, {{1, 2, 2, 8}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_RMSNormRankExpandingGamma_all_singleton_prefix,
                         RMSNormRankExpandingGamma,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_rank_expanding),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::Shape{1, 1, 1, 1, 8})),
                         RMSNormRankExpandingGamma::getTestCaseName);

// Same rank expansion, but gamma also varies along the last-but-one axis, so the
// rejection path is hit while the broadcast is still a valid rank expansion.
const std::vector<std::vector<InputShape>> input_shapes_rank_expanding_grouped = {
    {{{1, 2, 2, 8}, {{1, 2, 2, 8}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_RMSNormRankExpandingGamma_group_varying,
                         RMSNormRankExpandingGamma,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_rank_expanding_grouped),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::Shape{1, 1, 1, 2, 8})),
                         RMSNormRankExpandingGamma::getTestCaseName);
}  // namespace
