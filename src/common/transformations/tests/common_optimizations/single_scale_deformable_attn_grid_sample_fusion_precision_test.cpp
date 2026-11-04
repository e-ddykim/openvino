// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "transformations/common_optimizations/multi_scale_deformable_attn_grid_sample_fusion.hpp"

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
namespace v9 = ov::op::v9;

std::shared_ptr<v0::Constant> i64_const(std::vector<int64_t> values) {
    return v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

// Reproducer for the PR #39248 high-impact precision issue.
//
// The graph is a valid single-level deformable-attention graph with:
//   image data:       f16
//   GridSample grid:  f32
//   attention weights: f16
//
// The original graph samples with f32 coordinates. The fusion pass converts the
// coordinate source to the value type (f16) before building the MSDA location.
// For x = 0.5003:
//   original GridSample coordinate c = 2*x - 1 = 0.0006
//   original pixel coordinate  = (c + 1) * W / 2 - 0.5 = 2.0015
//   fused MSDA location       = f16(x) = 0.500488
//   fused pixel coordinate    = f16(x) * W - 0.5 = 2.00244
//
// With image pixel (1, 3) = 100 and other pixels zero, the original and fused
// outputs differ (about 0.14996 vs 0.24414). The pass therefore must not fuse
// this graph unless the coordinate precision is preserved or the mixed types are
// rejected.
std::shared_ptr<ov::Model> getMixedGridValueModel() {
    constexpr int64_t B = 1;
    constexpr int64_t Q = 2;
    constexpr int64_t H = 1;
    constexpr int64_t D = 1;
    constexpr int64_t P = 2;
    constexpr int64_t h = 3;
    constexpr int64_t w = 5;

    auto value = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{B, h * w, H * D});
    auto locations = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{B, Q, H, P, 2});
    auto weights = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{B, Q, H, P});

    auto value_t = std::make_shared<v1::Transpose>(value, i64_const({0, 2, 1}));
    auto value_heads = std::make_shared<v1::Reshape>(value_t, i64_const({B, H, D, h * w}), false);
    auto image = std::make_shared<v1::Reshape>(value_heads, i64_const({B * H, D, h, w}), false);

    auto scaled = std::make_shared<v1::Multiply>(
        locations, v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.f}));
    auto normalized = std::make_shared<v1::Add>(
        scaled, v0::Constant::create(ov::element::f32, ov::Shape{1}, {-1.f}));
    auto identity_slice = std::make_shared<v8::Slice>(
        normalized, i64_const({0}), i64_const({P}), i64_const({1}), i64_const({3}));
    auto coords = std::make_shared<v1::Transpose>(identity_slice, i64_const({0, 2, 1, 3, 4}));
    auto grid_coords = std::make_shared<v1::Reshape>(coords, i64_const({B * H, Q, P, 2}), false);

    v9::GridSample::Attributes attributes{false,
                                          v9::GridSample::InterpolationMode::BILINEAR,
                                          v9::GridSample::PaddingMode::ZEROS};
    auto grid = std::make_shared<v9::GridSample>(image, grid_coords, attributes);

    auto weights_t = std::make_shared<v1::Transpose>(weights, i64_const({0, 2, 1, 3}));
    auto weights_r = std::make_shared<v1::Reshape>(weights_t, i64_const({B * H, 1, Q, P}), false);
    auto weighted = std::make_shared<v1::Multiply>(grid, weights_r);
    auto sum = std::make_shared<v1::ReduceSum>(weighted, i64_const({-1}), false);
    auto output = std::make_shared<v1::Reshape>(sum, i64_const({B, H * D, Q}), false);

    return std::make_shared<ov::Model>(
        ov::OutputVector{output}, ov::ParameterVector{value, locations, weights});
}

}  // namespace

class SingleScaleDeformableAttnGridSampleFusionPrecisionTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::SingleScaleDeformableAttnGridSampleFusion>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
};

TEST_F(SingleScaleDeformableAttnGridSampleFusionPrecisionTest, MixedGridValueTypeMustNotFuse) {
    // model_ref is deliberately left empty: this graph must remain unchanged.
    // If the fusion fires, the f32 GridSample coordinates are converted to f16 and
    // the numerical sampling changes, so the model-vs-clone comparison fails.
    model = getMixedGridValueModel();
}
