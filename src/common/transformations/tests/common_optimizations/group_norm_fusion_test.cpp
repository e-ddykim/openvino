// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/group_norm_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;
using namespace opset12;

namespace {
std::shared_ptr<Model> gen_model_ref(const std::vector<PartialShape>& input_shapes,
                                     element::Type elem_type,
                                     int64_t num_groups,
                                     double eps) {
    const auto data = std::make_shared<Parameter>(elem_type, input_shapes[0]);
    const auto scale = std::make_shared<Parameter>(elem_type, input_shapes[1]);
    const auto bias = std::make_shared<Parameter>(elem_type, input_shapes[2]);

    auto scale_1d = std::make_shared<ov::op::v0::Squeeze>(scale);
    auto bias_1d = std::make_shared<ov::op::v0::Squeeze>(bias);

    const auto group_norm = std::make_shared<GroupNormalization>(data, scale_1d, bias_1d, num_groups, eps);

    return std::make_shared<Model>(OutputVector{group_norm->output(0)}, ParameterVector{data, scale, bias});
}

std::shared_ptr<Model> gen_model(const std::vector<PartialShape>& input_shapes,
                                 element::Type elem_type,
                                 int64_t num_groups,
                                 float eps) {
    const auto data = std::make_shared<Parameter>(elem_type, input_shapes[0]);
    const auto scale = std::make_shared<Parameter>(elem_type, input_shapes[1]);
    const auto bias = std::make_shared<Parameter>(elem_type, input_shapes[2]);

    auto data_rank_size = data->get_partial_shape().rank().get_length();
    int64_t data_reshape_spatial_size = input_shapes[0][1].get_max_length() / num_groups;
    for (int64_t i = 2; i < data_rank_size; ++i) {
        data_reshape_spatial_size *= input_shapes[0][i].get_max_length();
    }
    auto pre_reshape_const = Constant::create(element::i64, Shape{3}, {input_shapes[0][0].get_max_length(), num_groups, data_reshape_spatial_size});
    auto pre_reshape_node = std::make_shared<Reshape>(data, pre_reshape_const, true);

    auto axes_const = Constant::create(element::i64, Shape{1}, {0});
    auto mvn_node = std::make_shared<MVN>(pre_reshape_node, axes_const, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    auto data_shape_node = std::make_shared<ShapeOf>(data, element::i64);
    auto post_reshape_node = std::make_shared<Reshape>(mvn_node, data_shape_node, true);

    auto mul_node = std::make_shared<Multiply>(post_reshape_node, scale);
    auto add_node = std::make_shared<Add>(mul_node, bias);

    return std::make_shared<Model>(OutputVector{add_node->output(0)}, ParameterVector{data, scale, bias});
}

}  // namespace

TEST_F(TransformationTestsF, GroupNormFusionF32) {
    std::vector<PartialShape> input_shapes{PartialShape{1, 12, 6, 8}, PartialShape{1, 12, 1, 1}, PartialShape{1, 12, 1, 1}};
    const int64_t num_groups = 4;
    element::Type elem_type = element::f32;

    model = gen_model(input_shapes, elem_type, num_groups, 1e-3f);
    manager.register_pass<pass::GroupNormFusion>();

    model_ref = gen_model_ref(input_shapes, elem_type, num_groups, 1e-3f);
}
