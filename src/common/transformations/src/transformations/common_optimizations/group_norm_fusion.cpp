// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/group_norm_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

ov::pass::GroupNormFusion::GroupNormFusion() {
    MATCHER_SCOPE(GroupNormFusion);
    // Replaces a decomposed GroupNormalization sub-graph with a GroupNormalization op.
    // y = scale * MVN(x) + bias
    auto data_m = any_input();
    auto pre_reshape_const_m = wrap_type<ov::op::v0::Constant>();
    auto pre_reshape_m = wrap_type<ov::op::v1::Reshape>({data_m, pre_reshape_const_m});
    auto axes_const_m = wrap_type<ov::op::v0::Constant>();
    auto mvn_m = wrap_type<ov::op::v6::MVN>({pre_reshape_m, axes_const_m});
    auto shapeof_m = wrap_type<ov::op::v3::ShapeOf>({data_m});
    auto post_reshape_m = wrap_type<ov::op::v1::Reshape>({mvn_m, shapeof_m});
    auto scale_input_m = any_input();
    auto convert_scale_input_m = wrap_type<ov::op::v0::Convert>({scale_input_m});
    auto scale_m = std::make_shared<Or>(OutputVector{scale_input_m, convert_scale_input_m});
    auto mul_m = wrap_type<ov::op::v1::Multiply>({post_reshape_m, scale_m});
    auto bias_input_m = any_input();
    auto convert_bias_input_m = wrap_type<ov::op::v0::Convert>({bias_input_m});
    auto bias_m = std::make_shared<Or>(OutputVector{bias_input_m, convert_bias_input_m});
    auto add_m = wrap_type<ov::op::v1::Add>({mul_m, bias_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto data = pattern_map.at(data_m);
        auto data_pshape = data.get_partial_shape();
        // Feature dim should be static.
        if (data_pshape[1].is_dynamic()) {
            return false;
        }
        auto feature_dim = data_pshape[1].get_max_length();

        auto scale = pattern_map.at(scale_input_m);
        {
            // The total number of elements in scale must be equal to feature_dim.
            if (scale.get_partial_shape().is_dynamic()) {
                return false;
            }
            auto scale_shape = scale.get_shape();
            int64_t scale_size = 1;
            for (auto& dim : scale_shape) {
                scale_size *= dim;
            }
            if (scale_size != feature_dim) {
                return false;
            }
        }
        if (pattern_map.count(convert_scale_input_m) != 0) {
            scale = pattern_map.at(convert_scale_input_m);
        }
        auto scale_1d = std::make_shared<ov::op::v0::Squeeze>(scale);
        scale_1d->get_rt_info()["fused_names_0"] = scale.get_any_name();
        auto bias = pattern_map.at(bias_input_m);
        {
            // The total number of elements in bias must be equal to feature_dim.
            if (bias.get_partial_shape().is_dynamic()) {
                return false;
            }
            auto bias_shape = bias.get_shape();
            int64_t bias_size = 1;
            for (auto& dim : bias_shape) {
                bias_size *= dim;
            }
            if (bias_size != feature_dim) {
                return false;
            }
        }
        if (pattern_map.count(convert_bias_input_m) != 0) {
            bias = pattern_map.at(convert_bias_input_m);
        }
        auto bias_1d = std::make_shared<ov::op::v0::Squeeze>(bias);
        bias_1d->get_rt_info()["fused_names_0"] = bias.get_any_name();

        auto pre_reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(pattern_map.at(pre_reshape_m).get_node_shared_ptr());
        auto pre_reshape_pshape = pre_reshape->get_output_partial_shape(0);
        // Feature dim should be static.
        if (pre_reshape_pshape[1].is_dynamic()) {
            return false;
        }
        auto num_groups = pre_reshape_pshape[1].get_max_length();
        // Feature dim should be divided by the number of groups.
        if ((feature_dim % num_groups) != 0) {
            return false;
        }

        auto mvn = std::dynamic_pointer_cast<ov::op::v6::MVN>(pattern_map.at(mvn_m).get_node_shared_ptr());
        // MVNEpsMode should be inside_sqrt.
        if (mvn->get_eps_mode() != ov::op::MVNEpsMode::INSIDE_SQRT) {
            return false;
        }

        auto group_norm = std::make_shared<ov::op::v12::GroupNormalization>(data, scale_1d, bias_1d, num_groups, mvn->get_eps());
        group_norm->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), group_norm);
        ov::replace_node(m.get_match_root(), group_norm);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_m, "GroupNormFusion");
    this->register_matcher(m, callback);
}
