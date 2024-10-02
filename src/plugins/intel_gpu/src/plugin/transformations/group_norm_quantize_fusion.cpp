// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_norm_quantize_fusion.hpp"

#include "intel_gpu/op/group_norm_quantize.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

GroupNormQuantizeFusion::GroupNormQuantizeFusion() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    // auto last_dim_static = [](const ov::Output<ov::Node>& output) {
    //     auto out_ps = output.get_node()->get_output_partial_shape(0);
    //     return out_ps.rank().is_static() && out_ps[out_ps.rank().get_length() - 1].is_static() && out_ps.size() <= 5;
    // };

    // Detect VAE decoder basic block pattern
    // GroupNormalization -- Swish -- Convolution
    auto data_m = any_input();
    auto scale_m = any_input();
    auto bias_m = any_input();
    auto group_norm_m = wrap_type<ov::op::v12::GroupNormalization>({data_m, scale_m, bias_m});

    auto swish_m = wrap_type<ov::op::v4::Swish>({group_norm_m});

    auto input_m = any_input();
    auto weights_m = any_input(has_static_dim(0));
    auto convolution_m = wrap_type<ov::op::v1::Convolution>({ swish_m, weights_m });

    // // VariadicSplit(X, axis, split_lengths) = Xw, Xv
    // auto axis_const_m = wrap_type<ov::op::v0::Constant>();
    // auto split_lengths_const_m = wrap_type<ov::op::v0::Constant>();
    // auto variadic_split_m = wrap_type<ov::op::v1::VariadicSplit>({data_m, axis_const_m, split_lengths_const_m});
    // variadic_split_m->set_output_size(2);

    // // Swish(Xw) = Xw * (1.0 + exp(-beta * Xw))
    // auto swish_m = wrap_type<ov::op::v4::Swish>({variadic_split_m->output(0)});
    // auto gelu_m = wrap_type<ov::op::v7::Gelu>({variadic_split_m->output(0)});

    // // Mul(Xw, Xv) = Swish(Xw) * Xv
    // auto glu_m = std::make_shared<Or>(OutputVector{swish_m, gelu_m});
    // auto mul_m = wrap_type<ov::op::v1::Multiply>({glu_m, variadic_split_m->output(1)});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(group_norm_m));
        OPENVINO_ASSERT(pattern_map.count(swish_m));
        OPENVINO_ASSERT(pattern_map.count(convolution_m));

        auto group_norm = std::dynamic_pointer_cast<ov::op::v12::GroupNormalization>(pattern_map.at(group_norm_m).get_node_shared_ptr());
        if (!group_norm)
            return false;

        auto swish = std::dynamic_pointer_cast<ov::op::v4::Swish>(pattern_map.at(swish_m).get_node_shared_ptr());
        if (!swish)
            return false;

        auto conv = std::dynamic_pointer_cast<ov::op::v1::Convolution>(pattern_map.at(convolution_m).get_node_shared_ptr());
        if (!conv || transformation_callback(conv))
            return false;

        // auto isSwiGLU = pattern_map.count(swish_m);
        // auto isGeGLU = pattern_map.count(gelu_m);
        // size_t split_to_glu_idx = 0;
        // ov::intel_gpu::op::SwiGLU::GluType glu_type = ov::intel_gpu::op::SwiGLU::GluType::Swish;

        // if (isSwiGLU) {
        //     auto swish = std::dynamic_pointer_cast<ov::op::v4::Swish>(pattern_map.at(swish_m).get_node_shared_ptr());
        //     glu_type = ov::intel_gpu::op::SwiGLU::GluType::Swish;
        //     split_to_glu_idx = swish->input_value(0).get_index();

        //     size_t split_in_idx = ov::is_type<ov::op::v4::Swish>(mul->get_input_node_shared_ptr(0)) ? 1 : 0;
        //     if (mul->input_value(split_in_idx).get_index() == split_to_glu_idx)
        //         return false;
        // } else if (isGeGLU) {
        //     auto gelu = std::dynamic_pointer_cast<ov::op::v7::Gelu>(pattern_map.at(gelu_m).get_node_shared_ptr());
        //     glu_type = (gelu->get_approximation_mode() == ov::op::GeluApproximationMode::ERF) ? ov::intel_gpu::op::SwiGLU::GluType::Gelu
        //                                                                                       : ov::intel_gpu::op::SwiGLU::GluType::Gelu_Tanh;
        //     split_to_glu_idx = gelu->input_value(0).get_index();

        //     size_t split_in_idx = ov::is_type<ov::op::v7::Gelu>(mul->get_input_node_shared_ptr(0)) ? 1 : 0;
        //     if (mul->input_value(split_in_idx).get_index() == split_to_glu_idx)
        //         return false;
        // } else {
        //     OPENVINO_THROW("'glu_type' not initialized");
        // }

        // auto variadic_split = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(pattern_map.at(variadic_split_m).get_node_shared_ptr());
        // auto variadic_split_in_ps = variadic_split->get_input_partial_shape(0);
        // auto last_dim = variadic_split_in_ps.rank().get_length() - 1;

        // auto axis = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(axis_const_m).get_node_shared_ptr());
        // bool valid_axis_const_values = ov::op::util::has_constant_value<int64_t>(axis, -1) ||
        //                                ov::op::util::has_constant_value<int64_t>(axis, last_dim);
        // if (!valid_axis_const_values)
        //     return false;
        // auto axis_value = axis->cast_vector<int64_t>()[0];

        // auto split_lengths = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(split_lengths_const_m).get_node_shared_ptr());
        // auto split_lengths_value = split_lengths->cast_vector<int64_t>()[0];
        // // Allow only case that exactly splits in half along the last dimension
        // auto split_length = variadic_split_in_ps[last_dim].get_length() / 2;
        // if (split_lengths_value != split_length)
        //     return false;

        auto data = pattern_map.at(data_m);
        auto scale = pattern_map.at(scale_m);
        auto bias = pattern_map.at(bias_m);
        // auto output_type = m.get_match_root()->get_output_element_type(0);
        auto output_type = group_norm->get_input_element_type(0);
        if (output_type == ov::element::f32) {
            output_type = ov::element::f16;
        } else {
            return false;
        }

        auto conv_weight = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(weights_m).get_node_shared_ptr());
        auto conv_weight_convert = std::make_shared<ov::op::v0::Convert>(conv_weight, ov::element::f16);
        ov::replace_node(conv_weight, conv_weight_convert);

        auto group_norm_quant = std::make_shared<op::GroupNormQuantize>(data, scale, bias,
                                                                        group_norm->get_num_groups(),
                                                                        group_norm->get_epsilon(),
                                                                        output_type);
        group_norm_quant->set_friendly_name(group_norm->get_friendly_name());
        ov::copy_runtime_info({group_norm, swish}, group_norm_quant);
        swish->output(0).replace(group_norm_quant->output(0));
        group_norm_quant->add_node_control_dependents(swish);
        group_norm_quant->add_node_control_dependencies(swish);
        swish->clear_control_dependents();
        
        auto scale_mul = std::make_shared<ov::op::v1::Multiply>(group_norm_quant->output(1),
                                                                group_norm_quant->output(1));

        ov::replace_node(conv, scale_mul);
        scale_mul->input(0).replace_source_output(conv->output(0));
 
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convolution_m, "GroupNormQuantizeFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
