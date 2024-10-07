// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_scale_convolution.hpp"

#include "intel_gpu/op/dynamic_scale.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

DynamicScaleConvolution::DynamicScaleConvolution(ov::element::Type infer_precision) {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    auto data_m = any_input();
    auto weights_m = wrap_type<ov::op::v0::Constant>(type_matches_any({ov::element::f32}));
    auto convolution_m = wrap_type<ov::op::v1::Convolution>({ data_m, weights_m });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (infer_precision != ov::element::f32)
            return false;

        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(convolution_m));

        auto conv = std::dynamic_pointer_cast<ov::op::v1::Convolution>(pattern_map.at(convolution_m).get_node_shared_ptr());
        if (!conv || transformation_callback(conv))
            return false;

        auto data = pattern_map.at(data_m);

        auto output_type = conv->get_input_element_type(0);
        if (output_type == ov::element::f32) {
            output_type = ov::element::f16;
        } else {
            return false;
        }

        auto conv_weight = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(weights_m).get_node_shared_ptr());
        auto conv_weight_convert = std::make_shared<ov::op::v0::Convert>(conv_weight, ov::element::f16);
        ov::replace_node(conv_weight, conv_weight_convert);

        auto dynamic_scale = std::make_shared<op::DynamicScale>(data, output_type);
        conv->input(0).replace_source_output(dynamic_scale->output(0));
        // dynamic_scale->add_node_control_dependents(data.get_node_shared_ptr());
        // dynamic_scale->add_node_control_dependencies(data.get_node_shared_ptr());
        // data.get_node_shared_ptr()->clear_control_dependents();
        
        auto scale_mul = std::make_shared<ov::op::v1::Multiply>(conv->output(0),
                                                                dynamic_scale->output(1));

        ov::replace_node(conv, scale_mul);
        // scale_mul->input(0).replace_source_output(conv->output(0));
 
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convolution_m, "DynamicScaleConvolution");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
