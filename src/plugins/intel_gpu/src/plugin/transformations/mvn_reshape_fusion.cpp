// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_reshape_fusion.hpp"

#include "intel_gpu/op/mvn.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

MVNReshapeFusion::MVNReshapeFusion() {
    using namespace ov::pass::pattern;

    auto data_m = any_input();
    auto axes_const_m = wrap_type<ov::op::v0::Constant>();
    auto mvn_m = wrap_type<ov::op::v6::MVN>({data_m, axes_const_m});
    auto shape_m = any_input();
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({mvn_m, shape_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto mvn = std::dynamic_pointer_cast<ov::op::v6::MVN>(pattern_map.at(mvn_m).get_node_shared_ptr());
        auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(pattern_map.at(reshape_m).get_node_shared_ptr());
        auto data = pattern_map.at(data_m);
        auto axes = pattern_map.at(axes_const_m);
        auto shape = pattern_map.at(shape_m);
        auto output_type = m.get_match_root()->get_output_element_type(0);

        auto reshaped_mvn = std::make_shared<op::MVN>(data,
                                                      axes,
                                                      shape,
                                                      mvn->get_normalize_variance(),
                                                      mvn->get_eps(),
                                                      mvn->get_eps_mode(),
                                                      reshape->get_special_zero(),
                                                      output_type);

        reshaped_mvn->set_friendly_name(mvn->get_friendly_name());
        ov::copy_runtime_info(ov::NodeVector{mvn, reshape}, reshaped_mvn);
        ov::replace_node(m.get_match_root(), reshaped_mvn);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_m, "MVNReshapeFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
