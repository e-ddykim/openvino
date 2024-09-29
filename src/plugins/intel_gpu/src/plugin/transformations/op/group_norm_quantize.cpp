// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/group_norm_quantize.hpp"
#include "group_normalization_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

GroupNormQuantize::GroupNormQuantize(const Output<Node>& data,
                                     const Output<Node>& scale,
                                     const Output<Node>& bias,
                                     int64_t num_groups,
                                     double epsilon,
                                     const ov::element::Type output_type)
    : ov::op::v12::GroupNormalization({data, scale, bias, num_groups, epsilon}), m_output_type(output_type) {
    set_output_size(2);
    validate_and_infer_types();
}

bool GroupNormQuantize::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void GroupNormQuantize::validate_and_infer_types() {
    const auto output_shapes = shape_infer(dynamic_cast<const ov::op::v12::GroupNormalization*>(this), ov::util::get_node_input_partial_shapes(*this));

    PartialShape per_tensor_shape = output_shapes.at(0);
    for (size_t i = 1; i < per_tensor_shape.size(); i++) {
        per_tensor_shape[i] = 1;
    }

    set_output_type(0, m_output_type, output_shapes.at(0));
    set_output_type(1, get_input_element_type(0), per_tensor_shape);
}

std::shared_ptr<Node> GroupNormQuantize::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<GroupNormQuantize>(new_args.at(0),
                                               new_args.at(1),
                                               new_args.at(2),
                                               get_num_groups(),
                                               get_epsilon(),
                                               m_output_type);
}
}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
