// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/dynamic_scale.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

DynamicScale::DynamicScale(const Output<Node>& data,
                           const ov::element::Type output_type)
    : Op({data}), m_output_type(output_type) {
    set_output_size(2);
    validate_and_infer_types();
}

bool DynamicScale::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void DynamicScale::validate_and_infer_types() {
    const auto output_shape = get_input_partial_shape(0);

    PartialShape per_tensor_shape = output_shape;
    for (size_t i = 1; i < per_tensor_shape.size(); i++) {
        per_tensor_shape[i] = 1;
    }

    set_output_type(0, m_output_type, output_shape);
    set_output_type(1, get_input_element_type(0), per_tensor_shape);
}

std::shared_ptr<Node> DynamicScale::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<DynamicScale>(new_args.at(0),
                                          m_output_type);
}
}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
