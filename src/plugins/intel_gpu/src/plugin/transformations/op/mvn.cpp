// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/mvn.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "matmul_shape_inference.hpp"
#include "broadcast_shape_inference.hpp"
#include "reshape_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

MVN::MVN(const Output<Node>& data,
         const Output<Node>& reduction_axes,
         const Output<Node>& shape,
         bool normalize_variance,
         float eps,
         ov::op::MVNEpsMode eps_mode,
         bool zero_flag,
         const ov::element::Type output_type)
    : ov::op::v6::MVN()
    , m_special_zero(zero_flag)
    , m_output_type(output_type) {
    set_arguments({data, reduction_axes, shape});
    set_normalize_variance(normalize_variance);
    set_eps(eps);
    set_eps_mode(eps_mode);
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> MVN::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    
    return std::make_shared<MVN>(new_args.at(0), new_args.at(1), new_args.at(2), get_normalize_variance(),
                                 get_eps(), get_eps_mode(), m_special_zero, m_output_type);
}

void MVN::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 3,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected 3.");

    std::vector<ov::PartialShape> input_shapes;
    input_shapes.emplace_back(get_input_partial_shape(0));
    input_shapes.emplace_back(get_input_partial_shape(2));

    auto out_shapes = shape_infer(this, input_shapes);

    auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out_shapes[0]);
}

bool MVN::visit_attributes(ov::AttributeVisitor &visitor) {
    ov::op::v6::MVN::visit_attributes(visitor);
    visitor.on_attribute("special_zero", m_special_zero);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

std::vector<ov::PartialShape> shape_infer(const MVN* op,
                                          const std::vector<ov::PartialShape>& input_shapes) {

    auto reshape_node = std::make_shared<ov::op::v1::Reshape>(op->get_input_node_shared_ptr(0), op->get_input_node_shared_ptr(2), op->get_special_zero());

    return ov::op::v1::shape_infer(reshape_node.get(), input_shapes);
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
