// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

/// \brief Operator performing Dynamic Scaling
class DynamicScale : public ov::op::Op {
public:
    OPENVINO_OP("DynamicScale", "gpu_opset");

    DynamicScale() = default;

    /// \brief Constructs an DynamicScale operation.
    DynamicScale(const Output<Node>& data,
                 const ov::element::Type output_type);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override {
        return false;
    }

    ov::element::Type get_output_type() const { return m_output_type; }

protected:
    ov::element::Type m_output_type;
};
}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
