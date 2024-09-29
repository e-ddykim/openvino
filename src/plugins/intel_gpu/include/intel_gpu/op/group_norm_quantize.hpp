// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/group_normalization.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

/// \brief Operator performing Group Normalization and Dynamic Quantization
class GroupNormQuantize : public ov::op::v12::GroupNormalization {
public:
    OPENVINO_OP("GroupNormQuantize", "gpu_opset");

    GroupNormQuantize() = default;

    /// \brief Constructs an GroupNormQuantize operation.
    ///
    /// \param data Input tensor to be normalized
    /// \param scale Tensor containing scale values for each channel
    /// \param bias Tensor containing bias values for each channel
    /// \param num_groups Number of groups that the channel dimension will be divided into
    /// \param epsilon Value that prevents divisions by zero in GroupNormQuantize formula
    GroupNormQuantize(const Output<Node>& data,
                      const Output<Node>& scale,
                      const Output<Node>& bias,
                      int64_t num_groups,
                      double epsilon,
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
