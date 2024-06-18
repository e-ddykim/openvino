// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class MVN : public ov::op::v6::MVN {
public:
    OPENVINO_OP("MVN", "gpu_opset");

    MVN() = default;

    MVN(const Output<Node>& data,
        const Output<Node>& reduction_axes,
        const Output<Node>& shape,
        bool normalize_variance,
        float eps,
        ov::op::MVNEpsMode eps_mode,
        bool zero_flag,
        const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool get_special_zero() const {
        return m_special_zero;
    }
    void set_special_zero(bool special_zero) {
        m_special_zero = special_zero;
    }

protected:
    bool m_special_zero;
    ov::element::Type m_output_type;
};

std::vector<ov::PartialShape> shape_infer(const MVN* op,
                                          const std::vector<ov::PartialShape>& input_shapes);

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
