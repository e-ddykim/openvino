// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/dynamic_scale.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/dynamic_scale.hpp"
#include "intel_gpu/runtime/layout.hpp"

namespace ov {
namespace op {
namespace internal {
using DynamicScale = ov::intel_gpu::op::DynamicScale;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateDynamicScaleOp(ProgramBuilder& p, const std::shared_ptr<op::DynamicScale>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    cldnn::dynamic_scale dynamicScalePrimitive {
        layerName,
        inputs[0],
        {cldnn::optional_data_type(op->get_output_type()), cldnn::optional_data_type(op->get_input_element_type(0))}
    };
    p.add_primitive(*op, dynamicScalePrimitive);
}

REGISTER_FACTORY_IMPL(internal, DynamicScale);

}  // namespace intel_gpu
}  // namespace ov
