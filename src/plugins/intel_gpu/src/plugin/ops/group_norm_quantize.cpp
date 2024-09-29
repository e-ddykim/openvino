// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/group_norm_quantize.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/group_norm_quantize.hpp"
#include "intel_gpu/runtime/layout.hpp"

namespace ov {
namespace op {
namespace internal {
using GroupNormQuantize = ov::intel_gpu::op::GroupNormQuantize;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateGroupNormQuantizeOp(ProgramBuilder& p, const std::shared_ptr<op::GroupNormQuantize>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    cldnn::group_norm_quantize groupNormQuantizePrimitive {
        layerName,
        inputs[0],
        inputs[1],
        inputs[2],
        op->get_num_groups(),
        op->get_epsilon(),
        {cldnn::optional_data_type(op->get_output_type()), cldnn::optional_data_type(op->get_input_element_type(0))}
    };
    p.add_primitive(*op, groupNormQuantizePrimitive);
}

REGISTER_FACTORY_IMPL(internal, GroupNormQuantize);

}  // namespace intel_gpu
}  // namespace ov
