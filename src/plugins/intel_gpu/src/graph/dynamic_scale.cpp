// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/dynamic_scale.hpp"
#include "dynamic_scale_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(dynamic_scale)

layout dynamic_scale_inst::calc_output_layout(dynamic_scale_node const& node, kernel_impl_params const& impl_param) {
    auto input_node_layout = impl_param.get_non_padded_input_layout();
    auto output_type = impl_param.desc->output_data_types[0].value_or(input_node_layout.data_type);

    if (impl_param.has_fused_primitives())
        output_type = impl_param.get_output_element_type();

    return layout(output_type, input_node_layout.format, input_node_layout.get_tensor());
}

template<typename ShapeType>
std::vector<layout> dynamic_scale_inst::calc_output_layouts(dynamic_scale_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_scale>();
    auto input_node_layout = impl_param.get_non_padded_input_layout();
    auto output_type = desc->output_data_types[0].value_or(input_node_layout.data_type);
    auto scale_type = desc->output_data_types[1].value_or(input_node_layout.data_type);

    if (impl_param.has_fused_primitives())
        output_type = impl_param.get_output_element_type();

    ShapeType per_tensor_shape = input_node_layout.get<ShapeType>();
    for (size_t i = 1; i < per_tensor_shape.size(); i++) {
        per_tensor_shape[i] = 1;
    }

    return {layout(input_node_layout.get<ShapeType>(), output_type, input_node_layout.format),
            layout(per_tensor_shape, scale_type, format::get_default_format(input_node_layout.get_rank()))};
}

std::string dynamic_scale_inst::to_string(dynamic_scale_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

dynamic_scale_inst::typed_primitive_inst(network& network, dynamic_scale_node const& node) : parent(network, node) {}

} // namespace cldnn
