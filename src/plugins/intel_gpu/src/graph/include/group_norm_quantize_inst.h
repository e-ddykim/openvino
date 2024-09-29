// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "intel_gpu/primitives/group_norm_quantize.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<group_norm_quantize> : public typed_program_node_base<group_norm_quantize> {
    using parent = typed_program_node_base<group_norm_quantize>;

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
using group_norm_quantize_node = typed_program_node<group_norm_quantize>;

template <>
class typed_primitive_inst<group_norm_quantize> : public typed_primitive_inst_base<group_norm_quantize> {
    using parent = typed_primitive_inst_base<group_norm_quantize>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(group_norm_quantize_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(group_norm_quantize_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(group_norm_quantize_node const& node);

    typed_primitive_inst(network& network, group_norm_quantize_node const& desc);
};

using group_norm_quantize_inst = typed_primitive_inst<group_norm_quantize>;

}  // namespace cldnn
