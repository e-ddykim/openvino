// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "intel_gpu/primitives/dynamic_scale.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<dynamic_scale> : public typed_program_node_base<dynamic_scale> {
    using parent = typed_program_node_base<dynamic_scale>;

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
using dynamic_scale_node = typed_program_node<dynamic_scale>;

template <>
class typed_primitive_inst<dynamic_scale> : public typed_primitive_inst_base<dynamic_scale> {
    using parent = typed_primitive_inst_base<dynamic_scale>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(dynamic_scale_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(dynamic_scale_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(dynamic_scale_node const& node);

    typed_primitive_inst(network& network, dynamic_scale_node const& desc);
};

using dynamic_scale_inst = typed_primitive_inst<dynamic_scale>;

}  // namespace cldnn
