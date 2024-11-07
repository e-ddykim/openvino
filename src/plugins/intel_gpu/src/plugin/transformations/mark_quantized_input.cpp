// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "mark_quantized_input.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
namespace ov {
namespace intel_gpu {
void mark_as_quantized_input(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[QuantizedInput::get_type_info_static()] = QuantizedInput();
}
void unmark_as_quantized_input(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(QuantizedInput::get_type_info_static());
}
bool has_quantized_input(const std::shared_ptr<Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(QuantizedInput::get_type_info_static());
}
bool check_quantized(const std::shared_ptr<Node>& node, int depth) {
    if (depth > 10)
        return false;
    if (ov::is_type<ov::op::v0::MatMul>(node) ||
        ov::is_type<ov::op::v1::Convolution>(node)) {
        mark_as_quantized_input(node);
        return true;
    } else {
        bool marked = false;
        for (auto& child : node->get_output_target_inputs(0)) {
            marked |= check_quantized(child.get_node()->shared_from_this(), (depth + 1));
        }
        return marked;
    }
}
MarkQuantizedInput::MarkQuantizedInput() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;
    auto fakequantize_m = wrap_type<ov::op::v0::FakeQuantize>({any_input(), any_input(), any_input(), any_input(), any_input()});
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(fakequantize_m));
        auto fakequantize = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(pattern_map.at(fakequantize_m).get_node_shared_ptr());
        if (fakequantize->get_friendly_name().compare("Transpose_14096") == 0)
            std::cout << "!" << std::endl;
        if (transformation_callback(fakequantize))
            return false;
        return check_quantized(fakequantize, 0);
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(fakequantize_m, "MarkQuantizedInput");
    this->register_matcher(m, callback);
}
}  // namespace intel_gpu
}  // namespace ov
