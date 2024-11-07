// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"
namespace ov {
namespace intel_gpu {
TRANSFORMATIONS_API void mark_as_quantized_input(const std::shared_ptr<Node>& node);
TRANSFORMATIONS_API void unmark_as_quantized_input(const std::shared_ptr<Node>& node);
TRANSFORMATIONS_API bool has_quantized_input(const std::shared_ptr<Node>& node);
bool check_quantized(const std::shared_ptr<Node>& node, int depth);
class TRANSFORMATIONS_API QuantizedInput : public RuntimeAttribute {
public:
    OPENVINO_RTTI("quantized_input", "0");
    bool is_copyable() const override {
        return true;
    }
};
class MarkQuantizedInput : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkQuantizedInput", "0");
    MarkQuantizedInput();
};
}   // namespace intel_gpu
}   // namespace ov
