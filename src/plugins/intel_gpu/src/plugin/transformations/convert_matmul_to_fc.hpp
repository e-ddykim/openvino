// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class MoveAddBeforeVariadicSplit : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveAddBeforeVariadicSplit");
    MoveAddBeforeVariadicSplit();

    static bool can_be_transformed(const std::shared_ptr<const ov::Node>& node);
};

class ConvertMatMulToFullyConnected: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertMatMulToFullyConnected");
    ConvertMatMulToFullyConnected(bool supports_immad = false);
};

}   // namespace ov::intel_gpu
