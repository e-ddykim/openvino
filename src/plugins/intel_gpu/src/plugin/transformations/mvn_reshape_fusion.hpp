// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class MVNReshapeFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("i", "0");
    MVNReshapeFusion();
};

}   // namespace intel_gpu
}   // namespace ov
