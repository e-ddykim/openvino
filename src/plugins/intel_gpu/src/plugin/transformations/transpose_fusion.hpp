// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class TransposeFusion: public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("TransposeFusion", "0");
    TransposeFusion();
};

class TransposeMatMulMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeMatMulMatcher", "0");
    TransposeMatMulMatcher();
};

class TransposeMatMulTransposeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeMatMulTransposeMatcher", "0");
    TransposeMatMulTransposeMatcher();
};

class TransposeSDPAMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeSDPAMatcher", "0");
    TransposeSDPAMatcher();
};

class TransposeTransposedMatMulMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeTransposedMatMulMatcher", "0");
    TransposeTransposedMatMulMatcher();
};

}   // namespace intel_gpu
}   // namespace ov
