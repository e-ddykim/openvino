// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GroupNormFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief GroupNormFusion transformation replaces a sub-graph
 * (reshape - MVN - reshape - Multiply - Add) with a GroupNormalization op.
 */
class ov::pass::GroupNormFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GroupNormFusion", "0");
    GroupNormFusion();
};
