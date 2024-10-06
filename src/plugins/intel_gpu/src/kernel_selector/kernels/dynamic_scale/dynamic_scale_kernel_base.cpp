// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_scale_kernel_base.h"
#include <kernel_selector_utils.h>

namespace kernel_selector {

bool DynamicScaleKernelBase::Validate(const Params& params) const {
    const dynamic_scale_params& orgParams = static_cast<const dynamic_scale_params&>(params);

    for (auto& fused_op : orgParams.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

JitConstants DynamicScaleKernelBase::GetJitConstants(const dynamic_scale_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    return jit;
}

Datatype DynamicScaleKernelBase::GetActivationType(const dynamic_scale_params& params) const {
    if (params.inputs[0].GetDType() == Datatype::F16)
        return Datatype::F16;
    return Datatype::F32;
}

Datatype DynamicScaleKernelBase::GetAccumulatorType(const dynamic_scale_params& params) const {
    const auto& input_dt = params.inputs[0].GetDType();

    switch (input_dt) {
        case Datatype::INT8:
        case Datatype::UINT8:
            return Datatype::INT32;
        default:
            return Datatype::F32;
    }
}

} // namespace kernel_selector
