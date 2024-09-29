// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_norm_quantize_kernel_base.h"
#include <kernel_selector_utils.h>

namespace kernel_selector {

bool GroupNormQuantizeKernelBase::Validate(const Params& params) const {
    const group_norm_quantize_params& orgParams = static_cast<const group_norm_quantize_params&>(params);

    for (auto& fused_op : orgParams.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

JitConstants GroupNormQuantizeKernelBase::GetJitConstants(const group_norm_quantize_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("EPSILON", static_cast<float>(params.epsilon)),
        MakeJitConstant("NUM_GROUPS", params.num_groups),
        MakeJitConstant("IS_QUANTIZED", true),
    });

    return jit;
}

Datatype GroupNormQuantizeKernelBase::GetActivationType(const group_norm_quantize_params& params) const {
    if (params.inputs[0].GetDType() == Datatype::F16)
        return Datatype::F16;
    return Datatype::F32;
}

Datatype GroupNormQuantizeKernelBase::GetAccumulatorType(const group_norm_quantize_params& params) const {
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
