// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "group_norm_quantize_kernel_base.h"

namespace kernel_selector {
class GroupNormQuantizeKernelBfyx : public GroupNormQuantizeKernelBase {
public:
    using Parent = GroupNormQuantizeKernelBase;

    GroupNormQuantizeKernelBfyx() : GroupNormQuantizeKernelBase{"group_norm_quantize_gpu_bfyx_opt"} {}
    virtual ~GroupNormQuantizeKernelBfyx() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE
        };
    }
    MultiDispatchData SetDefault(const group_norm_quantize_params& params) const;
    JitConstants GetJitConstants(const group_norm_quantize_params& params, GroupNormQuantizeKernelBase::DispatchData dispatchData) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
