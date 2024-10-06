// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "dynamic_scale_kernel_base.h"

namespace kernel_selector {
class DynamicScaleKernel_bfyx : public DynamicScaleKernelBase {
public:
    using Parent = DynamicScaleKernelBase;

    DynamicScaleKernel_bfyx() : DynamicScaleKernelBase{"dynamic_scale_gpu_bfyx"} {}
    virtual ~DynamicScaleKernel_bfyx() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {};
    }
    MultiDispatchData SetDefault(const dynamic_scale_params& params) const;
    JitConstants GetJitConstants(const dynamic_scale_params& params, DynamicScaleKernelBase::DispatchData dispatchData) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
