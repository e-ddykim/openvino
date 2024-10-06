// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "dynamic_scale_kernel_base.h"

namespace kernel_selector {
class DynamicScaleKernel_b_fs_yx_fsv16 : public DynamicScaleKernelBase {
public:
    using Parent = DynamicScaleKernelBase;

    DynamicScaleKernel_b_fs_yx_fsv16() : DynamicScaleKernelBase{"dynamic_scale_gpu_b_fs_yx_fsv16"} {}
    virtual ~DynamicScaleKernel_b_fs_yx_fsv16() {}

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
    bool Validate(const Params& params) const override;
};

}  // namespace kernel_selector
