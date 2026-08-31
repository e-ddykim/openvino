// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "quantize_kernel_base.h"

namespace kernel_selector {

class QuantizeKernelScaleShift_2dblk : public QuantizeKernelBase {
public:
    using Parent = QuantizeKernelBase;

    QuantizeKernelScaleShift_2dblk() : QuantizeKernelBase("quantize_gpu_scale_shift_2dblk_opt") {}
    ~QuantizeKernelScaleShift_2dblk() override = default;

    CommonDispatchData SetDefault(const quantize_params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const quantize_params& params, const CommonDispatchData& dispatchData) const override;
};

}  // namespace kernel_selector