// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"

namespace kernel_selector {

class ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;

    ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad();
    ~ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad() override = default;

    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;

protected:
    bool Validate(const Params& params) const override;
    WeightsLayout GetPreferredWeightsLayout(const convolution_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {FusedOpType::ELTWISE, FusedOpType::QUANTIZE, FusedOpType::ACTIVATION};
    }
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
};

}  // namespace kernel_selector