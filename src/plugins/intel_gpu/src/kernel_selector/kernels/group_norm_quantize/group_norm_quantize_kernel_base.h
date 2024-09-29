// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GroupNormQuantizeParams
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct group_norm_quantize_params : public base_params {
    group_norm_quantize_params() : base_params(KernelType::GROUP_NORM_QUANTIZE) {}

    std::int64_t num_groups = 1;
    double epsilon = 0.0f;

    ParamsKey GetParamsKey() const override {
        return base_params::GetParamsKey();
    }
};

class GroupNormQuantizeKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~GroupNormQuantizeKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        size_t itemsNum;
        size_t leftovers;
        size_t dataSetsCount;
        size_t dataSetSize;
        size_t maxSlmSize;

        DispatchData() : itemsNum(0), leftovers(0), dataSetsCount(0), dataSetSize(0), maxSlmSize(0) {}
    };

    struct MultiDispatchData {
        DispatchData stage_1;
        DispatchData stage_2;
        DispatchData stage_max;
        DispatchData stage_final;

        size_t item_groups;

        MultiDispatchData() : item_groups(0) {}
    };

protected:
    bool Validate(const Params&) const override;
    JitConstants GetJitConstants(const group_norm_quantize_params& params) const;
    std::string GetKernelName(const group_norm_quantize_params&) const { return kernelName; }
    Datatype GetActivationType(const group_norm_quantize_params& params) const;
    Datatype GetAccumulatorType(const group_norm_quantize_params& params) const;
};

}  // namespace kernel_selector
