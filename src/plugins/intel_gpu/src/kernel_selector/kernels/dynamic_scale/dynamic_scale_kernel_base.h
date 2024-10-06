// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DynamicScaleParams
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct dynamic_scale_params : public base_params {
    dynamic_scale_params() : base_params(KernelType::DYNAMIC_SCALE) {}

    ParamsKey GetParamsKey() const override {
        return base_params::GetParamsKey();
    }
};

class DynamicScaleKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~DynamicScaleKernelBase() {}

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
        DispatchData stage_final;

        MultiDispatchData() {}
    };

protected:
    bool Validate(const Params&) const override;
    JitConstants GetJitConstants(const dynamic_scale_params& params) const;
    std::string GetKernelName(const dynamic_scale_params&) const { return kernelName; }
    Datatype GetActivationType(const dynamic_scale_params& params) const;
    Datatype GetAccumulatorType(const dynamic_scale_params& params) const;
};

}  // namespace kernel_selector
