// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_scale_kernel_bfyx.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey DynamicScaleKernel_bfyx::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableBatching();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

DynamicScaleKernelBase::MultiDispatchData DynamicScaleKernel_bfyx::SetDefault(const dynamic_scale_params &params) const {
    MultiDispatchData dispatchData;

    if (!params.has_dynamic_tensors()) {
        const auto& input = params.inputs[0];

        dispatchData.stage_1.gws[0] = input.X().v;
        dispatchData.stage_1.gws[1] = input.Y().v;
        dispatchData.stage_1.gws[2] = input.Z().v * input.Feature().v * input.Batch().v;

        dispatchData.stage_1.lws[0] = input.X().v;
        dispatchData.stage_1.lws[1] = input.Y().v;
        dispatchData.stage_1.lws[2] = input.Z().v;

        if ((input.X().v * input.Y().v * input.Z().v) > params.engineInfo.maxWorkGroupSize) {
            if (input.Z().v > params.engineInfo.maxWorkGroupSize) {
                dispatchData.stage_1.lws[0] = 1;
                dispatchData.stage_1.lws[1] = 1;
                for (size_t lws = 2; lws <= input.Z().v; ++lws) {
                    if (input.Z().v % lws == 0 && (input.Z().v / lws) <= params.engineInfo.maxWorkGroupSize) {
                        dispatchData.stage_1.lws[2] = input.Z().v / lws;
                        dispatchData.stage_1.gws[2] = dispatchData.stage_1.lws[2] * input.Feature().v * input.Batch().v;
                        break;
                    }
                }
            } else {
                if ((input.Y().v * input.Z().v) > params.engineInfo.maxWorkGroupSize) {
                    dispatchData.stage_1.lws[0] = 1;
                    for (size_t lws = 2; lws <= input.Y().v; ++lws) {
                        if (input.Y().v % lws == 0 && (input.Y().v / lws * input.Z().v) <= params.engineInfo.maxWorkGroupSize) {
                            dispatchData.stage_1.lws[1] = input.Y().v / lws;
                            break;
                        }
                    }
                } else {
                    for (size_t lws = 2; lws <= input.X().v; ++lws) {
                        if (input.X().v % lws == 0 && (input.X().v / lws * input.Y().v * input.Z().v) <= params.engineInfo.maxWorkGroupSize) {
                            dispatchData.stage_1.lws[0] = input.X().v / lws;
                            break;
                        }
                    }
                }
            }
        }
        dispatchData.stage_1.gws[0] = dispatchData.stage_1.lws[0];
        dispatchData.stage_1.gws[1] = dispatchData.stage_1.lws[1];

        dispatchData.stage_2.gws[0] = input.Feature().v;
        dispatchData.stage_2.gws[1] = input.Batch().v;
        dispatchData.stage_2.gws[2] = 1;

        dispatchData.stage_2.lws[0] = input.Feature().v;
        dispatchData.stage_2.lws[1] = 1;
        dispatchData.stage_2.lws[2] = 1;

        size_t divisor = 2;
        while (dispatchData.stage_2.lws[0] > params.engineInfo.maxWorkGroupSize) {
            if (input.Feature().v % divisor == 0) {
                dispatchData.stage_2.lws[0] = input.Feature().v / divisor;
            }
            divisor += 1;
        }

        dispatchData.stage_final.gws[0] = input.X().v * input.Y().v * input.Z().v;
        dispatchData.stage_final.gws[1] = input.Feature().v * input.Batch().v;
        dispatchData.stage_final.gws[2] = 1;

        dispatchData.stage_final.lws[0] = input.X().v * input.Y().v * input.Z().v;
        dispatchData.stage_final.lws[1] = input.Feature().v * input.Batch().v;
        dispatchData.stage_final.lws[2] = 1;

        divisor = 2;
        while (dispatchData.stage_final.lws[0] > params.engineInfo.maxWorkGroupSize) {
            if (dispatchData.stage_final.gws[0] % divisor == 0) {
                dispatchData.stage_final.lws[0] = dispatchData.stage_final.gws[0] / divisor;
            }
            divisor += 1;
        }

        divisor = 2;
        while ((dispatchData.stage_final.lws[0] * dispatchData.stage_final.lws[1]) > params.engineInfo.maxWorkGroupSize) {
            if (dispatchData.stage_final.gws[1] % divisor == 0) {
                dispatchData.stage_final.lws[1] = dispatchData.stage_final.gws[1] / divisor;
            }
            divisor += 1;
        }
    }

    return dispatchData;
}

JitConstants DynamicScaleKernel_bfyx::GetJitConstants(const dynamic_scale_params &params,
                                                           DynamicScaleKernelBase::DispatchData dispatchData) const {
    auto jit = DynamicScaleKernelBase::GetJitConstants(params);

    if (params.has_dynamic_tensors()) {
        jit.AddConstants({
            MakeJitConstant("GWS0", "get_global_size(0)"),
            MakeJitConstant("LWS0", "get_local_size(0)"),
            MakeJitConstant("LWS1", "get_local_size(1)"),
            MakeJitConstant("LWS2", "get_local_size(2)"),
        });
    } else {
        jit.AddConstants({
            MakeJitConstant("GWS0", dispatchData.gws[0]),
            MakeJitConstant("LWS0", dispatchData.lws[0]),
            MakeJitConstant("LWS1", dispatchData.lws[1]),
            MakeJitConstant("LWS2", dispatchData.lws[2]),
        });
    }
    auto activation_dt = GetActivationType(params);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    return jit;
}

void DynamicScaleKernel_bfyx::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const dynamic_scale_params&>(params);
        auto dispatchData = SetDefault(prim_params);

        kd.kernels[0].params.workGroups.global = dispatchData.stage_1.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.stage_1.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params, 0);

        kd.kernels[1].params.workGroups.global = dispatchData.stage_2.gws;
        kd.kernels[1].params.workGroups.local = dispatchData.stage_2.lws;
        kd.kernels[1].skip_execution = KernelData::SkipKernelExecution(prim_params, 1);

        kd.kernels[2].params.workGroups.global = dispatchData.stage_final.gws;
        kd.kernels[2].params.workGroups.local = dispatchData.stage_final.lws;
        kd.kernels[2].skip_execution = KernelData::SkipKernelExecution(prim_params, 2);

        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(prim_params.outputs[0].Batch().v * prim_params.outputs[0].Feature().v * 4);
    };
}

KernelsData DynamicScaleKernel_bfyx::GetKernelsData(const Params &params) const {
    assert(params.GetType() == KernelType::DYNAMIC_SCALE);

    if (!Validate(params))
        return {};

    const dynamic_scale_params& prim_params = static_cast<const dynamic_scale_params&>(params);

    MultiDispatchData dispatchData = SetDefault(prim_params);

    KernelData kd = KernelData::Default<dynamic_scale_params>(params, 3);
    kd.internalBufferDataType = GetAccumulatorType(prim_params);
    GetUpdateDispatchDataFunc(kd);

    auto finalKernelName = GetKernelName(prim_params);
    size_t entry_part_id = 0;

    if (!prim_params.has_dynamic_tensors()) {
        kd.internalBufferSizes.push_back(prim_params.outputs[0].Batch().v * prim_params.outputs[0].Feature().v * 4);
    }

    {
        // Max per feature stage
        auto cldnn_jit = GetJitConstants(prim_params, dispatchData.stage_1);
        cldnn_jit.AddConstant(MakeJitConstant("DYNAMIC_SCALE_KERNEL_FEATURE_MAX", 1));
        auto entry_point = GetEntryPoint(finalKernelName, prim_params.layerID, params, entry_part_id++);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel,
                        dispatchData.stage_1,
                        params.engineInfo,
                        finalKernelName,
                        jit,
                        entry_point,
                        "",
                        false,
                        false,
                        1,
                        0,
                        0,
                        prim_params.is_shape_agnostic);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    }
    {
        // Max per batch stage
        auto cldnn_jit = GetJitConstants(prim_params, dispatchData.stage_2);
        cldnn_jit.AddConstant(MakeJitConstant("DYNAMIC_SCALE_KERNEL_BATCH_MAX", 1));
        auto entry_point = GetEntryPoint(finalKernelName, prim_params.layerID, params, entry_part_id++);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[1];
        FillCLKernelData(kernel,
                        dispatchData.stage_2,
                        params.engineInfo,
                        finalKernelName,
                        jit,
                        entry_point,
                        "",
                        false,
                        false,
                        0,
                        0,
                        2);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    }
    {
        // final stage
        auto cldnn_jit = GetJitConstants(prim_params, dispatchData.stage_final);
        cldnn_jit.AddConstant(MakeJitConstant("DYNAMIC_SCALE_KERNEL_FINAL", 1));
        auto entry_point = GetEntryPoint(finalKernelName, prim_params.layerID, params, entry_part_id++);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[2];
        FillCLKernelData(kernel,
                        dispatchData.stage_final,
                        params.engineInfo,
                        finalKernelName,
                        jit,
                        entry_point,
                        "",
                        false,
                        false,
                        1,
                        0,
                        2,
                        prim_params.is_shape_agnostic);
    }

    return {kd};
}

KernelsPriority DynamicScaleKernel_bfyx::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_7;
}
} // namespace kernel_selector
