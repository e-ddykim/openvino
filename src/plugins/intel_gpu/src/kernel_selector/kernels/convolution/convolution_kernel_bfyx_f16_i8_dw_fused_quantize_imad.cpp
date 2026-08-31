// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_bfyx_f16_i8_dw_fused_quantize_imad.h"

#include <cmath>

namespace kernel_selector {
namespace {
constexpr size_t subgroup_size = 16;
// Each lane owns channels_per_lane channels, subgroup_size apart, so that a single block write of
// that width covers channel_block contiguous output channels.
constexpr size_t channels_per_lane = 2;
constexpr size_t channel_block = subgroup_size * channels_per_lane;
// Subgroups no longer cooperate on one tile - each one takes its own y tile, so this only sets the
// work-group granularity. Measured to be nearly irrelevant: 2 and 4 differ by under 1%.
constexpr size_t subgroups_per_work_group = 4;
constexpr size_t tile_x = 12;
// The row cache is a three-slot sliding window, so tile_y costs no registers. It trades halo
// re-fetch, (tile_y + 2) / tile_y, against the number of threads launched, and the threads win:
// tile_y 8 cuts input traffic 16% but measured 8% slower, so this kernel is latency-bound rather
// than bandwidth-bound and wants many short threads. 4 is the best measured value.
constexpr size_t tile_y = 4;
}  // namespace

ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad::ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad()
    : ConvolutionKernelBase("convolution_gpu_bfyx_f16_i8_dw_fused_quantize_imad") {}

ParamsKey ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::F16);
    key.EnableInputWeightsType(WeightsType::INT8);
    // Channels-innermost on both sides: byxf(B, C, H, W) in, bfyx(B, H, W, C) out. Those are the
    // same byte order, which is what makes the absorbed output transpose free.
    key.EnableInputLayout(DataLayout::byxf);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableDifferentTypes();
    key.EnableDifferentInputWeightsTypes();
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBiasPerFeature();
    key.EnableNonBiasTerm();
    key.EnableBatching();
    key.EnableGroupedConvolution();
    key.EnableFusedInputQuantization();
    key.EnableFusedOutputTranspose();
    return key;
}

DeviceFeaturesKey ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad::get_required_device_features_key(const Params& params) const {
    auto key = get_common_subgroups_device_features_key(params);
    key.requires_blocked_read_write_short();
    return key;
}

bool ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad::Validate(const Params& params) const {
    if (!Parent::Validate(params))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    const auto& convolution_params = static_cast<const kernel_selector::convolution_params&>(params);
    if (!convolution_params.fused_input_quantization || !convolution_params.fused_output_transpose || convolution_params.engineInfo.arch < gpu_arch::xe2 ||
        !convolution_params.engineInfo.supports_imad || convolution_params.inputs.size() != 3 || convolution_params.inputs[0].GetDType() != Datatype::F16 ||
        convolution_params.inputs[0].GetLayout() != DataLayout::byxf || convolution_params.outputs[0].GetDType() != Datatype::F16 ||
        convolution_params.outputs[0].GetLayout() != DataLayout::bfyx || convolution_params.weights.GetDType() != WeightsType::INT8 ||
        convolution_params.quantization != QuantizationType::NONE || convolution_params.filterSize.x != 3 || convolution_params.filterSize.y != 3 ||
        convolution_params.stride.x != 1 || convolution_params.stride.y != 1 || convolution_params.dilation.x != 1 || convolution_params.dilation.y != 1 ||
        convolution_params.padding_begin.x != 1 || convolution_params.padding_begin.y != 1 || convolution_params.padding_end.x != 1 ||
        convolution_params.padding_end.y != 1 || !convolution_params.weights_zero_points.empty() || !convolution_params.activations_zero_points.empty() ||
        !convolution_params.compensation.empty() || !std::isfinite(convolution_params.input_quantization_output_shift)) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const auto& input = convolution_params.inputs[0];
    const auto& scale = convolution_params.inputs[1];
    const auto& shift = convolution_params.inputs[2];
    const auto& output = convolution_params.outputs[0];
    if (input.Batch().is_dynamic || input.Feature().is_dynamic || input.Y().is_dynamic || input.X().is_dynamic || output.Batch().is_dynamic ||
        output.Feature().is_dynamic || output.Y().is_dynamic || output.X().is_dynamic || input.Batch().v == 0 || input.Feature().v == 0 ||
        input.Y().v == 0 || input.X().v == 0 || input.Feature().v % channel_block != 0 ||
        convolution_params.groups != input.Feature().v || convolution_params.weights.G().v != input.Feature().v || convolution_params.weights.OFM().v != 1 ||
        convolution_params.weights.IFM().v != 1 || output.Batch().v != input.Batch().v || output.Feature().v != input.Y().v || output.Y().v != input.X().v ||
        output.X().v != input.Feature().v || scale.GetDType() != Datatype::F32 || shift.GetDType() != Datatype::F32 || scale.Batch().v != 1 ||
        scale.Feature().v != input.Feature().v || scale.Y().v != 1 || scale.X().v != 1 || shift.Batch().v != 1 || shift.Feature().v != input.Feature().v ||
        shift.Y().v != 1 || shift.X().v != 1 || input.GetFirstElementOffset() != 0 || output.GetFirstElementOffset() != 0 ||
        input.PitchesDifferFromLogicalDims() || output.PitchesDifferFromLogicalDims()) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    return true;
}

WeightsLayout ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad::GetPreferredWeightsLayout(const convolution_params&) const {
    return WeightsLayout::gs_oi_yxs_gsv16_yxsv4;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad::SetDefault(const convolution_params& params, int) const {
    DispatchData dispatch_data;
    const auto& input = params.inputs[0];
    // Dimension 1 indexes y tiles directly - one per subgroup - so it is padded up to the local size
    // and the kernel drops the padded tiles.
    dispatch_data.gws = {CeilDiv(input.X().v, tile_x) * subgroup_size,
                         Align(CeilDiv(input.Y().v, tile_y), subgroups_per_work_group),
                         input.Batch().v * input.Feature().v / channel_block};
    dispatch_data.lws = {subgroup_size, subgroups_per_work_group, 1};
    return dispatch_data;
}

JitConstants ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad::GetJitConstants(const convolution_params& params, const DispatchData& dispatch_data) const {
    auto jit = Parent::GetJitConstants(params, dispatch_data);
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("SUBGROUPS_PER_WORK_GROUP", subgroups_per_work_group));
    jit.AddConstant(MakeJitConstant("CHANNELS_PER_LANE", channels_per_lane));
    jit.AddConstant(MakeJitConstant("CHANNEL_BLOCK", channel_block));
    jit.AddConstant(MakeJitConstant("TILE_X", tile_x));
    jit.AddConstant(MakeJitConstant("TILE_Y", tile_y));
    jit.AddConstant(MakeJitConstant("INPUT_QUANTIZATION_OUTPUT_SHIFT", params.input_quantization_output_shift));
    jit.AddConstant(MakeJitConstant("INPUT_QUANTIZATION_USE_FP16_ARITHMETIC", params.input_quantization_use_fp16_arithmetic));

    if (!params.fused_ops.empty()) {
        auto config = FusedOpsConfiguration("",
                                            {"b", "fused_ops_f", "fused_ops_y", "fused_ops_x"},
                                            "fused_ops_in",
                                            GetActivationType(params),
                                            1,
                                            LoadType::LT_UNALIGNED,
                                            BoundaryCheck::ENABLED,
                                            IndexType::TENSOR_COORD,
                                            Tensor::DataChannelName::FEATURE);
        jit.Merge(MakeFusedOpsJitConstants(params, {config}));
    }

    return jit;
}

KernelsData ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad::GetKernelsData(const Params& params) const {
    auto kernels_data = GetCommonKernelsData(params);
    if (kernels_data.empty())
        return {};

    auto& arguments = kernels_data[0].kernels[0].params.arguments;
    arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
    return kernels_data;
}

KernelsPriority ConvolutionKernel_bfyx_f16_i8_dw_fused_quantize_imad::GetKernelsPriority(const Params&) const {
    return FORCE_PRIORITY_1;
}

}  // namespace kernel_selector