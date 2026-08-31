// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantize_kernel_scale_shift_2dblk_opt.h"

#include "kernel_selector_utils.h"

namespace kernel_selector {
namespace {

constexpr size_t subgroup_size = 16;
constexpr size_t subgroups_per_work_group = 16;
constexpr size_t work_group_size = subgroup_size * subgroups_per_work_group;
constexpr size_t feature_block_size = 32;
constexpr size_t spatial_tile_size = 8;

bool IsFeatureBroadcast(const DataTensor& tensor, size_t feature_size) {
    return tensor.LogicalSize() == feature_size && tensor.Feature().v == feature_size;
}

}  // namespace

ParamsKey QuantizeKernelScaleShift_2dblk::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::INT8);
    key.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    key.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    key.EnableBatching();
    key.EnableDifferentTypes();
    key.EnableQuantizeScaleShiftOpt();
    return key;
}

CommonDispatchData QuantizeKernelScaleShift_2dblk::SetDefault(const quantize_params& params) const {
    CommonDispatchData dispatch_data;
    const auto& output = params.outputs[0];
    const size_t spatial_size = output.X().v * output.Y().v;
    const size_t spatial_pair_rows = spatial_size / 2;
    const size_t tile_count = 2 * CeilDiv(spatial_pair_rows, spatial_tile_size);

    dispatch_data.gws[0] = work_group_size;
    dispatch_data.gws[1] = CeilDiv(tile_count, subgroups_per_work_group);
    dispatch_data.gws[2] = output.Batch().v * output.Feature().v / feature_block_size;
    dispatch_data.lws[0] = work_group_size;
    dispatch_data.lws[1] = 1;
    dispatch_data.lws[2] = 1;
    return dispatch_data;
}

JitConstants QuantizeKernelScaleShift_2dblk::GetJitConstants(const quantize_params& params, const CommonDispatchData& dispatch_data) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatch_data);
    const auto& output = params.outputs[0];
    const size_t spatial_size = output.X().v * output.Y().v;
    const size_t spatial_pair_rows = spatial_size / 2;

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("SUBGROUPS_PER_WORK_GROUP", subgroups_per_work_group));
    jit.AddConstant(MakeJitConstant("FEATURE_BLOCK_SIZE", feature_block_size));
    jit.AddConstant(MakeJitConstant("SPATIAL_TILE_SIZE", spatial_tile_size));
    jit.AddConstant(MakeJitConstant("SPATIAL_SIZE", spatial_size));
    jit.AddConstant(MakeJitConstant("SPATIAL_PAIR_ROWS", spatial_pair_rows));
    jit.AddConstant(MakeJitConstant("TILE_COUNT", 2 * CeilDiv(spatial_pair_rows, spatial_tile_size)));
    jit.AddConstant(MakeJitConstant("FEATURE_BLOCKS", output.Feature().v / feature_block_size));
    jit.AddConstant(MakeJitConstant("HAS_POST_SCALE", params.has_post_scale));
    jit.AddConstant(MakeJitConstant("HAS_POST_SHIFT", params.has_post_shift));
    jit.AddConstant(MakeJitConstant("HAS_CLAMP", params.has_clamp));
    jit.AddConstant(MakeJitConstant("HAS_MIN_CLAMP", params.has_min_clamp));
    jit.AddConstant(MakeJitConstant("HAS_MAX_CLAMP", params.has_max_clamp));
    jit.AddConstant(MakeJitConstant("HAS_OUTPUT_RANGE_ROUND", false));
    jit.AddConstant(MakeJitConstant("OUT_LO_VAL", params.out_lo));
    jit.AddConstant(MakeJitConstant("OUT_HI_VAL", params.out_hi));
    jit.AddConstant(MakeJitConstant("OUT_SCALE_VAL", params.out_scale));
    jit.AddConstant(MakeJitConstant("OUT_SHIFT_VAL", params.out_shift));
    return jit;
}

bool QuantizeKernelScaleShift_2dblk::Validate(const Params& p) const {
    const auto& params = static_cast<const quantize_params&>(p);
    if (params.inputs.size() != 9)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];
    if (input.GetDType() != Datatype::F16 || output.GetDType() != Datatype::INT8 || input.GetLayout() != DataLayout::b_fs_yx_fsv32 ||
        output.GetLayout() != DataLayout::b_fs_yx_fsv32 || input.Dimentions() != 4 || output.Dimentions() != 4 || input.is_dynamic() || output.is_dynamic())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const size_t feature_size = output.Feature().v;
    const size_t spatial_size = output.X().v * output.Y().v;
    if (input.Batch().v != output.Batch().v || input.Feature().v != feature_size || input.X().v != output.X().v || input.Y().v != output.Y().v ||
        feature_size % feature_block_size != 0 || spatial_size % 2 != 0 || input.GetFirstElementOffset() != 0 || output.GetFirstElementOffset() != 0 ||
        input.PhysicalSize() != input.LogicalSize() || output.PhysicalSize() != output.LogicalSize())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.per_tensor_input_range || !params.per_tensor_output_range || params.per_tensor_input_scale || !params.per_tensor_output_scale ||
        !params.per_tensor_output_shift || !params.has_pre_shift || params.per_tensor_input_shift || params.output_round_to_even ||
        params.out_lo >= params.out_hi)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.inputs[1].GetDType() != Datatype::F16 || params.inputs[3].GetDType() != Datatype::F16 || !IsFeatureBroadcast(params.inputs[5], feature_size) ||
        !IsFeatureBroadcast(params.inputs[6], feature_size))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

KernelsPriority QuantizeKernelScaleShift_2dblk::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

}  // namespace kernel_selector