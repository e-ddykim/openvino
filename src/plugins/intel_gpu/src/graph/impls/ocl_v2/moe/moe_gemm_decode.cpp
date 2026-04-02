// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_gemm_decode.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "moe_gemm_inst.h"
#include "../utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

static size_t get_subgroup_size(gpu_arch arch) {
    // return arch >= gpu_arch::xe2 ? 32 : 16;
    return 16;
}

// Performance tuning parameters
#define K_BLOCK      12
#define N_BLOCK      32
// #define SUBGROUP_NUM 16

JitConstants MoEGemmDecodeGenerator::get_jit_constants(const RuntimeParams& params) const {
    auto jit = make_base_jit_constants(params);

    const auto& desc = params.typed_desc<moe_gemm>();
    auto& engine = params.prog->get_engine();
    const auto& info = engine.get_device_info();
    auto gate_up_group_size = desc->moe_config.group_size;
    auto down_group_size = desc->moe_config.group_size;
    if (desc->moe_config.group_size == std::numeric_limits<size_t>::max()) {
        gate_up_group_size = desc->moe_config.hidden_size;
        down_group_size = desc->moe_config.inter_size;
    }

    jit.make("MAX_TOPK", desc->moe_config.top_k);
    jit.make("EXPERT_NUM", desc->moe_config.num_expert);
    jit.make("HIDDEN_SIZE", desc->moe_config.hidden_size);
    jit.make("INTERMEDIATE_SIZE", desc->moe_config.inter_size);
    jit.make("N_BLOCK", N_BLOCK);
    jit.make("K_BLOCK", K_BLOCK);
    jit.make("SUBGROUP_SIZE", get_subgroup_size(info.arch));
    // jit.make("SUBGROUP_NUM", SUBGROUP_NUM);
    jit.make("GATE_UP_GROUP_SIZE", gate_up_group_size);
    jit.make("DOWN_GROUP_SIZE", down_group_size);
    jit.make("MOE_DTYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
    jit.make("MOE_DTYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);

    ov::element::Type weight_dt = params.get_input_layout(static_cast<size_t>(moe_gemm::MoEGemmInputIdx::WEIGHT)).data_type;
    // auto scale_dt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SCALE_0)).data_type;
    // auto zp_dt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::ZP_0)).data_type;
    if (weight_dt == ov::element::u4 || weight_dt == ov::element::i4) {
        jit.make("WEIGHT_COMPRESSEION_DT", 0);
        jit.make("MOE_WEI_DT", "uchar");
        jit.make("MOE_SCALE_DT", "half");
        jit.make("MOE_ZP_DT", "uchar");
        if (weight_dt == ov::element::i4)
            jit.make("IS_WEIGHT_SIGNED", 1);
    } else if (weight_dt == ov::element::u8 || weight_dt == ov::element::i8) {
        jit.make("WEIGHT_COMPRESSEION_DT", 1);
        jit.make("MOE_WEI_DT", "uchar");
        jit.make("MOE_SCALE_DT", "half");
        jit.make("MOE_ZP_DT", "uchar");
    } else if (weight_dt == ov::element::f16) {
        jit.make("WEIGHT_COMPRESSEION_DT", 2);
        jit.make("MOE_WEI_DT", "half");
        jit.make("MOE_SCALE_DT", "half");  // not use
        jit.make("MOE_ZP_DT", "half");     // not use
    }

    jit.make("IS_UP_PHASE", desc->moe_phase == moe_gemm::MoEGemmPhase::UP ? 1 : 0);

    auto moe_cfg = get_moe_cfg(params);
    std::vector<moe_gemm::MoEGemmInputIdx> input_ids = {moe_gemm::MoEGemmInputIdx::INPUT,
                                                        moe_gemm::MoEGemmInputIdx::WEIGHT,
                                                        moe_gemm::MoEGemmInputIdx::EXPERTS_IDS,
                                                        moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT,
                                                        moe_gemm::MoEGemmInputIdx::INPUT_TOKENS_LENS};
    bool has_bias = moe_cfg.has_bias;
    bool is_u4_i4 = (params.input_layouts[1].data_type == data_types::u4 || params.input_layouts[1].data_type == data_types::i4);
    auto weight_idx = moe_gemm::MoEGemmInputIdx::WEIGHT;
    auto bias_idx = moe_gemm::MoEGemmInputIdx::BIAS;
    auto scale_idx = moe_cfg.weight_scale_idx;
    auto zp_idx = moe_cfg.weight_zp_idx;
    const auto& weight_shape = params.input_layouts[weight_idx].get_shape();
    if (moe_cfg.is_weight_quantized) {
        const auto& scale_shape = params.input_layouts[scale_idx].get_shape();
        const auto& bias_shape = params.input_layouts[bias_idx].get_shape();
        if (has_bias) {
            jit.make("BIAS_DT", to_ocl_type(data_types::f16));
            jit.make("BIAS_STRIDE", bias_shape[1] * bias_shape[2]);
        }
        input_ids.push_back((moe_gemm::MoEGemmInputIdx)(static_cast<int32_t>(scale_idx)));
        if (!moe_cfg.is_weight_symmetric_quantized)
            input_ids.push_back((moe_gemm::MoEGemmInputIdx)(static_cast<int32_t>(zp_idx)));

        jit.make("WEIGHT_SCALE_DT", to_ocl_type(data_types::f16));
        if (moe_cfg.weight_group_size > 0)
            jit.make("NUM_GROUPS", scale_shape[2]);
        else
            jit.make("NUM_GROUPS", 1);
        size_t expert_stride = weight_shape.size() == 4 ? (weight_shape[1] * weight_shape[2] * weight_shape[3]) : (weight_shape[1] * weight_shape[2]);
        if (is_u4_i4) {
            jit.make("EXPERT_STRIDE", expert_stride / 2);
            jit.make("WEIGHT_COMPRESSED_INT4", 1);
        } else {
            jit.make("EXPERT_STRIDE", expert_stride);
        }
        if (!moe_cfg.is_weight_symmetric_quantized)
            jit.make("WEIGHT_ZP_DT", to_ocl_type(data_types::f16));
    } else {
        jit.make("EXPERT_STRIDE", params.input_layouts[1].get_shape()[1] * params.input_layouts[1].get_shape()[2]);
    }

    const auto& in_offsets_map = params.in_port_to_shape_info_offset;
    const auto& out_offsets_map = params.out_port_to_shape_info_offset;
    for (size_t i = 0; i < input_ids.size(); i++) {
        const size_t tensor_id = input_ids[i];
        jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
    }
    jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));
    jit.make("INPUT_STRIDE",
             params.input_layouts[1].get_shape().size() == 4 ? params.input_layouts[1].get_shape()[2] * params.input_layouts[1].get_shape()[3]
                                                             : params.input_layouts[1].get_shape()[2]);
    jit.make("OUTPUT_STRIDE", params.input_layouts[1].get_shape()[1]);

    if (moe_cfg.has_batch_dim) {
        jit.make("INPUT_SEQ_LEN", "INPUT0_FEATURE_NUM");
    } else {
        jit.make("INPUT_SEQ_LEN", "INPUT0_BATCH_NUM");
    }
    return jit;
}

DispatchDataFunc MoEGemmDecodeGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());

        // auto* rtp = static_cast<MoEGemmRuntimeParams*>(rt_params);
        const auto& desc = params.typed_desc<moe_gemm>();
        const auto& device_info = params.get_device_info();

        size_t subgroup_size = get_subgroup_size(device_info.arch);
        size_t N = desc->moe_phase == moe_gemm::MoEGemmPhase::UP ? desc->moe_config.inter_size : desc->moe_config.hidden_size;
        size_t K = desc->moe_config.hidden_size;

        auto& wgs = kd.params.workGroups;
        // wgs.global = {desc->moe_config.top_k, subgroup_size, static_cast<size_t>(N / N_BLOCK)};
        // wgs.local = {1, subgroup_size, SUBGROUP_NUM};
        wgs.global = {(N / N_BLOCK * subgroup_size), (K / subgroup_size / K_BLOCK), desc->moe_config.top_k};
        wgs.local = {subgroup_size, (K /subgroup_size / K_BLOCK), 1};
    }};
}

Arguments MoEGemmDecodeGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    auto cfg = get_moe_cfg(params);
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    args.push_back({ArgumentDescriptor::Types::INPUT, moe_gemm::MoEGemmInputIdx::INPUT});
    args.push_back({ArgumentDescriptor::Types::INPUT, moe_gemm::MoEGemmInputIdx::WEIGHT});
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    args.push_back({ArgumentDescriptor::Types::INPUT, moe_gemm::MoEGemmInputIdx::EXPERTS_IDS});

    if (cfg.has_bias) {
        args.push_back({ArgumentDescriptor::Types::INPUT, moe_gemm::MoEGemmInputIdx::BIAS});
    }

    if (cfg.is_weight_quantized) {
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_scale_idx)});
        if (!cfg.is_weight_symmetric_quantized)
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_zp_idx)});
    }

    return args;
}

}  // namespace ov::intel_gpu::ocl
