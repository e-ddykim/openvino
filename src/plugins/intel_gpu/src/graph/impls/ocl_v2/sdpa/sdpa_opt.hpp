// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "scaled_dot_product_attention_inst.h"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {
struct SDPAStage {
    constexpr static size_t SINGLE_TOKEN = 0;
    constexpr static size_t MULTI_TOKENS = 1;
    constexpr static size_t FINALIZATION = 2;
    constexpr static size_t MICRO = 3;
};

struct SDPAOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::sdpa::opt")
    explicit SDPAOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] static bool supports_micro_sdpa(const kernel_impl_params& params);
    [[nodiscard]] static bool has_per_channel_compressed_kv(const kernel_impl_params& params);
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        const auto desc = node.as<scaled_dot_product_attention>().get_primitive();
        const auto& supported_precisions = ov::intel_gpu::op::SDPA::get_supported_precisions();
        const auto is_supported_precision = [&supported_precisions](const ov::element::Type& dt) {
            return one_of(dt, supported_precisions);
        };
        const auto is_supported_kv_precision = [&is_supported_precision](const ov::element::Type& dt) {
            return is_supported_precision(dt) || dt == ov::element::i8;
        };
        const auto& q_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::QUERY);
        const auto& k_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::KEY);
        const auto& v_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::VALUE);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            return false;
        }

        if (!is_supported_kv_precision(k_layout.data_type) || !is_supported_kv_precision(v_layout.data_type)) {
            return false;
        }

        if (!is_supported_precision(q_layout.data_type) || !is_supported_precision(out_layout.data_type)) {
            return false;
        }

        auto k_head_size = k_layout.get_partial_shape()[desc->input_k_transpose_order.back()];
        auto v_head_size = v_layout.get_partial_shape()[desc->input_v_transpose_order.back()];
        if (k_head_size.is_dynamic() || v_head_size.is_dynamic()) {
            return false;
        }

        if (desc->has_sink_input) {
            auto sink_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::SINK);
            auto q_heads_num = q_layout.get_partial_shape()[1].get_length();
            if (sink_layout.count() != static_cast<size_t>(q_heads_num)) {
                OPENVINO_THROW("Currently only supporting per-head sink.Sink_layout : ", sink_layout.to_short_string(), " heads_num  :", q_heads_num);
            }
        }

        auto p = node.get_kernel_impl_params();
        if (has_per_channel_compressed_kv(*p) && !supports_micro_sdpa(*p)) {
            return false;
        }
        // The sdpa_opt kernel decodes asymmetric quantized KV with both Planar and combined
        // scale/zp storage in every stage, so asymmetric must not push the node off this impl:
        // the sdpa_ref fallback cannot decode sub-byte (INT4) packed KV at all and silently
        // produces wrong results on devices without micro SDPA.
        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
