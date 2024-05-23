// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "transpose_fusion.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

namespace ov {
namespace intel_gpu {

TransposeFusion::TransposeFusion() {
    add_matcher<TransposeTransposedMatMulMatcher>();
    add_matcher<TransposeMatMulTransposeMatcher>();
    add_matcher<TransposeMatMulMatcher>();
    add_matcher<TransposeSDPAMatcher>();
}

// This matcher removes 'transpose_b' of MatMul.
// before: Transpose -> Multiply -> Reshape -> MatMul with transpose_b
// after: Transpose with transposed last 2 dims -> Multiply -> Reshape -> MatMul without transpose_b
TransposeTransposedMatMulMatcher::TransposeTransposedMatMulMatcher() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
            case ov::element::f16:
            case ov::element::f32: return true;
            default: return false;
        }
    };
    auto not_transpose = [is_fp_type](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
               && is_fp_type(output);
    };

    auto input_a_m = any_input(not_transpose);
    auto input_b_m = any_input(not_transpose);

    auto transpose_b_order_m = wrap_type<ov::op::v0::Constant>();
    auto transpose_b_m = wrap_type<ov::op::v1::Transpose>({input_b_m, transpose_b_order_m}, is_fp_type);

    auto multiply_b_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto multiply_b_m = wrap_type<ov::op::v1::Multiply>({transpose_b_m, multiply_b_const_m}, is_fp_type);

    auto reshape_b_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto reshape_b_m = wrap_type<ov::op::v1::Reshape>({multiply_b_m, reshape_b_const_m}, is_fp_type);

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({ input_a_m, reshape_b_m });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        if (matmul->get_transpose_b()) {
            std::cout << "@@@@@@ " << matmul->get_friendly_name() << std::endl;

            bool multiplied_by_scalar = false;

            auto multiply_b_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(multiply_b_const_m).get_node_shared_ptr());
            if (multiply_b_const->get_output_shape(0).size() == 0) {
                std::cout << "scalar!" << std::endl;
                multiplied_by_scalar = true;
            }

            bool last_2_dims_unchanged = false;
            if (multiplied_by_scalar) {
                auto reshape_b = std::dynamic_pointer_cast<ov::op::v1::Reshape>(pattern_map.at(reshape_b_m).get_node_shared_ptr());
                auto input_shape = reshape_b->get_input_partial_shape(0);
                auto output_shape = reshape_b->get_output_partial_shape(0);

                int64_t input_size = 1;
                for (auto& dim : input_shape) {
                    if (dim.is_static() && dim.get_max_length() > 0)
                        input_size *= dim.get_max_length();
                }

                int64_t output_size = 1;
                for (auto& dim : output_shape) {
                    if (dim.is_static() && dim.get_max_length() > 0)
                        output_size *= dim.get_max_length();
                }

                if (input_size == output_size) {
                    if (input_shape[input_shape.size()-2].get_max_length() == output_shape[output_shape.size()-2].get_max_length() &&
                        input_shape[input_shape.size()-1].get_max_length() == output_shape[output_shape.size()-1].get_max_length()) {
                        std::cout << "last 2 dims are preserved by reshape!" << std::endl;
                        last_2_dims_unchanged = true;
                    }
                }
            }

            if (last_2_dims_unchanged) {
                {
                    auto tranpose_b_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_b_order_m).get_node_shared_ptr());
                    auto order_b = tranpose_b_order->cast_vector<int64_t>();

                    auto tmp = order_b[order_b.size()-1];
                    order_b[order_b.size()-1] = order_b[order_b.size()-2];
                    order_b[order_b.size()-2] = tmp;

                    auto fused_transpoed_b_order = ov::op::v0::Constant::create(tranpose_b_order->get_element_type(),
                                                                                tranpose_b_order->get_shape(), order_b);

                    auto tranpose_b = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_b_m).get_node_shared_ptr());
                    auto new_tranpose_b = std::make_shared<ov::op::v1::Transpose>(tranpose_b->get_input_node_shared_ptr(0), fused_transpoed_b_order);
                    new_tranpose_b->set_friendly_name(tranpose_b->get_friendly_name());
                    ov::copy_runtime_info(tranpose_b, new_tranpose_b);
                    ov::replace_node(tranpose_b, new_tranpose_b);
                }

                {
                    auto reshape_b_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(reshape_b_const_m).get_node_shared_ptr());
                    auto shape_b = reshape_b_const->cast_vector<int64_t>();

                    auto tmp = shape_b[shape_b.size()-1];
                    shape_b[shape_b.size()-1] = shape_b[shape_b.size()-2];
                    shape_b[shape_b.size()-2] = tmp;

                    auto new_reshape_b_const = ov::op::v0::Constant::create(reshape_b_const->get_element_type(),
                                                                            reshape_b_const->get_shape(), shape_b);

                    auto reshape_b = std::dynamic_pointer_cast<ov::op::v1::Reshape>(pattern_map.at(reshape_b_m).get_node_shared_ptr());
                    auto new_reshape_b = std::make_shared<ov::op::v1::Reshape>(reshape_b->get_input_node_shared_ptr(0), new_reshape_b_const, false);
                    new_reshape_b->set_friendly_name(reshape_b->get_friendly_name());
                    ov::copy_runtime_info(reshape_b, new_reshape_b);
                    ov::replace_node(reshape_b, new_reshape_b);
                }

                matmul->set_transpose_b(false);

                std::cout << "transpose fusion!!" << std::endl;
                return true;
            }
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_m, "TransposeTransposeMatMulMatcher");
    this->register_matcher(m, callback);
}

TransposeSDPAMatcher::TransposeSDPAMatcher() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
            case ov::element::f16:
            case ov::element::f32: return true;
            default: return false;
        }
    };
    auto not_transpose = [is_fp_type](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
               && is_fp_type(output);
    };
    auto is_dynamic = [](const ov::Output<ov::Node>& output) -> bool {
        bool is_dynamic = output.get_node_shared_ptr()->get_output_partial_shape(0).is_dynamic();
        size_t num_inputs = output.get_node_shared_ptr()->get_input_size();
        for (size_t idx = 0; idx < num_inputs; idx++) {
            is_dynamic |= output.get_node_shared_ptr()->get_input_partial_shape(idx).is_dynamic();
        }
        return is_dynamic;
    };

    auto input_q_m = any_input(not_transpose);
    auto input_k_m = any_input(not_transpose);
    auto input_v_m = any_input(not_transpose);
    auto input_attn_mask = any_input(not_transpose);
    auto input_scale = any_input(not_transpose);
    auto transpose_q_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_k_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_v_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_q_m = wrap_type<ov::op::v1::Transpose>({input_q_m, transpose_q_order_m}, is_fp_type);
    auto transpose_k_m = wrap_type<ov::op::v1::Transpose>({input_k_m, transpose_k_order_m}, is_fp_type);
    auto transpose_v_m = wrap_type<ov::op::v1::Transpose>({input_v_m, transpose_v_order_m}, is_fp_type);

    auto sdpa_in_q = std::make_shared<Or>(OutputVector{input_q_m, transpose_q_m});
    auto sdpa_in_k = std::make_shared<Or>(OutputVector{input_k_m, transpose_k_m});
    auto sdpa_in_v = std::make_shared<Or>(OutputVector{input_v_m, transpose_v_m});

    auto sdpa_without_attn_mask_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({ sdpa_in_q, sdpa_in_k, sdpa_in_v }, is_dynamic);
    auto sdpa_with_attn_mask_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({ sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask }, is_dynamic);
    auto sdpa_with_attn_mask_and_scale_m =
        wrap_type<ov::op::v13::ScaledDotProductAttention>({ sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask, input_scale }, is_dynamic);

    auto sdpa_m = std::make_shared<Or>(OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdpa;
        if (pattern_map.find(sdpa_without_attn_mask_m) != pattern_map.end()) {
            sdpa = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(pattern_map.at(sdpa_without_attn_mask_m).get_node_shared_ptr());
        } else if (pattern_map.find(sdpa_with_attn_mask_m) != pattern_map.end()) {
            sdpa = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(pattern_map.at(sdpa_with_attn_mask_m).get_node_shared_ptr());
        } else if (pattern_map.find(sdpa_with_attn_mask_and_scale_m) != pattern_map.end()) {
            sdpa = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(pattern_map.at(sdpa_with_attn_mask_and_scale_m).get_node_shared_ptr());
        }

        if (!sdpa || transformation_callback(sdpa)) {
            return false;
        }

        auto order_q = op::SDPA::default_order(sdpa->get_input_partial_shape(0).size());
        auto order_k = op::SDPA::default_order(sdpa->get_input_partial_shape(1).size());
        auto order_v = op::SDPA::default_order(sdpa->get_input_partial_shape(2).size());
        auto order_output = op::SDPA::default_order(sdpa->get_output_partial_shape(0).size());
        size_t input_q_output_idx = sdpa->get_input_source_output(0).get_index();
        size_t input_k_output_idx = sdpa->get_input_source_output(1).get_index();
        size_t input_v_output_idx = sdpa->get_input_source_output(2).get_index();

        if (pattern_map.count(transpose_q_m) > 0) {
            auto tranpose_a_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_q_order_m).get_node_shared_ptr());
            order_q = tranpose_a_order->cast_vector<int64_t>();
            if (order_q.back() != static_cast<int64_t>(order_q.size() - 1)) // Allow any transposes without head_size dim position change
                return false;

            auto tranpose_a = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_q_m).get_node_shared_ptr());
            input_q_output_idx = tranpose_a->get_input_source_output(0).get_index();
        }
        if (pattern_map.count(transpose_k_m) > 0) {
            auto tranpose_b_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_k_order_m).get_node_shared_ptr());
            order_k = tranpose_b_order->cast_vector<int64_t>();
            if (order_k.back() != static_cast<int64_t>(order_k.size() - 1)) // Allow any transposes without head_size dim position change
                return false;

            auto tranpose_b = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_k_m).get_node_shared_ptr());
            input_k_output_idx = tranpose_b->get_input_source_output(0).get_index();
        }
        if (pattern_map.count(transpose_v_m) > 0) {
            auto tranpose_c_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_v_order_m).get_node_shared_ptr());
            order_v = tranpose_c_order->cast_vector<int64_t>();
            if (order_v.back() != static_cast<int64_t>(order_v.size() - 1)) // Allow any transposes without head_size dim position change
                return false;

            auto tranpose_c = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_k_m).get_node_shared_ptr());
            input_v_output_idx = tranpose_c->get_input_source_output(0).get_index();
        }

        auto input_q = ov::Output<Node>(pattern_map.at(input_q_m).get_node_shared_ptr(), input_q_output_idx);
        auto input_k = ov::Output<Node>(pattern_map.at(input_k_m).get_node_shared_ptr(), input_k_output_idx);
        auto input_v = ov::Output<Node>(pattern_map.at(input_v_m).get_node_shared_ptr(), input_v_output_idx);

        std::shared_ptr<op::SDPA> sdpa_new;
        if (pattern_map.find(sdpa_without_attn_mask_m) != pattern_map.end()) {
            sdpa_new = std::make_shared<op::SDPA>(input_q, input_k, input_v, order_q, order_k, order_v, order_output, sdpa->get_causal());
        } else if (pattern_map.find(sdpa_with_attn_mask_m) != pattern_map.end()) {
            auto attn_mask = sdpa->get_input_source_output(3);
            sdpa_new = std::make_shared<op::SDPA>(input_q, input_k, input_v, attn_mask, order_q, order_k, order_v, order_output, sdpa->get_causal());
        } else if (pattern_map.find(sdpa_with_attn_mask_and_scale_m) != pattern_map.end()) {
            auto attn_mask = sdpa->get_input_source_output(3);
            auto scale = sdpa->get_input_source_output(4);
            sdpa_new = std::make_shared<op::SDPA>(input_q, input_k, input_v, attn_mask, scale, order_q, order_k, order_v, order_output, sdpa->get_causal());
        }

        sdpa_new->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa_new);
        ov::replace_node(sdpa, sdpa_new);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "TransposeSDPAMatcher");
    this->register_matcher(m, callback);
}

TransposeMatMulMatcher::TransposeMatMulMatcher() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
            case ov::element::f16:
            case ov::element::f32: return true;
            default: return false;
        }
    };
    auto not_transpose = [is_fp_type](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
               && is_fp_type(output);
    };
    auto is_dynamic = [](const ov::Output<ov::Node>& output) -> bool {
        bool is_dynamic = output.get_node_shared_ptr()->get_output_partial_shape(0).is_dynamic();
        size_t num_inputs = output.get_node_shared_ptr()->get_input_size();
        for (size_t idx = 0; idx < num_inputs; idx++) {
            is_dynamic |= output.get_node_shared_ptr()->get_input_partial_shape(idx).is_dynamic();
        }
        return is_dynamic;
    };
    auto input_a_m = any_input(not_transpose);
    auto input_b_m = any_input(not_transpose);
    auto transpose_a_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_b_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_a_m = wrap_type<ov::op::v1::Transpose>({input_a_m, transpose_a_order_m}, is_fp_type);
    auto transpose_b_m = wrap_type<ov::op::v1::Transpose>({input_b_m, transpose_b_order_m}, is_fp_type);

    auto matmul_in_a = std::make_shared<Or>(OutputVector{input_a_m, transpose_a_m});
    auto matmul_in_b = std::make_shared<Or>(OutputVector{input_b_m, transpose_b_m});

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({ matmul_in_a, matmul_in_b }, is_dynamic);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto users = matmul->get_output_target_inputs(0);
        if (users.size() == 1 && dynamic_cast<ov::op::v1::Transpose*>(users.begin()->get_node()) != nullptr) {
            return false;
        }

        auto order_a = op::Gemm::default_order(matmul->get_input_partial_shape(0).size());
        auto order_b = op::Gemm::default_order(matmul->get_input_partial_shape(1).size());
        auto order_c = op::Gemm::default_order(matmul->get_output_partial_shape(0).size());
        size_t input_a_output_idx = matmul->get_input_source_output(0).get_index();
        size_t input_b_output_idx = matmul->get_input_source_output(1).get_index();

        if (pattern_map.count(transpose_a_m) > 0) {
            auto tranpose_a_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_a_order_m).get_node_shared_ptr());
            order_a = tranpose_a_order->cast_vector<int64_t>();
            auto tranpose_a = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_a_m).get_node_shared_ptr());
            input_a_output_idx = tranpose_a->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_a() && order_a.size() > 1) {
            std::swap(*(order_a.end() - 1), *(order_a.end() - 2));
        }
        if (pattern_map.count(transpose_b_m) > 0) {
            auto tranpose_b_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_b_order_m).get_node_shared_ptr());
            order_b = tranpose_b_order->cast_vector<int64_t>();
            auto tranpose_b = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_b_m).get_node_shared_ptr());
            input_b_output_idx = tranpose_b->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_b() && order_b.size() > 1) {
            std::swap(*(order_b.end() - 1), *(order_b.end() - 2));
        }

        auto input_a = ov::Output<Node>(pattern_map.at(input_a_m).get_node_shared_ptr(), input_a_output_idx);
        auto input_b = ov::Output<Node>(pattern_map.at(input_b_m).get_node_shared_ptr(), input_b_output_idx);

        auto gemm = std::make_shared<op::Gemm>(input_a, input_b, order_a, order_b, order_c);
        gemm->set_friendly_name(matmul->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), gemm);
        ov::replace_node(matmul, gemm);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_m, "TransposeMatMulMatcher");
    this->register_matcher(m, callback);
}

TransposeMatMulTransposeMatcher::TransposeMatMulTransposeMatcher() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
            case ov::element::f16:
            case ov::element::f32: return true;
            default: return false;
        }
    };
    auto not_transpose = [is_fp_type](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
               && is_fp_type(output);
    };
    auto is_dynamic = [](const ov::Output<ov::Node>& output) -> bool {
        bool is_dynamic = output.get_node_shared_ptr()->get_output_partial_shape(0).is_dynamic();
        size_t num_inputs = output.get_node_shared_ptr()->get_input_size();
        for (size_t idx = 0; idx < num_inputs; idx++) {
            is_dynamic |= output.get_node_shared_ptr()->get_input_partial_shape(idx).is_dynamic();
        }
        return is_dynamic;
    };
    auto input_a_m = any_input(not_transpose);
    auto input_b_m = any_input(not_transpose);
    auto transpose_a_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_b_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_a_m = wrap_type<ov::op::v1::Transpose>({input_a_m, transpose_a_order_m}, is_fp_type);
    auto transpose_b_m = wrap_type<ov::op::v1::Transpose>({input_b_m, transpose_b_order_m}, is_fp_type);

    auto matmul_in_a = std::make_shared<Or>(OutputVector{input_a_m, transpose_a_m});
    auto matmul_in_b = std::make_shared<Or>(OutputVector{input_b_m, transpose_b_m});

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({ matmul_in_a, matmul_in_b }, is_dynamic);
    auto transpose_c_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_c_m = wrap_type<ov::op::v1::Transpose>({matmul_m, transpose_c_order_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto tranpose_c_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_c_order_m).get_node_shared_ptr());
        auto order_a = op::Gemm::default_order(matmul->get_input_partial_shape(0).size());
        auto order_b = op::Gemm::default_order(matmul->get_input_partial_shape(1).size());
        auto order_c = tranpose_c_order->cast_vector<int64_t>();
        size_t input_a_output_idx = matmul->get_input_source_output(0).get_index();
        size_t input_b_output_idx = matmul->get_input_source_output(1).get_index();

        if (pattern_map.count(transpose_a_m) > 0) {
            auto tranpose_a_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_a_order_m).get_node_shared_ptr());
            order_a = tranpose_a_order->cast_vector<int64_t>();
            auto tranpose_a = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_a_m).get_node_shared_ptr());
            input_a_output_idx = tranpose_a->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_a() && order_a.size() > 1) {
            std::swap(*(order_a.end() - 1), *(order_a.end() - 2));
        }
        if (pattern_map.count(transpose_b_m) > 0) {
            auto tranpose_b_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_b_order_m).get_node_shared_ptr());
            order_b = tranpose_b_order->cast_vector<int64_t>();
            auto tranpose_b = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_b_m).get_node_shared_ptr());
            input_b_output_idx = tranpose_b->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_b() && order_b.size() > 1) {
            std::swap(*(order_b.end() - 1), *(order_b.end() - 2));
        }

        auto input_a = ov::Output<Node>(pattern_map.at(input_a_m).get_node_shared_ptr(), input_a_output_idx);
        auto input_b = ov::Output<Node>(pattern_map.at(input_b_m).get_node_shared_ptr(), input_b_output_idx);

        auto gemm = std::make_shared<op::Gemm>(input_a, input_b, order_a, order_b, order_c);
        gemm->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), gemm);
        ov::replace_node(m.get_match_root(), gemm);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_c_m, "TransposeMatMulTransposeMatcher");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
