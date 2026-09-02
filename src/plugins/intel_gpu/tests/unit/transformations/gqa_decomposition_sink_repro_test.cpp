// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Repro tests for PR #37608 (GPU GroupQueryAttentionDecomposition override).
//
// The GPU override elides the attention mask under:
//
//     causal && !sliding_window_cache && !external_bias && scale == 0.0f
//
// Two independent defects follow from that condition:
//
//   (A) It does not account for the sink branch in decompose(). When it returns
//       nullptr while a sink is present, make_sdpa() skips the attn_mask slot
//       but still appends scale and sink, so the positional optional inputs of
//       v13::SDPA shift down by one:
//           slot 3 (attn_mask) <- scale scalar
//           slot 4 (scale)     <- sink [1, H, 1, 1]
//           slot 5 (sink)      <- absent
//       The sink branch also hardcodes is_causal=false, so the result is an
//       SDPA with no masking of any kind.
//
//   (B) It does not check local_window_size. Sliding window attention backed by
//       a plain (non-rotating) KV cache satisfies !sliding_window_cache, so the
//       window mask is dropped. is_causal cannot express a left window bound.
//
// gpt-oss alternates windowed and full attention per layer and carries a
// per-head sink on every layer, so (A) and (B) fire on different layers of the
// same graph.
//
// Placement: src/plugins/intel_gpu/tests/unit/transformations/
// Run:       ./ov_gpu_unit_tests --gtest_filter=GqaDecompositionRepro.*

#include <gtest/gtest.h>

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/group_query_attention_decomposition.hpp"

namespace ov::test::intel_gpu {
namespace {

using QuantType = ov::op::internal::GroupQueryAttentionQuantType;

constexpr int64_t num_heads = 2;
constexpr int64_t kv_num_heads = 1;
constexpr int64_t head_size = 16;

// Positional order of the trailing ctor arguments, confirmed empirically:
// passing true into slot 10 raised
//     "sliding_window_cache requires local_window_size >= 1, got -1"
// which pins slot 10 to sliding_window_cache, and therefore slot 11 to
// smooth_softmax. Slot 12 is causal (exercised by the control test).
// field_mapping_sanity below re-checks this at runtime so a future ctor change
// cannot silently invalidate the rest of the suite.
struct GqaConfig {
    float scale = 0.0f;              // 0.0f == "use 1/sqrt(head_size)"
    bool flag_a = false;             // do_rotary
    bool flag_b = false;             // rotary_interleaved
    int64_t softcap = 0;
    QuantType kv_quant = QuantType::NONE;
    QuantType out_quant = QuantType::NONE;
    int64_t local_window_size = -1;  // >= 1 enables sliding window attention
    bool sliding_window_cache = false;
    bool smooth_softmax = false;     // adds an extra logit -> sink branch
    bool causal = true;
};

std::shared_ptr<ov::Model> make_gqa_model(const GqaConfig& cfg) {
    const auto f32 = ov::element::f32;
    const auto past_len = ov::Dimension::dynamic();

    auto query = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, num_heads, 1, head_size});
    auto key = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, 1, head_size});
    auto value = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, 1, head_size});
    auto past_key = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, past_len, head_size});
    auto past_value =
        std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, past_len, head_size});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});

    ov::OutputVector inputs{query, key, value, past_key, past_value, seqlens_k, total_sequence_length};

    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(inputs,
                                                                      num_heads,
                                                                      kv_num_heads,
                                                                      cfg.scale,
                                                                      cfg.flag_a,
                                                                      cfg.flag_b,
                                                                      cfg.softcap,
                                                                      cfg.kv_quant,
                                                                      cfg.out_quant,
                                                                      cfg.local_window_size,
                                                                      cfg.sliding_window_cache,
                                                                      cfg.smooth_softmax,
                                                                      cfg.causal);

    ov::ResultVector results;
    for (const auto& output : gqa->outputs()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(output));
    }
    return std::make_shared<ov::Model>(results,
                                       ov::ParameterVector{query,
                                                           key,
                                                           value,
                                                           past_key,
                                                           past_value,
                                                           seqlens_k,
                                                           total_sequence_length});
}

std::shared_ptr<ov::intel_gpu::op::SDPA> decompose_and_get_sdpa(const GqaConfig& cfg) {
    auto model = make_gqa_model(cfg);
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::GroupQueryAttentionDecomposition>();
    manager.run_passes(model);

    std::shared_ptr<ov::intel_gpu::op::SDPA> result;
    for (const auto& node : model->get_ordered_ops()) {
        EXPECT_FALSE(ov::is_type<ov::op::internal::GroupQueryAttention>(node));
        if (auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SDPA>(node)) {
            result = sdpa;
        }
    }
    return result;
}

// A mask produced by make_attention_mask() is a rank-4 broadcastable tensor.
// A rank-0 value in that slot means the optional inputs have shifted and the
// scale scalar landed there instead.
::testing::AssertionResult slot_holds_a_mask(const ov::Output<ov::Node>& slot) {
    const auto& ps = slot.get_partial_shape();
    if (ps.rank().is_static() && ps.rank().get_length() == 0) {
        return ::testing::AssertionFailure()
               << "attn_mask slot holds a rank-0 (scalar) value, produced by "
               << slot.get_node()->get_type_name() << " '" << slot.get_node()->get_friendly_name()
               << "' -- this is the scale scalar, not an attention mask";
    }
    return ::testing::AssertionSuccess();
}

// ---------------------------------------------------------------------------
// Sanity checks. These must pass before any failure below can be attributed to
// the decomposition rather than to this harness.
// ---------------------------------------------------------------------------

// Pins the ctor slot mapping using the op's own validation rule
// (sliding_window_cache requires local_window_size >= 1), without depending on
// getter names. If either half fails, GqaConfig no longer matches the ctor and
// every other result in this file is meaningless.
TEST(GqaDecompositionRepro, field_mapping_sanity) {
    {
        GqaConfig cfg;
        cfg.sliding_window_cache = true;
        cfg.local_window_size = -1;
        EXPECT_ANY_THROW(make_gqa_model(cfg))
            << "slot 10 is not sliding_window_cache -- GqaConfig field order is wrong";
    }
    {
        GqaConfig cfg;
        cfg.smooth_softmax = true;
        cfg.local_window_size = -1;
        EXPECT_NO_THROW(make_gqa_model(cfg))
            << "slot 11 rejected local_window_size=-1, so it is not smooth_softmax "
               "-- GqaConfig field order is wrong";
    }
}

// The case the PR was written for: plain causal, no sink, no window, scale = 0.
// Expected to PASS on the PR head.
TEST(GqaDecompositionRepro, control_plain_causal_uses_lower_right_without_mask) {
    GqaConfig cfg;
    const auto sdpa = decompose_and_get_sdpa(cfg);

    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_size(), 3u) << "Q, K, V only -- the mask is elided here by design";
    EXPECT_TRUE(sdpa->get_causal());
    EXPECT_EQ(sdpa->get_causal_mask_alignment(), ov::intel_gpu::op::SDPA::CausalMaskAlignment::LOWER_RIGHT);
}

// Second control, discovered accidentally: when sliding_window_cache is true the
// nullptr condition is correctly defeated and a real mask is built. This shows
// the guard works when it is actually consulted, which is what makes (B) below
// a genuine gap rather than a design choice. Expected to PASS.
TEST(GqaDecompositionRepro, control_sliding_window_cache_defeats_mask_elision) {
    GqaConfig cfg;
    cfg.local_window_size = 128;
    cfg.sliding_window_cache = true;

    const auto sdpa = decompose_and_get_sdpa(cfg);
    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_size(), 4u) << "Q, K, V, mask";
    EXPECT_TRUE(slot_holds_a_mask(sdpa->input_value(3)));
}

// ---------------------------------------------------------------------------
// (A) Sink + causal -> optional inputs shift by one slot.
//
// decompose() takes the sink branch, which always materialises a scale_node
// (1/sqrt(head_size) when the attribute is 0), so the `scale == 0.0f` term in
// the nullptr condition does not protect this path.
// ---------------------------------------------------------------------------
TEST(GqaDecompositionRepro, sink_causal_must_not_shift_optional_inputs) {
    GqaConfig cfg;
    cfg.smooth_softmax = true;
    cfg.causal = true;
    cfg.scale = 0.0f;

    const auto sdpa = decompose_and_get_sdpa(cfg);
    ASSERT_NE(sdpa, nullptr);

    // v13::SDPA slots are positional: 3=attn_mask, 4=scale, 5=sink.
    ASSERT_EQ(sdpa->get_input_size(), 6u)
        << "expected Q,K,V,mask,scale,sink. Got " << sdpa->get_input_size()
        << " inputs -- the attn_mask slot was skipped while scale and sink were still "
           "appended, so each optional input moved down one position and the sink fell "
           "off the end.";

    EXPECT_TRUE(slot_holds_a_mask(sdpa->input_value(3)));

    const bool has_real_mask = sdpa->get_input_size() > 3 && slot_holds_a_mask(sdpa->input_value(3));
    EXPECT_TRUE(sdpa->get_causal() || has_real_mask)
        << "SDPA is neither is_causal nor masked: full bidirectional attention, "
           "every query position can see future tokens.";
}

// Same defect with an explicit non-zero scale, showing that `scale != 0` is what
// accidentally protects the non-sink branch and does nothing here.
TEST(GqaDecompositionRepro, sink_causal_explicit_scale_must_not_shift_optional_inputs) {
    GqaConfig cfg;
    cfg.smooth_softmax = true;
    cfg.causal = true;
    cfg.scale = 0.125f;

    const auto sdpa = decompose_and_get_sdpa(cfg);
    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_size(), 6u);
    EXPECT_TRUE(slot_holds_a_mask(sdpa->input_value(3)));
}

// ---------------------------------------------------------------------------
// (B) Sliding window with a plain KV cache -> window mask dropped.
// CONFIRMED: this reproduces with get_input_size() == 3.
// ---------------------------------------------------------------------------
TEST(GqaDecompositionRepro, sliding_window_must_keep_explicit_mask) {
    GqaConfig cfg;
    cfg.causal = true;
    cfg.local_window_size = 128;
    cfg.sliding_window_cache = false;  // plain past/present cache, not rotating

    const auto sdpa = decompose_and_get_sdpa(cfg);
    ASSERT_NE(sdpa, nullptr);

    EXPECT_GT(sdpa->get_input_size(), 3u)
        << "local_window_size=" << cfg.local_window_size
        << " requires an explicit window mask, but the mask was elided. is_causal alone "
           "cannot express a left window bound, so attention now spans the whole KV cache.";
}

// Worst case for gpt-oss: a windowed layer that also carries a sink. Both
// defects apply -- no window bound and no causal bound.
TEST(GqaDecompositionRepro, sliding_window_with_sink_must_stay_masked) {
    GqaConfig cfg;
    cfg.causal = true;
    cfg.smooth_softmax = true;
    cfg.local_window_size = 128;
    cfg.sliding_window_cache = false;

    const auto sdpa = decompose_and_get_sdpa(cfg);
    ASSERT_NE(sdpa, nullptr);

    const bool has_real_mask = sdpa->get_input_size() > 3 && slot_holds_a_mask(sdpa->input_value(3));
    EXPECT_TRUE(has_real_mask) << "windowed + sink layer ended up with no attention mask at all";
    EXPECT_EQ(sdpa->get_input_size(), 6u);
}

}  // namespace
}  // namespace ov::test::intel_gpu
