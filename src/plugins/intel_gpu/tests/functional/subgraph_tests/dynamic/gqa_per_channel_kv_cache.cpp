// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

// GroupQueryAttention with a per-channel 8-bit KV cache, executed end to end on GPU (f16) against the TEMPLATE plugin (f32).
// The GPU GroupQueryAttentionDecomposition hands an i8 cache of a head size up to 512 to SDPA as is; a u8 cache and a head
// size above 512 are dequantized in the graph. The current K / V are integer codes times the channel scale, so both plugins
// quantize them exactly and the present caches must match bit for bit. GPUs without XMX never hand the cache to SDPA, so the
// suite runs on XMX GPUs only.
namespace {

struct GQAPerChannelKVParams {
    ov::element::Type cache_type;  // i8 or u8
    size_t head_size;
};

constexpr size_t num_heads = 8;
constexpr size_t kv_num_heads = 2;

// Powers of two, different per channel so a K/V channel mix-up shows.
std::vector<float> make_scales(const GQAPerChannelKVParams& p, std::mt19937& gen) {
    std::uniform_int_distribution<int> exponent(-7, -4);
    std::vector<float> scales(kv_num_heads * p.head_size);
    for (auto& scale : scales) {
        scale = std::ldexp(1.0f, exponent(gen));
    }
    return scales;
}

std::shared_ptr<ov::Model> make_gqa_model(const GQAPerChannelKVParams& p,
                                          const std::vector<float>& key_scales,
                                          const std::vector<float>& value_scales,
                                          const ov::element::Type& type) {
    const auto D = static_cast<int64_t>(p.head_size);
    const auto H = static_cast<int64_t>(num_heads);
    const auto KVH = static_cast<int64_t>(kv_num_heads);

    auto query = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{1, H, -1, D});
    auto key = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{1, KVH, -1, D});
    auto value = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{1, KVH, -1, D});
    auto past_key = std::make_shared<ov::op::v0::Parameter>(p.cache_type, ov::PartialShape{1, KVH, -1, D});
    auto past_value = std::make_shared<ov::op::v0::Parameter>(p.cache_type, ov::PartialShape{1, KVH, -1, D});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});
    const ov::ParameterVector parameters{query, key, value, past_key, past_value, seqlens_k, total_sequence_length};
    // Named like a model read from a file (the frontends name every Parameter tensor).
    const std::vector<std::string> names = {"query", "key", "value", "past_key", "past_value", "seqlens_k", "total_sequence_length"};
    for (size_t i = 0; i < parameters.size(); ++i) {
        parameters[i]->output(0).set_names({names[i]});
    }

    ov::OutputVector inputs(14);
    for (size_t i = 0; i < parameters.size(); ++i) {
        inputs[i] = parameters[i];
    }
    for (size_t i = 7; i <= 11; ++i) {
        inputs[i] = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
    }
    inputs[12] = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{key_scales.size()}, key_scales);
    inputs[13] = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{value_scales.size()}, value_scales);

    const auto per_channel = ov::op::internal::GroupQueryAttentionQuantType::PER_CHANNEL;
    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(inputs, H, KVH, 0.0f, false, false, 8, per_channel, per_channel, -1, false);

    ov::ResultVector results;
    for (size_t i = 0; i < gqa->get_output_size(); ++i) {
        results.push_back(std::make_shared<ov::op::v0::Result>(gqa->output(i)));
    }
    return std::make_shared<ov::Model>(results, parameters);
}

// Integer codes times the channel scale: u8 codes 0..127 (the same bytes under i8 or u8 storage), i8 codes -127..127.
ov::Tensor coded_tensor(const GQAPerChannelKVParams& p, const std::vector<float>& scales, size_t seq, std::mt19937& gen) {
    ov::Tensor tensor(ov::element::f16, ov::Shape{1, kv_num_heads, seq, p.head_size});
    std::uniform_int_distribution<int> code(p.cache_type == ov::element::u8 ? 0 : -127, 127);
    auto* data = tensor.data<ov::float16>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        const size_t channel = i % p.head_size;
        const size_t kv_head = (i / (p.head_size * seq)) % kv_num_heads;
        data[i] = ov::float16(static_cast<float>(code(gen)) * scales[kv_head * p.head_size + channel]);
    }
    return tensor;
}

// Past cache bytes: u8 0..255, i8 -127..127.
ov::Tensor cache_tensor(const GQAPerChannelKVParams& p, size_t past, std::mt19937& gen) {
    ov::Tensor tensor(p.cache_type, ov::Shape{1, kv_num_heads, past, p.head_size});
    const bool u8 = p.cache_type == ov::element::u8;
    std::uniform_int_distribution<int> byte(u8 ? 0 : -127, u8 ? 255 : 127);
    auto* data = static_cast<uint8_t*>(tensor.data());
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = static_cast<uint8_t>(byte(gen));
    }
    return tensor;
}

ov::Tensor to_f32(const ov::Tensor& tensor) {
    ov::Tensor result(ov::element::f32, tensor.get_shape());
    const auto* src = tensor.data<const ov::float16>();
    auto* dst = result.data<float>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
    return result;
}

class GQAPerChannelKVCacheTest : public ::testing::TestWithParam<GQAPerChannelKVParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<GQAPerChannelKVParams>& info) {
        const auto& p = info.param;
        return p.cache_type.get_type_name() + "_D" + std::to_string(p.head_size);
    }
};

TEST_P(GQAPerChannelKVCacheTest, MatchesTemplate) {
    const auto& p = GetParam();
    auto core = ov::test::utils::PluginCache::get().core();
    const auto caps = core->get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
    if (std::find(caps.begin(), caps.end(), ov::intel_gpu::capability::HW_MATMUL) == caps.end()) {
        GTEST_SKIP() << "The GQA decomposition hands a per-channel cache to SDPA only on GPUs with HW_MATMUL";
    }
    std::mt19937 gen(static_cast<uint32_t>(p.head_size * 10 + (p.cache_type == ov::element::u8 ? 1 : 0)));
    const auto key_scales = make_scales(p, gen);
    const auto value_scales = make_scales(p, gen);
    auto gpu_model = core->compile_model(make_gqa_model(p, key_scales, value_scales, ov::element::f16),
                                         ov::test::utils::DEVICE_GPU,
                                         ov::hint::inference_precision(ov::element::f16));
    auto ref_model = core->compile_model(make_gqa_model(p, key_scales, value_scales, ov::element::f32), ov::test::utils::DEVICE_TEMPLATE);
    auto gpu = gpu_model.create_infer_request();
    auto ref = ref_model.create_infer_request();

    // {past, current}: prefill, decode at two cache lengths, chunked prefill on a past.
    const std::vector<std::pair<size_t, size_t>> steps = {{0, 40}, {40, 1}, {300, 1}, {64, 33}};
    for (const auto& [past, current] : steps) {
        SCOPED_TRACE("past " + std::to_string(past) + ", current " + std::to_string(current));
        ov::Tensor query(ov::element::f16, ov::Shape{1, num_heads, current, p.head_size});
        std::uniform_real_distribution<float> query_dist(-1.0f, 1.0f);
        for (size_t i = 0; i < query.get_size(); ++i) {
            query.data<ov::float16>()[i] = ov::float16(query_dist(gen));
        }
        std::vector<ov::Tensor> inputs = {query,
                                          coded_tensor(p, key_scales, current, gen),
                                          coded_tensor(p, value_scales, current, gen),
                                          cache_tensor(p, past, gen),
                                          cache_tensor(p, past, gen),
                                          ov::Tensor(ov::element::i32, ov::Shape{1}),
                                          ov::Tensor(ov::element::i32, ov::Shape{})};
        inputs[5].data<int32_t>()[0] = static_cast<int32_t>(past + current - 1);
        inputs[6].data<int32_t>()[0] = static_cast<int32_t>(past + current);
        for (size_t i = 0; i < inputs.size(); ++i) {
            gpu.set_input_tensor(i, inputs[i]);
            ref.set_input_tensor(i, inputs[i].get_element_type() == ov::element::f16 ? to_f32(inputs[i]) : inputs[i]);
        }
        gpu.infer();
        ref.infer();

        for (size_t o = 0; o < 3; ++o) {
            const auto expected = ref.get_output_tensor(o);
            const auto actual = gpu.get_output_tensor(o);
            ASSERT_EQ(expected.get_shape(), actual.get_shape()) << "output " << o;
            if (o > 0) {
                // Present caches: the past bytes plus the exactly quantized current tokens.
                ASSERT_EQ(expected.get_element_type(), actual.get_element_type()) << "output " << o;
                const auto* expected_bytes = static_cast<const uint8_t*>(expected.data());
                const auto* actual_bytes = static_cast<const uint8_t*>(actual.data());
                size_t mismatches = 0;
                for (size_t i = 0; i < expected.get_byte_size(); ++i) {
                    mismatches += expected_bytes[i] != actual_bytes[i] ? 1 : 0;
                }
                EXPECT_EQ(mismatches, 0u) << "output " << o << ": " << mismatches << " of " << expected.get_byte_size() << " bytes differ";
                continue;
            }
            // Attention output: f16 compute against f32.
            size_t mismatches = 0;
            float max_diff = 0.0f;
            for (size_t i = 0; i < expected.get_size(); ++i) {
                const float e = expected.data<const float>()[i];
                const float diff = std::fabs(e - static_cast<float>(actual.data<const ov::float16>()[i]));
                if (!(diff <= 0.02f + 0.01f * std::fabs(e))) {  // NaN counts as a mismatch
                    ++mismatches;
                }
                if (diff > max_diff || std::isnan(diff)) {
                    max_diff = diff;
                }
            }
            EXPECT_EQ(mismatches, 0u) << "output 0: " << mismatches << " of " << expected.get_size() << " elements off, max diff " << max_diff;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_GQAPerChannelKVCache,
                         GQAPerChannelKVCacheTest,
                         ::testing::Values(GQAPerChannelKVParams{ov::element::i8, 64},
                                           GQAPerChannelKVParams{ov::element::i8, 128},
                                           GQAPerChannelKVParams{ov::element::u8, 64},
                                           GQAPerChannelKVParams{ov::element::u8, 128},
                                           GQAPerChannelKVParams{ov::element::i8, 576}),
                         GQAPerChannelKVCacheTest::get_test_case_name);

}  // namespace
