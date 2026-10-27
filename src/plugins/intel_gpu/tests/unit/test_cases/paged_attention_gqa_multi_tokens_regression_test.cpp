// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include "paged_attention_gpu_test.h"

namespace {

paged_attention_test_params make_gqa_regression_params(int num_heads,
                                                       int head_size,
                                                       int past_len = 300,
                                                       ov::element::Type cache_precision = ov::element::f16,
                                                       ov::internal::CacheQuantMode key_quant_mode = ov::internal::CacheQuantMode::BY_TOKEN) {
    paged_attention_test_params p{{{8, past_len}},
                                  num_heads,
                                  2,
                                  head_size,
                                  head_size,
                                  16,
                                  0,
                                  cache_precision != ov::element::f16,
                                  key_quant_mode,
                                  STATIC_INPUT_PAD,
                                  DISABLE_SCORES,
                                  DISABLE_ROTATION,
                                  DISABLE_FA_V2};
    p.kv_cache_precision = cache_precision == ov::element::u4 ? ov::element::u4 : ov::element::dynamic;
    return p;
}

}  // namespace

class paged_attention_gqa_value_cache_tail_test : public PagedAttentionTest<paged_attention_test_params> {};

TEST_P(paged_attention_gqa_value_cache_tail_test, ignores_unwritten_rows) {
    auto p = GetParam();
    ASSERT_TRUE(pam.has_value());
    ASSERT_EQ(p.subsequences.size(), 1u);

    auto result = run_gpu_inference(*pam, p);
    const auto reference = PagedAttentionReference(*pam).get_reference(result.key_cache_mem);
    ASSERT_NO_FATAL_FAILURE(compare(result.outputs.at("output_data").get_memory(), nullptr, nullptr, reference));

    const size_t valid_rows = (p.subsequences.front().past_len + p.subsequences.front().num_tokens) % p.block_size;
    ASSERT_GT(valid_rows, 0u);
    ASSERT_LT(valid_rows, static_cast<size_t>(p.block_size));
    const size_t physical_block = pam->block_indices.back();
    const ov::float16 nan = std::numeric_limits<ov::float16>::quiet_NaN();

    // Only the logically unused tail is poisoned; every valid Q/K/V value is unchanged.
    {
        cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::write> cache(result.value_cache_mem, tests::get_test_stream());
        for (int head = 0; head < p.num_kv_heads; ++head) {
            const size_t block_head = physical_block * p.num_kv_heads + head;
            for (size_t row = valid_rows; row < static_cast<size_t>(p.block_size); ++row) {
                if (!p.kv_cache_compression) {
                    const size_t offset = (block_head * p.block_size + row) * p.v_head_size * sizeof(nan);
                    for (int dim = 0; dim < p.v_head_size; ++dim) {
                        std::memcpy(cache.data() + offset + dim * sizeof(nan), &nan, sizeof(nan));
                    }
                } else if (pam->is_int4_kv_cache()) {
                    const size_t packed_head_size = p.v_head_size / 2;
                    const size_t row_stride = packed_head_size + 2 * sizeof(nan);
                    const size_t scale_offset = (block_head * p.block_size + row) * row_stride + packed_head_size;
                    std::memcpy(cache.data() + scale_offset, &nan, sizeof(nan));
                } else {
                    const size_t block_stride = p.block_size * (p.v_head_size + 2 * sizeof(nan));
                    const size_t scale_offset = block_head * block_stride + p.block_size * p.v_head_size + row * sizeof(nan);
                    std::memcpy(cache.data() + scale_offset, &nan, sizeof(nan));
                }
            }
        }
    }

    result.outputs = result.network->execute();
    auto pa = result.network->get_primitive("paged_attention");
    ASSERT_NE(pa, nullptr);
    SCOPED_TRACE(pa->get_impl()->get_kernels_dump_info(*pa->get_impl_params()).get_entries());
    compare(result.outputs.at("output_data").get_memory(), nullptr, nullptr, reference);
}

INSTANTIATE_TEST_SUITE_P(regression_paged_attention_gqa_tail,
                         paged_attention_gqa_value_cache_tail_test,
                         ::testing::Values(make_gqa_regression_params(8, 128, 35),
                                           make_gqa_regression_params(8, 128, 35, ov::element::i8),
                                           make_gqa_regression_params(8, 128, 35, ov::element::u4, ov::internal::CacheQuantMode::BY_CHANNEL),
                                           make_gqa_regression_params(8, 128, 291, ov::element::u4, ov::internal::CacheQuantMode::BY_CHANNEL)));

class paged_attention_gqa_partition_stats_test : public PagedAttentionTest<paged_attention_test_params> {};

TEST_P(paged_attention_gqa_partition_stats_test, writes_statistics_for_every_head) {
    auto p = GetParam();
    ASSERT_TRUE(pam.has_value());
    auto result = run_gpu_inference(*pam, p);
    const auto reference = PagedAttentionReference(*pam).get_reference(result.key_cache_mem);
    auto pa = result.network->get_primitive("paged_attention");
    ASSERT_NE(pa, nullptr);
    SCOPED_TRACE(pa->get_impl()->get_kernels_dump_info(*pa->get_impl_params()).get_entries());

    // Without scores, non-micro MIXED has seven buffers; statistics occupy indices 3 and 4.
    // Poison them so a missing per-head write cannot accidentally reuse correct stale values.
    const auto& buffers = pa->get_intermediates_memories();
    if (buffers.size() == 7) {
        for (const size_t index : {3u, 4u}) {
            cldnn::mem_lock<float, cldnn::mem_lock_type::write> statistics(buffers[index], tests::get_test_stream());
            std::fill(statistics.begin(), statistics.end(), std::numeric_limits<float>::quiet_NaN());
        }
    }

    result.outputs = result.network->execute();
    compare(result.outputs.at("output_data").get_memory(), nullptr, nullptr, reference);
}

INSTANTIATE_TEST_SUITE_P(regression_paged_attention_gqa_partitions,
                         paged_attention_gqa_partition_stats_test,
                         ::testing::Values(make_gqa_regression_params(4, 16),
                                           make_gqa_regression_params(6, 16),
                                           make_gqa_regression_params(8, 16),
                                           make_gqa_regression_params(6, 32),
                                           make_gqa_regression_params(6, 32, 300, ov::element::u4, ov::internal::CacheQuantMode::BY_CHANNEL),
                                           make_gqa_regression_params(6, 16, 300, ov::element::i8),
                                           // Controls: a single partition, and enough subgroups for all three heads.
                                           make_gqa_regression_params(6, 32, 35),
                                           make_gqa_regression_params(6, 64)));
