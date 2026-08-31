// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/batch_headers/imad.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

// Depthwise convolution does not mix channels, so a lane can own a channel outright: the horizontal
// taps come from neighbouring registers of the same lane instead of subgroup shuffles, and no SLM
// transpose is needed.
//
// Both sides of this kernel are channels-innermost, which is what makes it memory-efficient:
//   input  byxf(B, C, H, W)  -> a CHANNEL_BLOCK of channels is CHANNEL_BLOCK contiguous elements
//   output bfyx(B, H, W, C)  -> likewise, C sits on the innermost axis
// so one aligned CHANNELS_PER_LANE-wide block read feeds a whole 64-byte cache line into the
// subgroup, and one block write drains one back out. Note byxf(B,C,H,W) and bfyx(B,H,W,C) are the
// same bytes, so the absorbed output transpose is pure index arithmetic.
//
// Lane <-> data mapping:
//   lane             -> channel. Two channels per lane, SUB_GROUP_SIZE apart, so that one
//                       CHANNELS_PER_LANE-wide block access covers CHANNEL_BLOCK contiguous
//                       channels, i.e. a whole cache line.
//   get_group_id(0)  -> tile of TILE_X output columns
//   get_global_id(1) -> tile of TILE_Y output rows
//   get_group_id(2)  -> (batch, channel block)

#if CHANNELS_PER_LANE != 2
#    error convolution_gpu_bfyx_f16_i8_dw_fused_quantize_imad.cl - CHANNELS_PER_LANE must be 2.
#endif
#if CHANNEL_BLOCK != (SUB_GROUP_SIZE * CHANNELS_PER_LANE)
#    error convolution_gpu_bfyx_f16_i8_dw_fused_quantize_imad.cl - inconsistent CHANNEL_BLOCK.
#endif

#define FILTER_TYPE4               MAKE_VECTOR_TYPE(FILTER_TYPE, 4)
#define GET_WEIGHTS_INDEX(g, y, x) GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(FILTER, g, 0, 0, y, x)

#define INPUT_VEC  MAKE_VECTOR_TYPE(INPUT0_TYPE, CHANNELS_PER_LANE)
#define OUTPUT_VEC MAKE_VECTOR_TYPE(OUTPUT_TYPE, CHANNELS_PER_LANE)

// One output row reads exactly three input rows, so the row cache is a sliding window of three
// slots rather than the whole TILE_Y + 2 tall stripe. Input row index k, counted from the top of the
// stripe, lives in slot k % ROW_SLOTS; k is a compile-time constant everywhere, so the modulo folds
// away and the window slides by renaming rather than by copying.
#define ROW_SLOTS 3
// Position p of a row slot holds input x == output_x_start - 1 + p, so output column
// output_x_start + i reads its three taps at p == i, i + 1, i + 2.
#define ROW_POSITIONS (TILE_X + 2)

#if INPUT_QUANTIZATION_USE_FP16_ARITHMETIC
#    define INPUT_QUANTIZATION_TYPE           half
#    define TO_INPUT_QUANTIZATION_TYPE(value) convert_half(value)
#else
#    define INPUT_QUANTIZATION_TYPE           float
#    define TO_INPUT_QUANTIZATION_TYPE(value) convert_float(value)
#endif

// Reproduces quantize_gpu_scale_shift_opt semantics: the scale/shift expression is evaluated and
// rounded in INPUT_QUANTIZATION_TYPE before the output shift and the saturating conversion.
inline char FUNC(quantize_input)(INPUT0_TYPE value, INPUT_QUANTIZATION_TYPE scale, INPUT_QUANTIZATION_TYPE shift) {
    const INPUT_QUANTIZATION_TYPE scaled = TO_INPUT_QUANTIZATION_TYPE(value) * scale + shift;
    return convert_char_sat_rte(scaled + TO_INPUT_QUANTIZATION_TYPE(INPUT_QUANTIZATION_OUTPUT_SHIFT));
}

// Quantizes input row output_y_start + k - 1 into row_cache[slot]. Every position is one aligned
// block read of CHANNEL_BLOCK contiguous channels. Out-of-range positions are simply not loaded, so
// no access can leave the buffer. All the bounds are subgroup-uniform, which also keeps the block
// reads out of divergent control flow. Deliberately a macro and not a helper taking a pointer into
// row_cache: a private pointer would defeat SROA and push the cache into scratch.
#define LOAD_QUANTIZED_ROW(k, slot)                                                                       \
    {                                                                                                     \
        const int load_y = (int)output_y_start + (int)(k) - 1;                                             \
        const bool row_in_bounds = load_y >= 0 && load_y < INPUT0_SIZE_Y;                                  \
        unroll_for(uint p = 0; p < ROW_POSITIONS; ++p) {                                                   \
            const int load_x = (int)output_x_start + (int)p - 1;                                           \
            if (row_in_bounds && load_x >= 0 && load_x < INPUT0_SIZE_X) {                                  \
                const uint load_offset =                                                                   \
                    INPUT0_GET_INDEX(b, channel_block_start, (uint)load_y, (uint)load_x);                  \
                const INPUT_VEC raw = BLOCK_READN(INPUT0_TYPE, CHANNELS_PER_LANE, input, load_offset);     \
                row_cache[slot][p][0] = FUNC_CALL(quantize_input)(raw.s0, scales[0], shifts[0]);           \
                row_cache[slot][p][1] = FUNC_CALL(quantize_input)(raw.s1, scales[1], shifts[1]);           \
            } else {                                                                                      \
                row_cache[slot][p][0] = (char)0;                                                           \
                row_cache[slot][p][1] = (char)0;                                                           \
            }                                                                                             \
        }                                                                                                 \
    }

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, SUBGROUPS_PER_WORK_GROUP, 1))) KERNEL(convolution)(
    OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    ,
    const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    ,
    FUSED_OPS_DECLS
#endif
    ,
    const __global INPUT1_TYPE* input_scale,
    const __global INPUT2_TYPE* input_shift) {
    const uint lane = get_sub_group_local_id();
    const uint output_x_start = (uint)get_group_id(0) * TILE_X;
    const uint output_y_start = (uint)get_global_id(1) * TILE_Y;
    if (output_y_start >= INPUT0_SIZE_Y)
        return;

    const uint batch_channel_block = (uint)get_group_id(2);
    const uint channel_blocks = INPUT0_FEATURE_NUM / CHANNEL_BLOCK;
    const uint b = batch_channel_block / channel_blocks;
    const uint channel_block_start = (batch_channel_block % channel_blocks) * CHANNEL_BLOCK;

    uint channels[CHANNELS_PER_LANE];
    INPUT_QUANTIZATION_TYPE scales[CHANNELS_PER_LANE];
    INPUT_QUANTIZATION_TYPE shifts[CHANNELS_PER_LANE];
    FILTER_TYPE4 weights_block0[CHANNELS_PER_LANE];
    FILTER_TYPE4 weights_block1[CHANNELS_PER_LANE];
    FILTER_TYPE weights_tail[CHANNELS_PER_LANE];
#if BIAS_TERM
    float bias_values[CHANNELS_PER_LANE];
#endif
    char row_cache[ROW_SLOTS][ROW_POSITIONS][CHANNELS_PER_LANE];

#pragma unroll
    for (uint c = 0; c < CHANNELS_PER_LANE; ++c) {
        const uint channel = channel_block_start + c * SUB_GROUP_SIZE + lane;
        channels[c] = channel;

        scales[c] = TO_INPUT_QUANTIZATION_TYPE(input_scale[INPUT1_GET_INDEX_SAFE(0, channel, 0, 0)]);
        shifts[c] = TO_INPUT_QUANTIZATION_TYPE(input_shift[INPUT2_GET_INDEX_SAFE(0, channel, 0, 0)]);
        weights_block0[c] = vload4(0, weights + GET_WEIGHTS_INDEX(channel, 0, 0));
        weights_block1[c] = vload4(0, weights + GET_WEIGHTS_INDEX(channel, 1, 1));
        weights_tail[c] = weights[GET_WEIGHTS_INDEX(channel, 2, 2)];
#if BIAS_TERM
        bias_values[c] = convert_float(biases[BIAS_GET_INDEX(0, channel, 0, 0)]);
#endif
    }

    // Prime the window with the two leading input rows of the stripe.
    LOAD_QUANTIZED_ROW(0, 0);
    LOAD_QUANTIZED_ROW(1, 1);

#pragma unroll
    for (uint row = 0; row < TILE_Y; ++row) {
        // Slide in input row (row + 2), the last one this output row needs. Rows past the stripe
        // resolve to the zero path, so no out-of-range output row issues a memory access.
        LOAD_QUANTIZED_ROW(row + 2, (row + 2) % ROW_SLOTS);

        const uint output_y = output_y_start + row;
        if (output_y >= INPUT0_SIZE_Y)
            continue;

#pragma unroll
        for (uint column = 0; column < TILE_X; ++column) {
            const uint output_x = output_x_start + column;
            if (output_x >= INPUT0_SIZE_X)
                continue;

            OUTPUT_TYPE results[CHANNELS_PER_LANE];
#pragma unroll
            for (uint c = 0; c < CHANNELS_PER_LANE; ++c) {
                // Weights are gs_oi_yxs_gsv16_yxsv4, so the two vload4 groups hold filter taps
                // (0,0) (0,1) (0,2) (1,0) and (1,1) (1,2) (2,0) (2,1), with (2,2) left as the tail.
                const uint slot0 = (row + 0) % ROW_SLOTS;
                const uint slot1 = (row + 1) % ROW_SLOTS;
                const uint slot2 = (row + 2) % ROW_SLOTS;
                const char4 input_block0 = (char4)(row_cache[slot0][column + 0][c],
                                                   row_cache[slot0][column + 1][c],
                                                   row_cache[slot0][column + 2][c],
                                                   row_cache[slot1][column + 0][c]);
                const char4 input_block1 = (char4)(row_cache[slot1][column + 1][c],
                                                   row_cache[slot1][column + 2][c],
                                                   row_cache[slot2][column + 0][c],
                                                   row_cache[slot2][column + 1][c]);
                const char input_tail = row_cache[slot2][column + 2][c];

                int accumulator = IMAD(0, input_block0, weights_block0[c]);
                accumulator = IMAD(accumulator, input_block1, weights_block1[c]);
                accumulator += (int)input_tail * (int)weights_tail[c];

                float dequantized = convert_float(accumulator);
#if BIAS_TERM
                dequantized += bias_values[c];
#endif

#if HAS_FUSED_OPS
                {
                    uint fused_ops_f = channels[c];
                    uint fused_ops_y = output_y;
                    uint fused_ops_x = output_x;
                    float fused_ops_in = dequantized;
                    FUSED_OPS;
                    results[c] = FUSED_OPS_RESULT;
                }
#else
                results[c] = TO_OUTPUT_TYPE(dequantized);
#endif
            }

            // Output is BHWC, so channel_block_start + lane and + SUB_GROUP_SIZE + lane are
            // CHANNEL_BLOCK contiguous elements: one aligned block write per output position.
            const uint output_offset = OUTPUT_GET_INDEX(b, output_y, output_x, channel_block_start);
            BLOCK_WRITEN(OUTPUT_TYPE, CHANNELS_PER_LANE, output, output_offset, (OUTPUT_VEC)(results[0], results[1]));
        }
    }
}

#undef FILTER_TYPE4
#undef GET_WEIGHTS_INDEX
#undef INPUT_VEC
#undef OUTPUT_VEC
#undef ROW_SLOTS
#undef ROW_POSITIONS
#undef LOAD_QUANTIZED_ROW
#undef INPUT_QUANTIZATION_TYPE
#undef TO_INPUT_QUANTIZATION_TYPE
