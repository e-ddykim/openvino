// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable

#define INPUT0_VEC_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
#define INPUT1_COMPUTE_VEC_TYPE MAKE_VECTOR_TYPE(INPUT1_COMPUTE_TYPE, 2)
#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2)
#define TO_VECTOR_TYPE_IMPL_2(elem_type) CAT(convert_##elem_type, 2)
#define TO_VECTOR_TYPE(elem_type, size) CAT(TO_VECTOR_TYPE_IMPL_, size)(elem_type)
#define TO_VECTOR_TYPE_IMPL_SAT_RTE_2(elem_type) CAT(convert_##elem_type, 2##_sat_rte)
#define TO_VECTOR_TYPE_SAT_RTE(elem_type, size) CAT(TO_VECTOR_TYPE_IMPL_SAT_RTE_, size)(elem_type)

#define INPUT_SURFACE_WIDTH 64
#define INPUT_SURFACE_PITCH 128
#define OUTPUT_SURFACE_WIDTH 64
#define OUTPUT_SURFACE_PITCH 64

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
KERNEL(quantize_gpu_scale_shift_2dblk_opt)(OPTIONAL_SHAPE_INFO_ARG
                                           const __global INPUT0_TYPE* input,
                                           const __global INPUT1_TYPE* input_low,
                                           const __global INPUT2_TYPE* input_high,
                                           const __global INPUT3_TYPE* output_low,
                                           const __global INPUT4_TYPE* output_high,
                                           const __global INPUT5_TYPE* input_scale,
                                           const __global INPUT6_TYPE* input_shift,
                                           const __global INPUT7_TYPE* output_scale,
                                           const __global INPUT8_TYPE* output_shift,
                                                 __global OUTPUT_TYPE* output) {
    const uint tile_and_parity = get_group_id(1) * SUBGROUPS_PER_WORK_GROUP + get_sub_group_id();
    if (tile_and_parity >= TILE_COUNT)
        return;

    const uint parity = tile_and_parity & 1;
    const uint tile = tile_and_parity >> 1;
    const uint batch_feature_block = get_group_id(2);
    const uint feature_block = batch_feature_block % FEATURE_BLOCKS;
    const uint feature = feature_block * FEATURE_BLOCK_SIZE + get_sub_group_local_id() * 2;
    const uint pair_row = tile * SPATIAL_TILE_SIZE;
    const int surface_height = min((int)SPATIAL_TILE_SIZE, (int)SPATIAL_PAIR_ROWS - (int)pair_row);

    const size_t data_block_offset = (size_t)batch_feature_block * SPATIAL_SIZE * FEATURE_BLOCK_SIZE;
    const size_t input_offset = data_block_offset + (pair_row * 2 + parity) * FEATURE_BLOCK_SIZE;
    const size_t output_offset = data_block_offset + pair_row * 2 * FEATURE_BLOCK_SIZE;

    uint packed_input[SPATIAL_TILE_SIZE];
    ushort packed_output[SPATIAL_TILE_SIZE];
    intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)(input + input_offset),
                                              INPUT_SURFACE_WIDTH,
                                              surface_height,
                                              INPUT_SURFACE_PITCH,
                                              (int2)(0, 0),
                                              packed_input);

    const INPUT1_COMPUTE_VEC_TYPE input_scale_val = (INPUT1_COMPUTE_VEC_TYPE)(TO_INPUT1_COMPUTE_TYPE(input_scale[INPUT5_GET_INDEX_SAFE(0, feature, 0, 0)]),
                                                                              TO_INPUT1_COMPUTE_TYPE(input_scale[INPUT5_GET_INDEX_SAFE(0, feature + 1, 0, 0)]));
    const INPUT1_COMPUTE_VEC_TYPE input_shift_val = (INPUT1_COMPUTE_VEC_TYPE)(TO_INPUT1_COMPUTE_TYPE(input_shift[INPUT6_GET_INDEX_SAFE(0, feature, 0, 0)]),
                                                                              TO_INPUT1_COMPUTE_TYPE(input_shift[INPUT6_GET_INDEX_SAFE(0, feature + 1, 0, 0)]));

#pragma unroll
    for (uint row = 0; row < SPATIAL_TILE_SIZE; ++row) {
        const INPUT0_VEC_TYPE input_val = as_half2(packed_input[row]);
        INPUT1_COMPUTE_VEC_TYPE val = TO_VECTOR_TYPE(INPUT1_COMPUTE_TYPE, 2)(DECODE_INPUT0_COMPUTE_VECTOR_TYPE(input_val, 2));
        val = val * input_scale_val + input_shift_val;

#if HAS_OUTPUT_RANGE_ROUND
        val = round(val);
#endif
#if HAS_POST_SCALE
        val *= TO_INPUT1_COMPUTE_TYPE(OUT_SCALE_VAL);
#endif
#if HAS_POST_SHIFT
        val += TO_INPUT1_COMPUTE_TYPE(OUT_SHIFT_VAL);
#endif
#if HAS_CLAMP
#    if HAS_MIN_CLAMP && HAS_MAX_CLAMP
        val = clamp(val, TO_INPUT1_COMPUTE_TYPE(OUT_LO_VAL), TO_INPUT1_COMPUTE_TYPE(OUT_HI_VAL));
#    elif HAS_MIN_CLAMP
        val = max(val, TO_INPUT1_COMPUTE_TYPE(OUT_LO_VAL));
#    else
        val = min(val, TO_INPUT1_COMPUTE_TYPE(OUT_HI_VAL));
#    endif
#endif

        const OUTPUT_VEC_TYPE result = TO_VECTOR_TYPE_SAT_RTE(OUTPUT_TYPE, 2)(val);
        packed_output[row] = as_ushort(result);
    }

    intel_sub_group_2d_block_write_8b_8r32x1c((__global void*)(output + output_offset),
                                              OUTPUT_SURFACE_WIDTH,
                                              surface_height,
                                              OUTPUT_SURFACE_PITCH,
                                              (int2)(parity * FEATURE_BLOCK_SIZE, 0),
                                              packed_output);
}

#undef INPUT0_VEC_TYPE
#undef INPUT1_COMPUTE_VEC_TYPE
#undef OUTPUT_VEC_TYPE
#undef TO_VECTOR_TYPE_IMPL_2
#undef TO_VECTOR_TYPE
#undef TO_VECTOR_TYPE_IMPL_SAT_RTE_2
#undef TO_VECTOR_TYPE_SAT_RTE
#undef INPUT_SURFACE_WIDTH
#undef INPUT_SURFACE_PITCH
#undef OUTPUT_SURFACE_WIDTH
#undef OUTPUT_SURFACE_PITCH