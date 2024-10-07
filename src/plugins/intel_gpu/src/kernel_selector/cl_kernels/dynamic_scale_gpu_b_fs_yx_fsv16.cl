// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"

#ifdef DYNAMIC_SCALE_KERNEL_FEATURE_MAX
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(calc_max_per_feature)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global ACCUMULATOR_TYPE* internal_max
) {
    const uint data_set_idx = get_global_id(1);     // batch * feature split
    const uint in_data_set_idx = get_global_id(0);
    const uint workers_per_dataset = LWS0 / FSV;    // 16 datasets are handled by one local workgroup
    const uint data_set_size = INPUT0_SIZE_X * INPUT0_SIZE_Y;
    const uint items_num = data_set_size / workers_per_dataset;
    const uint leftovers = data_set_size - (items_num * workers_per_dataset);

    const uint INPUT0_ALIGNED_FEATURE_NUM = ALIGN(INPUT0_FEATURE_NUM, FSV);
    const uint b = (data_set_idx * FSV) / INPUT0_ALIGNED_FEATURE_NUM;
    const uint f_base = (data_set_idx * FSV) % INPUT0_ALIGNED_FEATURE_NUM;
    const uint data_set_offset = INPUT0_GET_INDEX(b, f_base, 0, 0);
    const uint my_data_offset = data_set_offset + in_data_set_idx;

    __local ACCUMULATOR_TYPE max_per_feature[SLM_SIZE];

    ACCUMULATOR_TYPE max = ACCUMULATOR_VAL_ZERO;

    for (uint i = 0; i < items_num; ++i) {
        max = ACCUMULATOR_MAX_FUNC(max, TO_ACCUMULATOR_TYPE(input[my_data_offset + i * workers_per_dataset * FSV]));
    }

    if (in_data_set_idx < leftovers) {
        max = ACCUMULATOR_MAX_FUNC(max, TO_ACCUMULATOR_TYPE(input[my_data_offset + items_num * workers_per_dataset * FSV + in_data_set_idx]));
    }

    max_per_feature[in_data_set_idx] = max;
    const uint num_local_workers = LWS0;
    const uint worker_block_idx = in_data_set_idx / FSV;
    uint reduce_add_level = 1;
    while ((SLM_SIZE / FSV) > reduce_add_level) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (worker_block_idx % (reduce_add_level * 2) == 0 && (in_data_set_idx + FSV * reduce_add_level) < num_local_workers) {
            max_per_feature[in_data_set_idx] = ACCUMULATOR_MAX_FUNC(max_per_feature[in_data_set_idx], max_per_feature[in_data_set_idx + FSV * reduce_add_level]);
        }
        reduce_add_level *= 2;
    }

    if (worker_block_idx == 0 && (f_base + in_data_set_idx) < INPUT0_FEATURE_NUM) {
        uint bf = b * INPUT0_FEATURE_NUM + f_base + in_data_set_idx;
        internal_max[bf] = max_per_feature[in_data_set_idx];
    }
}
#elif DYNAMIC_SCALE_KERNEL_BATCH_MAX
KERNEL(calc_max_per_batch)(
    const __global OUTPUT_TYPE* restrict output,
    __global INPUT0_TYPE* restrict output_scale,
    __global ACCUMULATOR_TYPE* internal_max
) {
    const uint data_idx = get_global_id(0) + get_global_id(1) * GWS0;
    const uint num_workers = LWS0;
    const uint feature_size = GWS0;
    const uint items_num = feature_size / num_workers;

    if ((data_idx % feature_size) < num_workers) {
        ACCUMULATOR_TYPE my_max = ACCUMULATOR_VAL_ZERO;
        for (uint i = 0; i < items_num; ++i) {
            my_max = ACCUMULATOR_MAX_FUNC(my_max, internal_max[data_idx + num_workers * i]);
        }

        ACCUMULATOR_TYPE max = work_group_reduce_max(my_max);
        for (uint i = 0; i < items_num; ++i) {
            internal_max[data_idx + num_workers * i] = max;
        }

        if (get_global_id(0) == 0) {
            const uint b = get_global_id(1);
            output_scale[b] = TO_INPUT0_TYPE(max / 1.f);
            printf("[%u] internal_max: %f, %f\n", b, max, output_scale[b]);
        }
    }
}
#elif DYNAMIC_SCALE_KERNEL_FINAL
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(dynamic_scale_b_fs_yx_fsv16)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* restrict output,
    const __global INPUT0_TYPE* restrict output_scale
) {
    const uint b = get_global_id(1) % OUTPUT_BATCH_NUM;
    const uint f = get_global_id(1) / OUTPUT_BATCH_NUM * FSV + get_sub_group_local_id();
    const uint yx = get_global_id(0) / FSV;
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
    const uint input_index = INPUT0_GET_INDEX(b, f, y, x);
    const uint output_index = OUTPUT_GET_INDEX(b, f, y, x);

    if (f < OUTPUT_FEATURE_NUM) {
        INPUT0_TYPE res = input[input_index] / output_scale[b];
        output[output_index] = TO_OUTPUT_TYPE(res);
    } else {
        output[output_index] = OUTPUT_VAL_ZERO;
    }
}
#endif
