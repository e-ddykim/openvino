// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"

#ifdef DYNAMIC_SCALE_KERNEL_FEATURE_MAX
#if !IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS0, LWS1, LWS2)))
#endif
KERNEL(calc_max_per_feature)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global ACCUMULATOR_TYPE* internal_max
) {
    #if INPUT0_DIMS == 5
        const uint bf = get_global_id(2) / LWS2;     // batch * feature
    #else
        const uint bf = get_global_id(2);     // batch * feature
    #endif
    const uint b = bf / INPUT0_FEATURE_NUM;
    const uint f = bf % INPUT0_FEATURE_NUM;
    #if INPUT0_DIMS == 5
        const uint z_num_workers = LWS2;
    #endif
    const uint y_num_workers = LWS1;
    const uint x_num_workers = LWS0;
    #if INPUT0_DIMS == 5
        const uint z_block_size = INPUT0_SIZE_Z / z_num_workers;
        const uint z_base = get_local_id(2) * z_block_size;
        const uint z_leftover = INPUT0_SIZE_Z - z_num_workers * z_block_size;
    #endif
    const uint y_block_size = INPUT0_SIZE_Y / y_num_workers;
    const uint y_base = get_local_id(1) * y_block_size;
    const uint y_leftover = INPUT0_SIZE_Y - y_num_workers * y_block_size;

    const uint x_block_size = INPUT0_SIZE_X / x_num_workers;
    const uint x_base = get_local_id(0);
    const uint x_leftover = INPUT0_SIZE_X - x_num_workers * x_block_size;

    ACCUMULATOR_TYPE max = ACCUMULATOR_VAL_ZERO;

    #if INPUT0_DIMS == 5
        for (uint z = z_base; z < (z_base + z_block_size); ++z) {
    #endif
            for (uint y = y_base; y < (y_base + y_block_size); ++y) {
                #if INPUT0_DIMS == 5
                    uint my_data_offset = INPUT0_GET_INDEX(b, f, z, y, x_base);
                #else
                    uint my_data_offset = INPUT0_GET_INDEX(b, f, y, x_base);
                #endif
                for (uint i = 0; i < x_block_size; ++i) {
                    max = ACCUMULATOR_MAX_FUNC(max, TO_ACCUMULATOR_TYPE(input[my_data_offset + i * x_num_workers]));
                }
            }
    #if INPUT0_DIMS == 5
        }
    #endif

    #if INPUT0_DIMS == 5
        if (get_local_id(2) < z_leftover) {
            for (uint y = y_base; y < (y_base + y_block_size); ++y) {
                uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(2) + z_num_workers * z_block_size), y, x_base);
                for (uint i = 0; i < x_block_size; ++i) {
                    max = ACCUMULATOR_MAX_FUNC(max, TO_ACCUMULATOR_TYPE(input[my_data_offset + i * x_num_workers]));
                }
            }
        }
    #endif

    if (get_local_id(1) < y_leftover) {
        #if INPUT0_DIMS == 5
            for (uint z = z_base; z < (z_base + z_block_size); ++z) {
                uint my_data_offset = INPUT0_GET_INDEX(b, f, z, (get_local_id(1) + y_num_workers * y_block_size), x_base);
        #else
                uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(1) + y_num_workers * y_block_size), x_base);
        #endif
                for (uint i = 0; i < x_block_size; ++i) {
                    max = ACCUMULATOR_MAX_FUNC(max, TO_ACCUMULATOR_TYPE(input[my_data_offset + i * x_num_workers]));
                }
        #if INPUT0_DIMS == 5
            }
        #endif
    }

    if (get_local_id(0) < x_leftover) {
        #if INPUT0_DIMS == 5
            for (uint z = z_base; z < (z_base + z_block_size); ++z) {
        #endif
                for (uint y = y_base; y < (y_base + y_block_size); ++y) {
                    #if INPUT0_DIMS == 5
                        uint my_data_offset = INPUT0_GET_INDEX(b, f, z, y, (get_local_id(0) + x_num_workers * x_block_size));
                    #else
                        uint my_data_offset = INPUT0_GET_INDEX(b, f, y, (get_local_id(0) + x_num_workers * x_block_size));
                    #endif
                    max = ACCUMULATOR_MAX_FUNC(max, TO_ACCUMULATOR_TYPE(input[my_data_offset]));
                }
        #if INPUT0_DIMS == 5
            }
        #endif
    }

    #if INPUT0_DIMS == 5
        if (get_local_id(2) < z_leftover && get_local_id(1) < y_leftover && get_local_id(0) < x_leftover) {
            uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(2) + z_num_workers * z_block_size),
                                                         (get_local_id(1) + y_num_workers * y_block_size),
                                                         (get_local_id(0) + x_num_workers * x_block_size));
    #else
        if (get_local_id(1) < y_leftover && get_local_id(0) < x_leftover) {
            uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(1) + y_num_workers * y_block_size),
                                                         (get_local_id(0) + x_num_workers * x_block_size));
    #endif
            max = ACCUMULATOR_MAX_FUNC(max, TO_ACCUMULATOR_TYPE(input[my_data_offset]));
        }

    #if INPUT0_DIMS == 5
        const uint num_local_workers = z_num_workers * y_num_workers * x_num_workers;
    #else
        const uint num_local_workers = y_num_workers * x_num_workers;
    #endif
    const uint worker_idx = get_local_linear_id();

    max = work_group_reduce_max(max);

    if (worker_idx == 0) {
        internal_max[bf] = max;
    }
}
#elif DYNAMIC_SCALE_KERNEL_BATCH_MAX
#if !IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS0, LWS1, LWS2)))
#endif
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
#if !IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS0, LWS1, LWS2)))
#endif
KERNEL(dynamic_scale_bfyx)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* restrict output,
    const __global INPUT0_TYPE* restrict output_scale
) {
    const uint bf = get_global_id(1);
    const uint b = bf / OUTPUT_FEATURE_NUM;
    const uint f = bf % OUTPUT_FEATURE_NUM;
    #if INPUT0_DIMS == 5
        const uint zyx = get_global_id(0);
        const uint z = zyx / (OUTPUT_SIZE_Y * OUTPUT_SIZE_X);
        const uint yx = zyx % (OUTPUT_SIZE_Y * OUTPUT_SIZE_X);
    #else
        const uint yx = get_global_id(0);
    #endif
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;

    #if INPUT0_DIMS == 5
        const uint input_data_index = INPUT0_GET_INDEX(b, f, z, y, x);
    #else
        const uint input_data_index = INPUT0_GET_INDEX(b, f, y, x);
    #endif

    #if INPUT0_DIMS == 5
        const uint output_data_index = OUTPUT_GET_INDEX(b, f, z, y, x);
    #else
        const uint output_data_index = OUTPUT_GET_INDEX(b, f, y, x);
    #endif

    INPUT0_TYPE res = input[input_data_index] / output_scale[b];
    output[output_data_index] = TO_OUTPUT_TYPE(res);
}
#endif
