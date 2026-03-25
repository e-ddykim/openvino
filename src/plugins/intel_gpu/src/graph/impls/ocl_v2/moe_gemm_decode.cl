// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#define unroll_for __attribute__((opencl_unroll_hint)) for

// Fake group size for compatibility and computation performance balance
#define FAKE_GROUP_SIZE 64

#if WEIGHT_COMPRESSEION_DT == 0

#ifdef IS_WEIGHT_SIGNED
#define UNPACK_LO(v) convert_half(((v) & 0x08) ? ((int)((v) & 0x0F) - 16) : (int)((v) & 0x0F))
#define UNPACK_HI(v) convert_half(((v) & 0x80) ? ((int)(((v) >> 4) & 0x0F) - 16) : (int)(((v) >> 4) & 0x0F))
#else
#define UNPACK_LO(v) convert_half((v) & 0x0F)
#define UNPACK_HI(v) convert_half((v) >> 4)
#endif

inline void down_gemv_n2x_u4(const __global uchar* weight,
                             __global half* scales,
                             __global uchar* zps,
                             __global half* bias,
                             __global half* y,
                             int N,
                             int K,
                             half* x2,
                             float* xg_sum) {
    int id_local = get_sub_group_local_id();
    int expert_no = get_global_id(0);
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K / 2;
#    ifdef WEIGHT_ZP_DT
        __global uchar* Z = zps + n / 2;
#    endif
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int g_base = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE);
            half s0 = scales[n * NUM_GROUPS + g_base];
            half s1 = scales[(n + 1) * NUM_GROUPS + g_base];
#    ifdef WEIGHT_ZP_DT
            int zp_offset = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N / 2;
            ushort z = Z[zp_offset];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);
#    endif

#    if SUBGROUP_SIZE == 32
            // us2 reads 2*32=64 halfs = FAKE_GROUP_SIZE (no OOB)
            // uc reads 1*32=32 bytes = FAKE_GROUP_SIZE/2 u4 values (no OOB)
            half2 a = as_half2(intel_sub_group_block_read_us2((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar b = intel_sub_group_block_read_uc((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar b2 = intel_sub_group_block_read_uc((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            // Per-lane quantization group (handles DOWN_GROUP_SIZE <= FAKE_GROUP_SIZE)
            int lane_global_k = gk * FAKE_GROUP_SIZE + id_local * 2;
            int qg = lane_global_k / DOWN_GROUP_SIZE;
            half s0_lane = scales[n * NUM_GROUPS + qg];
            half s1_lane = scales[(n + 1) * NUM_GROUPS + qg];

            // a.s0 = even activation, a.s1 = odd activation
            // b low nibble = even weight, b high nibble = odd weight
            float dot0 = fma((float)a.s0, (float)(UNPACK_LO(b)), 0.0f);
            dot0 = fma((float)a.s1, (float)(UNPACK_HI(b)), dot0);

            float dot1 = fma((float)a.s0, (float)(UNPACK_LO(b2)), 0.0f);
            dot1 = fma((float)a.s1, (float)(UNPACK_HI(b2)), dot1);

#    ifdef WEIGHT_ZP_DT
            ushort z_lane = Z[qg * N / 2];
            half z_hf0_lane = convert_half(z_lane & 0xf);
            half z_hf1_lane = convert_half(z_lane >> 4);
            float x_lane_sum = (float)(a.s0 + a.s1);
            sum_all0 += (dot0 - x_lane_sum * z_hf0_lane) * s0_lane;
            sum_all1 += (dot1 - x_lane_sum * z_hf1_lane) * s1_lane;
#    else
            sum_all0 += dot0 * s0_lane;
            sum_all1 += dot1 * s1_lane;
#    endif
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (UNPACK_LO(b.s0)), 0);
            sum0.s1 = fma(a.s1, (UNPACK_LO(b.s1)), 0);
            sum0.s2 = fma(a.s2, (UNPACK_LO(b.s2)), 0);
            sum0.s3 = fma(a.s3, (UNPACK_LO(b.s3)), 0);

            sum0.s0 = fma(a.s4, (UNPACK_HI(b.s0)), sum0.s0);
            sum0.s1 = fma(a.s5, (UNPACK_HI(b.s1)), sum0.s1);
            sum0.s2 = fma(a.s6, (UNPACK_HI(b.s2)), sum0.s2);
            sum0.s3 = fma(a.s7, (UNPACK_HI(b.s3)), sum0.s3);

            sum1.s0 = fma(a.s0, (UNPACK_LO(b2.s0)), 0);
            sum1.s1 = fma(a.s1, (UNPACK_LO(b2.s1)), 0);
            sum1.s2 = fma(a.s2, (UNPACK_LO(b2.s2)), 0);
            sum1.s3 = fma(a.s3, (UNPACK_LO(b2.s3)), 0);

            sum1.s0 = fma(a.s4, (UNPACK_HI(b2.s0)), sum1.s0);
            sum1.s1 = fma(a.s5, (UNPACK_HI(b2.s1)), sum1.s1);
            sum1.s2 = fma(a.s6, (UNPACK_HI(b2.s2)), sum1.s2);
            sum1.s3 = fma(a.s7, (UNPACK_HI(b2.s3)), sum1.s3);

#    ifdef WEIGHT_ZP_DT
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#    else
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#    endif
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        #ifdef BIAS_DT
            sum_all0 += bias[n];
            sum_all1 += bias[n + 1];
        #endif
        if (id_local == 0) {
            y[n] = sum_all0;
            y[n + 1] = sum_all1;
        }
    }
}

#elif WEIGHT_COMPRESSEION_DT == 1
inline void down_gemv_n2x_u8(const __global uchar* weight,
                             __global half* scales,
                             __global uchar* zps,
                             __global MOE_DTYPE* routing_weights,
                             __global half* y,
                             int N,
                             int K,
                             half* x2,
                             float* xg_sum) {
    int id_local = get_sub_group_local_id();
    int expert_no = get_global_id(0);
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K;
        __global half* S = scales + n;
        __global uchar* Z = zps + n;
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            int zp_offset = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];
            half z0 = convert_half(Z[zp_offset]);
            half z1 = convert_half(Z[zp_offset + 1]);

#    if SUBGROUP_SIZE == 32
            float2 sum0;
            float2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)B + K + gk * FAKE_GROUP_SIZE);

            sum0.s0 = fma((float)a.s0, (float)(convert_half(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(convert_half(b.s1)), 0.0f);
            sum0.s0 = fma((float)a.s2, (float)(convert_half(b.s2)), sum0.s0);
            sum0.s1 = fma((float)a.s3, (float)(convert_half(b.s3)), sum0.s1);

            sum1.s0 = fma((float)a.s0, (float)(convert_half(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(convert_half(b2.s1)), 0.0f);
            sum1.s0 = fma((float)a.s2, (float)(convert_half(b2.s2)), sum1.s0);
            sum1.s1 = fma((float)a.s3, (float)(convert_half(b2.s3)), sum1.s1);

            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z1) * s1;
#    else
            float4 sum0;
            float4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar8 b = intel_sub_group_block_read_uc8((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar8 b2 = intel_sub_group_block_read_uc8((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma((float)a.s0, (float)(convert_half(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(convert_half(b.s1)), 0.0f);
            sum0.s2 = fma((float)a.s2, (float)(convert_half(b.s2)), 0.0f);
            sum0.s3 = fma((float)a.s3, (float)(convert_half(b.s3)), 0.0f);

            sum0.s0 = fma((float)a.s4, (float)(convert_half(b.s4)), sum0.s0);
            sum0.s1 = fma((float)a.s5, (float)(convert_half(b.s5)), sum0.s1);
            sum0.s2 = fma((float)a.s6, (float)(convert_half(b.s6)), sum0.s2);
            sum0.s3 = fma((float)a.s7, (float)(convert_half(b.s7)), sum0.s3);

            sum1.s0 = fma((float)a.s0, (float)(convert_half(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(convert_half(b2.s1)), 0.0f);
            sum1.s2 = fma((float)a.s2, (float)(convert_half(b2.s2)), 0.0f);
            sum1.s3 = fma((float)a.s3, (float)(convert_half(b2.s3)), 0.0f);

            sum1.s0 = fma((float)a.s4, (float)(convert_half(b2.s4)), sum1.s0);
            sum1.s1 = fma((float)a.s5, (float)(convert_half(b2.s5)), sum1.s1);
            sum1.s2 = fma((float)a.s6, (float)(convert_half(b2.s6)), sum1.s2);
            sum1.s3 = fma((float)a.s7, (float)(convert_half(b2.s7)), sum1.s3);

            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z1) * s1;
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weights[expert_no];
            y[n + 1] = sum_all1 * routing_weights[expert_no];
        }
    }
}

#elif WEIGHT_COMPRESSEION_DT == 2
inline void down_gemv_n2x_f16(const __global half* weight, __global MOE_DTYPE* routing_weights, __global half* y, int N, int K, half* x2) {
    int id_local = get_sub_group_local_id();
    int expert_no = get_global_id(0);
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global half* B = weight + n * K;
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {

#    if SUBGROUP_SIZE == 32
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half4 b = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half4 b2 = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s0 = fma(a.s2, b.s2, sum0.s0);
            sum0.s1 = fma(a.s3, b.s3, sum0.s1);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s0 = fma(a.s2, b2.s2, sum1.s0);
            sum1.s1 = fma(a.s3, b2.s3, sum1.s1);

            sum_all0 += sum0[0] + sum0[1];
            sum_all1 += sum1[0] + sum1[1];
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half8 b = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half8 b2 = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s2 = fma(a.s2, b.s2, 0);
            sum0.s3 = fma(a.s3, b.s3, 0);

            sum0.s0 = fma(a.s4, b.s4, sum0.s0);
            sum0.s1 = fma(a.s5, b.s5, sum0.s1);
            sum0.s2 = fma(a.s6, b.s6, sum0.s2);
            sum0.s3 = fma(a.s7, b.s7, sum0.s3);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s2 = fma(a.s2, b2.s2, 0);
            sum1.s3 = fma(a.s3, b2.s3, 0);

            sum1.s0 = fma(a.s4, b2.s4, sum1.s0);
            sum1.s1 = fma(a.s5, b2.s5, sum1.s1);
            sum1.s2 = fma(a.s6, b2.s6, sum1.s2);
            sum1.s3 = fma(a.s7, b2.s7, sum1.s3);

            sum_all0 += sum0[0] + sum0[1] + sum0[2] + sum0[3];
            sum_all1 += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weights[expert_no];
            y[n + 1] = sum_all1 * routing_weights[expert_no];
        }
    }
}
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(moe_gemm_decode)(const global INPUT0_TYPE *input_ptr,
                                                                                #ifdef WEIGHT_COMPRESSED_INT4
                                                                                    const global uchar *weight_ptr,
                                                                                #else
                                                                                    const global INPUT1_TYPE *weight_ptr,
                                                                                #endif
                                                                                global OUTPUT_TYPE *out_ptr,
                                                                                const global INPUT2_TYPE *experts_ids
                                                                                #ifdef BIAS_DT
                                                                                    , const global BIAS_DT *bias_ptr
                                                                                #endif
                                                                                #ifdef WEIGHT_COMPRESSED_INT4
                                                                                    , const global WEIGHT_SCALE_DT *weight_scales
                                                                                    #ifdef WEIGHT_ZP_DT
                                                                                    , const global WEIGHT_ZP_DT *weight_zps
                                                                                    #endif
                                                                                #endif
) {
    #if IS_UP_PHASE
        #define OFM INTERMEDIATE_SIZE
    #else
        #define OFM HIDDEN_SIZE
    #endif

    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    #if !IS_UP_PHASE
        input_ptr += expert_no * HIDDEN_SIZE;
    #endif
    out_ptr += expert_no * OFM;

    #if WEIGHT_COMPRESSEION_DT == 0
        const int expert_wei_size = OFM * HIDDEN_SIZE / 2;
        const int expert_scale_size = OFM * HIDDEN_SIZE / DOWN_GROUP_SIZE;
        const int expert_zp_size = OFM * HIDDEN_SIZE / 2 / DOWN_GROUP_SIZE;
    #else
        const int expert_wei_size = OFM * HIDDEN_SIZE;
        const int expert_scale_size = OFM * HIDDEN_SIZE / DOWN_GROUP_SIZE;
        const int expert_zp_size = OFM * HIDDEN_SIZE / DOWN_GROUP_SIZE;
    #endif
    int expert_id = experts_ids[expert_no];

    // down, [OFM, HIDDEN_SIZE]
    __global MOE_WEI_DT* weight = (__global MOE_WEI_DT*)(weight_ptr + expert_id * expert_wei_size);
    __global MOE_SCALE_DT* scales = NULL;
    __global MOE_ZP_DT* zps = NULL;
    #ifdef WEIGHT_COMPRESSED_INT4
        scales = (__global MOE_SCALE_DT*)(weight_scales + expert_id * expert_scale_size);
        #ifdef WEIGHT_ZP_DT
            zps = (__global MOE_ZP_DT*)(weight_zps + expert_id * expert_zp_size);
        #endif
    #endif
    __global OUTPUT_TYPE* bias = NULL;
    #ifdef BIAS_DT
        bias = (__global OUTPUT_TYPE*)(bias_ptr + expert_id * OFM);
    #endif

    int N = OFM;
    int K = HIDDEN_SIZE;

    __local half x2[HIDDEN_SIZE];
    __local float xg_sum[HIDDEN_SIZE / FAKE_GROUP_SIZE];

#    if WEIGHT_COMPRESSEION_DT == 0
    //# interleaving x into x2
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = input_ptr + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < K / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE / 2; j += SUBGROUP_SIZE) {
            half even = px[2 * j + 0];
            half odd = px[2 * j + 1];
            px2[j] = even;
            px2[j + FAKE_GROUP_SIZE / 2] = odd;
            x_group_sum += even + odd;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }
#    else
    //# load x into slm
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = input_ptr + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < K / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE; j += SUBGROUP_SIZE) {
            half value = px[j];
            px2[j] = value;
            x_group_sum += value;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }
#    endif

    barrier(CLK_LOCAL_MEM_FENCE);

#    if WEIGHT_COMPRESSEION_DT == 0
    down_gemv_n2x_u4(weight, scales, zps, bias, out_ptr, N, K, x2, xg_sum);
#    elif WEIGHT_COMPRESSEION_DT == 1
    down_gemv_n2x_u8(weight, scales, zps, routing_weights, out_ptr, N, K, x2, xg_sum);
#    elif WEIGHT_COMPRESSEION_DT == 2
    down_gemv_n2x_f16(weight, routing_weights, out_ptr, N, K, x2);
#    endif
}
