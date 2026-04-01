// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#define unroll_for __attribute__((opencl_unroll_hint)) for

// Fake group size for compatibility and computation performance balance
#define FAKE_GROUP_SIZE 64

// #if WEIGHT_COMPRESSEION_DT == 0

#ifdef IS_WEIGHT_SIGNED
#define UNPACK_LO(v) convert_half(((v) & 0x08) ? ((int)((v) & 0x0F) - 16) : (int)((v) & 0x0F))
#define UNPACK_HI(v) convert_half(((v) & 0x80) ? ((int)(((v) >> 4) & 0x0F) - 16) : (int)(((v) >> 4) & 0x0F))
#else
#define UNPACK_LO(v) convert_half((v) & 0x0F)
#define UNPACK_HI(v) convert_half((v) >> 4)
#endif

// inline void down_gemv_n2x_u4(const __global uchar* weight,
//                              __local half* scales,
//                              __global uchar* zps,
//                              __global half* bias,
//                              __global half* y,
//                              int N,
//                              int K,
//                              __local half* x2) {
//     int id_local = get_sub_group_local_id();
//     int expert_no = get_global_id(0);
//     int n_start = get_global_id(2) * N_BLOCK;
//     int n_end = n_start + N_BLOCK;

//     for(int n = n_start; n < n_end; n += 2) {
//         const __global uchar* B = weight + n * K / 2;

//         float sum_all0 = 0;
//         float sum_all1 = 0;
//         for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
//             #if SUBGROUP_SIZE == 32
//                 // us2 reads 2*32=64 halfs = FAKE_GROUP_SIZE (no OOB)
//                 // uc reads 1*32=32 bytes = FAKE_GROUP_SIZE/2 u4 values (no OOB)
//                 half2 a = as_half2(intel_sub_group_block_read_us2((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
//                 uchar b = intel_sub_group_block_read_uc((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
//                 uchar b2 = intel_sub_group_block_read_uc((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

//                 // Per-lane quantization group (handles DOWN_GROUP_SIZE <= FAKE_GROUP_SIZE)
//                 int lane_global_k = gk * FAKE_GROUP_SIZE + id_local * 2;
//                 int qg = lane_global_k / DOWN_GROUP_SIZE;
//                 half s0_lane = scales[(n - n_start) * NUM_GROUPS + qg];
//                 half s1_lane = scales[(n - n_start + 1) * NUM_GROUPS + qg];

//                 // a.s0 = even activation, a.s1 = odd activation
//                 // b low nibble = even weight, b high nibble = odd weight
//                 half dot0 = fma((half)a.s0, (half)(UNPACK_LO(b)), (half)0.0f);
//                 dot0 = fma((half)a.s1, (half)(UNPACK_HI(b)), dot0);

//                 half dot1 = fma((half)a.s0, (half)(UNPACK_LO(b2)), (half)0.0f);
//                 dot1 = fma((half)a.s1, (half)(UNPACK_HI(b2)), dot1);

//                 sum_all0 += dot0 * s0_lane;
//                 sum_all1 += dot1 * s1_lane;
//                 // sum_all0 = fma((float)dot0, (float)s0_lane, sum_all0);
//                 // sum_all1 = fma((float)dot1, (float)s1_lane, sum_all1);
//             #elif SUBGROUP_SIZE == 16
//                 half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
//                 uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
//                 uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));
//                 // half s0 = scales[n * NUM_GROUPS + gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE)];
//                 // half s1 = scales[n * NUM_GROUPS + gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) + 1];
//                 // half s2 = scales[(n + 1) * NUM_GROUPS + gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE)];
//                 // half s3 = scales[(n + 1) * NUM_GROUPS + gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) + 1];
//                 __global half2* scales2 = (__global half2 *)scales;
//                 half2 s0 = sub_group_broadcast(scales2[(n * NUM_GROUPS + gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE)) / 2], 0);
//                 half2 s1 = sub_group_broadcast(scales2[((n + 1) * NUM_GROUPS + gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE)) / 2], 0);

//                 float dot0 = (float)a.s0 * (float)(UNPACK_LO(b.s0));
//                   dot0 = fma((float)a.s2 , (float)(UNPACK_HI(b.s0)), dot0);
//                 float dot1 = (float)a.s1 * (float)(UNPACK_LO(b.s1));
//                   dot1 = fma((float)a.s3 , (float)(UNPACK_HI(b.s1)), dot1);

//                 float dot2 = (float)a.s0 * (float)(UNPACK_LO(b2.s0));
//                   dot2 = fma((float)a.s2 , (float)(UNPACK_HI(b2.s0)), dot2);
//                 float dot3 = (float)a.s1 * (float)(UNPACK_LO(b2.s1));
//                   dot3 = fma((float)a.s3 , (float)(UNPACK_HI(b2.s1)), dot3);

//                 // sum_all0 += dot0 * s0.s0 + dot1 * s0.s1;
//                 // sum_all1 += dot2 * s1.s0 + dot3 * s1.s1;
//                 sum_all0 = fma((float)dot0, (float)s0.s0, sum_all0);
//                 sum_all0 = fma((float)dot1, (float)s0.s1, sum_all0);
//                 sum_all1 = fma((float)dot2, (float)s1.s0, sum_all1);
//                 sum_all1 = fma((float)dot3, (float)s1.s1, sum_all1);
//             #endif
//         }

//         sum_all0 = sub_group_reduce_add(sum_all0);
//         sum_all1 = sub_group_reduce_add(sum_all1);

//         #ifdef BIAS_DT
//             sum_all0 += bias[n];
//             sum_all1 += bias[n + 1];
//         #endif
//         if (id_local == 0) {
//             y[n] = sum_all0;
//             y[n + 1] = sum_all1;
//         }
//     }
// }

// #elif WEIGHT_COMPRESSEION_DT == 1
// inline void down_gemv_n2x_u8(const __global uchar* weight,
//                              __global half* scales,
//                              __global uchar* zps,
//                              __global MOE_DTYPE* routing_weights,
//                              __global half* y,
//                              int N,
//                              int K,
//                              half* x2,
//                              float* xg_sum) {
//     int id_local = get_sub_group_local_id();
//     int expert_no = get_global_id(0);
//     int n_start = get_global_id(2) * N_BLOCK;
//     int n_end = n_start + N_BLOCK;

//     unroll_for(int n = n_start; n < n_end; n += 2) {
//         const __global uchar* B = weight + n * K;
//         __global half* S = scales + n;
//         __global uchar* Z = zps + n;
//         float sum_all0 = 0;
//         float sum_all1 = 0;
//         unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
//             int scale_offset = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
//             int zp_offset = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
//             half s0 = S[scale_offset];
//             half s1 = S[scale_offset + 1];
//             half z0 = convert_half(Z[zp_offset]);
//             half z1 = convert_half(Z[zp_offset + 1]);

// #    if SUBGROUP_SIZE == 32
//             float2 sum0;
//             float2 sum1;
//             half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
//             uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
//             uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)B + K + gk * FAKE_GROUP_SIZE);

//             sum0.s0 = fma((float)a.s0, (float)(convert_half(b.s0)), 0.0f);
//             sum0.s1 = fma((float)a.s1, (float)(convert_half(b.s1)), 0.0f);
//             sum0.s0 = fma((float)a.s2, (float)(convert_half(b.s2)), sum0.s0);
//             sum0.s1 = fma((float)a.s3, (float)(convert_half(b.s3)), sum0.s1);

//             sum1.s0 = fma((float)a.s0, (float)(convert_half(b2.s0)), 0.0f);
//             sum1.s1 = fma((float)a.s1, (float)(convert_half(b2.s1)), 0.0f);
//             sum1.s0 = fma((float)a.s2, (float)(convert_half(b2.s2)), sum1.s0);
//             sum1.s1 = fma((float)a.s3, (float)(convert_half(b2.s3)), sum1.s1);

//             sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z0) * s0;
//             sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z1) * s1;
// #    else
//             float4 sum0;
//             float4 sum1;
//             half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
//             uchar8 b = intel_sub_group_block_read_uc8((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
//             uchar8 b2 = intel_sub_group_block_read_uc8((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

//             sum0.s0 = fma((float)a.s0, (float)(convert_half(b.s0)), 0.0f);
//             sum0.s1 = fma((float)a.s1, (float)(convert_half(b.s1)), 0.0f);
//             sum0.s2 = fma((float)a.s2, (float)(convert_half(b.s2)), 0.0f);
//             sum0.s3 = fma((float)a.s3, (float)(convert_half(b.s3)), 0.0f);

//             sum0.s0 = fma((float)a.s4, (float)(convert_half(b.s4)), sum0.s0);
//             sum0.s1 = fma((float)a.s5, (float)(convert_half(b.s5)), sum0.s1);
//             sum0.s2 = fma((float)a.s6, (float)(convert_half(b.s6)), sum0.s2);
//             sum0.s3 = fma((float)a.s7, (float)(convert_half(b.s7)), sum0.s3);

//             sum1.s0 = fma((float)a.s0, (float)(convert_half(b2.s0)), 0.0f);
//             sum1.s1 = fma((float)a.s1, (float)(convert_half(b2.s1)), 0.0f);
//             sum1.s2 = fma((float)a.s2, (float)(convert_half(b2.s2)), 0.0f);
//             sum1.s3 = fma((float)a.s3, (float)(convert_half(b2.s3)), 0.0f);

//             sum1.s0 = fma((float)a.s4, (float)(convert_half(b2.s4)), sum1.s0);
//             sum1.s1 = fma((float)a.s5, (float)(convert_half(b2.s5)), sum1.s1);
//             sum1.s2 = fma((float)a.s6, (float)(convert_half(b2.s6)), sum1.s2);
//             sum1.s3 = fma((float)a.s7, (float)(convert_half(b2.s7)), sum1.s3);

//             sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z0) * s0;
//             sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z1) * s1;
// #    endif
//         }
//         sum_all0 = sub_group_reduce_add(sum_all0);
//         sum_all1 = sub_group_reduce_add(sum_all1);
//         if (id_local == 0) {
//             y[n] = sum_all0 * routing_weights[expert_no];
//             y[n + 1] = sum_all1 * routing_weights[expert_no];
//         }
//     }
// }

// #elif WEIGHT_COMPRESSEION_DT == 2
// inline void down_gemv_n2x_f16(const __global half* weight, __global MOE_DTYPE* routing_weights, __global half* y, int N, int K, half* x2) {
//     int id_local = get_sub_group_local_id();
//     int expert_no = get_global_id(0);
//     int n_start = get_global_id(2) * N_BLOCK;
//     int n_end = n_start + N_BLOCK;

//     unroll_for(int n = n_start; n < n_end; n += 2) {
//         const __global half* B = weight + n * K;
//         float sum_all0 = 0;
//         float sum_all1 = 0;
//         unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {

// #    if SUBGROUP_SIZE == 32
//             half2 sum0;
//             half2 sum1;
//             half4 a = as_half4(intel_sub_group_block_read_us4((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
//             half4 b = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
//             half4 b2 = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

//             sum0.s0 = fma(a.s0, b.s0, 0);
//             sum0.s1 = fma(a.s1, b.s1, 0);
//             sum0.s0 = fma(a.s2, b.s2, sum0.s0);
//             sum0.s1 = fma(a.s3, b.s3, sum0.s1);

//             sum1.s0 = fma(a.s0, b2.s0, 0);
//             sum1.s1 = fma(a.s1, b2.s1, 0);
//             sum1.s0 = fma(a.s2, b2.s2, sum1.s0);
//             sum1.s1 = fma(a.s3, b2.s3, sum1.s1);

//             sum_all0 += sum0[0] + sum0[1];
//             sum_all1 += sum1[0] + sum1[1];
// #    else
//             half4 sum0;
//             half4 sum1;
//             half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
//             half8 b = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
//             half8 b2 = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

//             sum0.s0 = fma(a.s0, b.s0, 0);
//             sum0.s1 = fma(a.s1, b.s1, 0);
//             sum0.s2 = fma(a.s2, b.s2, 0);
//             sum0.s3 = fma(a.s3, b.s3, 0);

//             sum0.s0 = fma(a.s4, b.s4, sum0.s0);
//             sum0.s1 = fma(a.s5, b.s5, sum0.s1);
//             sum0.s2 = fma(a.s6, b.s6, sum0.s2);
//             sum0.s3 = fma(a.s7, b.s7, sum0.s3);

//             sum1.s0 = fma(a.s0, b2.s0, 0);
//             sum1.s1 = fma(a.s1, b2.s1, 0);
//             sum1.s2 = fma(a.s2, b2.s2, 0);
//             sum1.s3 = fma(a.s3, b2.s3, 0);

//             sum1.s0 = fma(a.s4, b2.s4, sum1.s0);
//             sum1.s1 = fma(a.s5, b2.s5, sum1.s1);
//             sum1.s2 = fma(a.s6, b2.s6, sum1.s2);
//             sum1.s3 = fma(a.s7, b2.s7, sum1.s3);

//             sum_all0 += sum0[0] + sum0[1] + sum0[2] + sum0[3];
//             sum_all1 += sum1[0] + sum1[1] + sum1[2] + sum1[3];
// #    endif
//         }
//         sum_all0 = sub_group_reduce_add(sum_all0);
//         sum_all1 = sub_group_reduce_add(sum_all1);
//         if (id_local == 0) {
//             y[n] = sum_all0 * routing_weights[expert_no];
//             y[n + 1] = sum_all1 * routing_weights[expert_no];
//         }
//     }
// }
// #endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(moe_gemm_decode)(OPTIONAL_SHAPE_INFO_ARG
                        const global INPUT0_TYPE *input_ptr,
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
        #define N INTERMEDIATE_SIZE
    #else
        #define N HIDDEN_SIZE
    #endif
    #define K HIDDEN_SIZE

    // gws = {(N / N_BLOCK * subgroup_size), (K / subgroup_size / K_BLOCK), desc->moe_config.top_k}
    // lws = {               subgroup_size , (K / subgroup_size / K_BLOCK),                      1}
    int expert_no = get_global_id(2);
    input_ptr += expert_no * K;
    out_ptr += expert_no * N;

    #if WEIGHT_COMPRESSEION_DT == 0
        const int expert_wei_size = N * K / 2;
        const int expert_scale_size = N * K / DOWN_GROUP_SIZE;
        const int expert_zp_size = N * K / 2 / DOWN_GROUP_SIZE;
    #else
        const int expert_wei_size = N * K;
        const int expert_scale_size = N * K / DOWN_GROUP_SIZE;
        const int expert_zp_size = N * K / DOWN_GROUP_SIZE;
    #endif
    int expert_id = experts_ids[expert_no];

    __global MOE_WEI_DT* weight = (__global MOE_WEI_DT*)(weight_ptr + expert_id * expert_wei_size);
    #ifdef WEIGHT_COMPRESSED_INT4
        __global MOE_SCALE_DT* scales = (__global MOE_SCALE_DT*)(weight_scales + expert_id * expert_scale_size);
        // const __global half2* wei_scales2 = (const __global half2*)scales;
        #ifdef WEIGHT_ZP_DT
            __global MOE_ZP_DT* zps = (__global MOE_ZP_DT*)(weight_zps + expert_id * expert_zp_size);
        #endif
    #endif
    #ifdef BIAS_DT
        __global BIAS_DT* bias = (__global BIAS_DT*)(bias_ptr + expert_id * N);
    #endif

    uint sgid = get_sub_group_id();
    const uint num_sg = K / SUBGROUP_SIZE / K_BLOCK; // should be equal to get_num_sub_groups()
    uint sglid = get_sub_group_local_id();

    uint k_start = sgid * (K / num_sg);
    uint k_end = k_start + (K / num_sg);

    uint n_start = get_global_id(0) / SUBGROUP_SIZE * N_BLOCK;

    // if (sgid == 0 && sglid == 0) {
    //     printf("k_start=%u, k_end=%u, n_start=%u\n", k_start, k_end, n_start);
    // }

    __local float out_local_buf[N_BLOCK * num_sg];
    float out_private_buf[N_BLOCK] = {0.0f};

	for (uint idx_k = k_start; idx_k < k_end; idx_k += SUBGROUP_SIZE * 4) {
		half4 act4 = as_half4(intel_sub_group_block_read2((const global uint*)input_ptr + idx_k / 2));

            // if (sgid == 0 && sglid == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
            //     printf("expert_id=%d, k_start=%u, k_end=%u, idx_k=%u, n_start=%u, act= %f, %f, %f, %f\n", expert_id, k_start, k_end, idx_k, n_start,
            //         act4.s0, act4.s1, act4.s2, act4.s3);
            // }
	
		unroll_for (uint idx_n = n_start; idx_n < n_start + N_BLOCK; idx_n += 1) {
            uchar2 w4 = intel_sub_group_block_read_uc2((const __global uchar*)weight + (idx_n * K + idx_k) / 2);
            half2 ws2 = ((const __global half2*)scales)[(idx_n * NUM_GROUPS + idx_k / DOWN_GROUP_SIZE) / 2];
			half dot0 = act4.s0 * UNPACK_LO(w4.s0);
             dot0 = fma(act4.s1 , UNPACK_HI(w4.s0), dot0);
            half dot1 = act4.s2 * UNPACK_LO(w4.s1);
             dot1 = fma(act4.s3 , UNPACK_HI(w4.s1), dot1);
            out_private_buf[idx_n - n_start] += dot0 * ws2.s0 + dot1 * ws2.s1;
		}
    }

	for (uint idx_n = 0; idx_n < N_BLOCK; idx_n += 1) {
        out_private_buf[idx_n] = sub_group_reduce_add(out_private_buf[idx_n]);
    }

    // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
    //     printf("out_prv_buf=%f,%f,%f,%f,%f,%f,%f,%f\n", out_private_buf[0], out_private_buf[1], out_private_buf[2], out_private_buf[3],
    //         out_private_buf[4], out_private_buf[5], out_private_buf[6], out_private_buf[7]);
    // }

    if (sglid == 0) {
        for (uint idx_n = 0; idx_n < N_BLOCK; idx_n += 1) {
            out_local_buf[sgid * N_BLOCK + idx_n] = out_private_buf[idx_n];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint subgroup_global_id = sgid * SUBGROUP_SIZE + sglid;

    if (subgroup_global_id < N_BLOCK) {
        float sum = 0.0f;
        for (uint i = 0; i < num_sg; i++) {
            sum += out_local_buf[i * N_BLOCK + subgroup_global_id];
        }
        #ifdef BIAS_DT
            sum += bias[n_start + subgroup_global_id];
        #endif
        out_ptr[n_start + subgroup_global_id] = sum;
    }

//     // __local half x2[K];
//     // __local float xg_sum[K / FAKE_GROUP_SIZE];
//     // __local half scales_SLM[N_BLOCK * NUM_GROUPS * SUBGROUP_NUM];

// #    if WEIGHT_COMPRESSEION_DT == 0
//     //# interleaving x into x2
//     int id_sg = get_sub_group_id();
//     int num_sg = get_num_sub_groups();
//     int id_local = get_sub_group_local_id();
//     const __global half* px = input_ptr + id_sg * FAKE_GROUP_SIZE;
//     half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
//     for(int i = id_sg; i < K / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
//         //# quantization group
//         // float x_group_sum = 0;
//         for(int j = id_local; j < FAKE_GROUP_SIZE / 2; j += SUBGROUP_SIZE) {
//             half even = px[2 * j + 0];
//             half odd = px[2 * j + 1];
//             px2[j] = even;
//             px2[j + FAKE_GROUP_SIZE / 2] = odd;
//             // x_group_sum += even + odd;
//         }
//         // x_group_sum = sub_group_reduce_add(x_group_sum);
//         // if (id_local == 0) {
//         //     xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
//         // }
//     }
//     int n_start = get_global_id(2) * N_BLOCK;
//     for (uint i = id_local; i < (N_BLOCK * NUM_GROUPS); i += SUBGROUP_SIZE) {
//         scales_SLM[N_BLOCK * NUM_GROUPS * id_sg + i] = weight_scales[expert_id * expert_scale_size + n_start * NUM_GROUPS + i];
//     }
//     __local half* scales = scales_SLM + N_BLOCK * NUM_GROUPS * id_sg;

// #    else
//     //# load x into slm
//     int id_sg = get_sub_group_id();
//     int num_sg = get_num_sub_groups();
//     int id_local = get_sub_group_local_id();
//     half* px = input_ptr + id_sg * FAKE_GROUP_SIZE;
//     half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
//     unroll_for(int i = id_sg; i < K / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
//         //# quantization group
//         float x_group_sum = 0;
//         unroll_for(int j = id_local; j < FAKE_GROUP_SIZE; j += SUBGROUP_SIZE) {
//             half value = px[j];
//             px2[j] = value;
//             x_group_sum += value;
//         }
//         x_group_sum = sub_group_reduce_add(x_group_sum);
//         if (id_local == 0) {
//             xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
//         }
//     }
// #    endif

//     barrier(CLK_LOCAL_MEM_FENCE);

// #    if WEIGHT_COMPRESSEION_DT == 0
//     down_gemv_n2x_u4(weight, scales, zps, bias, out_ptr, N, K, x2);
// #    elif WEIGHT_COMPRESSEION_DT == 1
//     down_gemv_n2x_u8(weight, scales, zps, routing_weights, out_ptr, N, K, x2, xg_sum);
// #    elif WEIGHT_COMPRESSEION_DT == 2
//     down_gemv_n2x_f16(weight, routing_weights, out_ptr, N, K, x2);
// #    endif
}
