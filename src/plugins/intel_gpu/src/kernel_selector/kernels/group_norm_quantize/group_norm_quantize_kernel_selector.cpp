// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_norm_quantize_kernel_selector.h"
#include "group_norm_quantize_kernel_bfyx_opt.h"

namespace kernel_selector {

group_norm_quantize_kernel_selector::group_norm_quantize_kernel_selector() {
    Attach<GroupNormQuantizeKernelBfyx>();
}

KernelsData group_norm_quantize_kernel_selector::GetBestKernels(const Params &params) const {
    return GetNaiveBestKernel(params, KernelType::GROUP_NORM_QUANTIZE);
}

}  // namespace kernel_selector
