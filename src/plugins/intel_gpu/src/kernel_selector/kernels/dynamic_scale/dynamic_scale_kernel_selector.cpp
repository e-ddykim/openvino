// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "dynamic_scale_kernel_selector.h"
#include "dynamic_scale_kernel_bfyx.h"
#include "dynamic_scale_kernel_b_fs_yx_fsv16.h"

namespace kernel_selector {

dynamic_scale_kernel_selector::dynamic_scale_kernel_selector() {
    Attach<DynamicScaleKernel_bfyx>();
    Attach<DynamicScaleKernel_b_fs_yx_fsv16>();
}

KernelsData dynamic_scale_kernel_selector::GetBestKernels(const Params &params) const {
    return GetNaiveBestKernel(params, KernelType::DYNAMIC_SCALE);
}

}  // namespace kernel_selector
