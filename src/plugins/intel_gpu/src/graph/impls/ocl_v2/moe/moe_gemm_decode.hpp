// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "moe_gemm_gen_opt.hpp"

namespace ov::intel_gpu::ocl {

class MoEGemmDecodeGenerator : public MoEGemmOptGeneratorBase {
public:
    explicit MoEGemmDecodeGenerator() : MoEGemmOptGeneratorBase("moe_gemm_decode", "") {}

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};
}  // namespace ov::intel_gpu::ocl
