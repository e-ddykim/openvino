// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "primitive.hpp"
#include "intel_gpu/op/dynamic_scale.hpp"

namespace cldnn {

/// @brief Performs the following transformation of the input tensor:
struct dynamic_scale : public primitive_base<dynamic_scale> {
    CLDNN_DECLARE_PRIMITIVE(dynamic_scale)

    dynamic_scale() : primitive_base("", {}) {}

    /// @brief Constructs dynamic_scale primitive.
    /// @param id This primitive id.
    /// @param data The input tensor to be normalized.
    dynamic_scale(const primitive_id& id,
                  const input_info& data,
                  const std::vector<optional_data_type> data_types)
    : primitive_base(id, {data}, 2, data_types) {}

    std::size_t hash() const override {
        return primitive::hash();
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<dynamic_scale>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<dynamic_scale>::load(ib);
    }
};

} // namespace cldnn
