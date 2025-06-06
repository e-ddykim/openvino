// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/variadic_split_transformation.hpp"
#include "common_test_utils/test_constants.hpp"


using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<LayerTestsDefinitions::VariadicSplitTransformationParam> params{
    // tensor quantization, split second dimension
    {
        { 256ul, ov::Shape{ }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f / 2.f } },
        2,
        std::vector<size_t>{9, 7}
    },
    // tensor quantization, split third dimension
    {
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { 0.f }, { 25.5f } },
        -1,
        std::vector<size_t>{15, 1}
    },
    // per-channel quantization with different values, per-channel split
    {
        {
            256ul,
            ov::Shape{ 1, 3, 1, 1 },
            { -127.f, 0.f, 128.f / 2.f },
            { 128.f / 4.f, 128.f / 2.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f / 4.f, 255.f / 2.f, 255.f }
        },
        1,
        std::vector<size_t>{1, 1, 1}
    },
    // per-channel quantization with different values, split third dimension
    {
        {
            256ul,
            ov::Shape{ 1, 3, 1, 1 },
            { -127.f, 0.f, 128.f / 2.f },
            { 128.f / 4.f, 128.f / 2.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f / 4.f, 255.f / 2.f, 255.f }
        },
        -1,
        std::vector<size_t>{4, 3, 2, 7}
    },
    // per-channel quantization with the same values, per-channel split
    {
        {
            256ul, ov::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f, 255.f }
        },
        1,
        std::vector<size_t>{1, 1, 1}
    },
    // per-channel quantization with the same values, split third dimension
    {
        {
            256ul, ov::Shape{ 1, 3, 1, 1 },
            { -127.f, -127.f, -127.f },
            { 128.f, 128.f, 128.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f, 255.f }
        },
        -1,
        std::vector<size_t>{4, 3, 2, 7}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, VariadicSplitTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(params)),
    VariadicSplitTransformation::getTestCaseName);

}  // namespace
