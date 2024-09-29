// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "intel_gpu/primitives/group_norm_quantize.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "openvino/reference/group_normalization.hpp"
#include "openvino/reference/swish.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"


using namespace cldnn;
using namespace ::tests;

namespace {

typedef std::tuple<
    std::vector<std::int32_t>,  // Input shape
    std::size_t,                // Number of groups
    double,                     // Epsilon
    format,                     // First input layout
    padding                     // Output padding
>
GroupNormalizationParams;

class GroupNormalizationGPUTest : public ::testing::TestWithParam<GroupNormalizationParams> {
public:
    GroupNormalizationGPUTest() = default;

    void SetUp() override {
        std::vector<std::int32_t> input_shape;
        const auto& params = GetParam();
        std::tie(input_shape, num_groups_, epsilon_, format_, output_pad_) = params;
        std::copy(std::begin(input_shape), std::end(input_shape), std::back_inserter(data_shape_));
        tests::random_generator rg{"GroupNormalizationGPUTest"};
        data_ = rg.generate_random_1d<float>(ov::shape_size(input_shape), -1, 1);
        scale_ = rg.generate_random_1d<float>(input_shape[1], -1, 1);
        bias_ = rg.generate_random_1d<float>(input_shape[1], -1, 1);
        const auto planar_format = format::dimension(format_) == 4 ? format::bfyx : format::bfzyx;

        topology tp;
        auto &engine = get_test_engine();
        data_layout_ = layout{data_types::f32, planar_format, tensor{input_shape}};
        scale_bias_layout_ = layout{data_types::f32, planar_format, tensor{1,
            static_cast<std::int32_t>(scale_.size()), 1, 1}};

        primitive_id reordered_data_primitive = data_primitive_ + "_reordered";
        tp.add(input_layout{data_primitive_, data_layout_});
        tp.add(input_layout{scale_primitive_, scale_bias_layout_});
        tp.add(input_layout{bias_primitive_, scale_bias_layout_});
        tp.add(reorder{reordered_data_primitive, data_primitive_, format_, data_types::f32});

        auto g = group_normalization{
            "group_normalization_output",
            input_info{reordered_data_primitive},
            input_info{scale_primitive_},
            input_info{bias_primitive_},
            static_cast<std::int64_t>(num_groups_),
            epsilon_
        };
        g.output_paddings = {output_pad_};
        tp.add(g);
        tp.add(reorder{"output", input_info("group_normalization_output"), planar_format, data_types::f32});

        network_ = std::make_shared<cldnn::network>(engine, tp, get_test_default_config(engine));
    }

    void Test() {
        auto &engine = get_test_engine();
        auto data_gpu_mem = engine.allocate_memory(data_layout_);
        auto scale_gpu_mem = engine.allocate_memory(scale_bias_layout_);
        auto bias_gpu_mem = engine.allocate_memory(scale_bias_layout_);
        set_values(data_gpu_mem, data_);
        set_values(scale_gpu_mem, scale_);
        set_values(bias_gpu_mem, bias_);
        network_->set_input_data(data_primitive_, data_gpu_mem);
        network_->set_input_data(scale_primitive_, scale_gpu_mem);
        network_->set_input_data(bias_primitive_, bias_gpu_mem);
        auto outputs = network_->execute();
        auto output = outputs.at("output").get_memory();
        cldnn::mem_lock<float> output_gpu_mem(output, get_test_stream());

        std::vector<float> reference_output(data_.size());
        ov::reference::group_normalization(data_.data(), scale_.data(), bias_.data(), reference_output.data(),
                                           ov::Shape{data_shape_}, num_groups_, epsilon_);

        ASSERT_EQ(output_gpu_mem.size(), reference_output.size());
        for (std::size_t i = 0; i < reference_output.size(); i++) {
            ASSERT_NEAR(output_gpu_mem[i], reference_output[i], 0.0001);
        }
    }

private:
    std::vector<float> data_{};
    std::vector<float> scale_{};
    std::vector<float> bias_{};
    std::size_t num_groups_{};
    double epsilon_{};
    format format_{format::any};
    padding output_pad_{padding()};
    network::ptr network_{};
    layout data_layout_{};
    layout scale_bias_layout_{};
    std::vector<std::size_t> data_shape_;
    static const primitive_id data_primitive_;
    static const primitive_id scale_primitive_;
    static const primitive_id bias_primitive_;
};

const primitive_id GroupNormalizationGPUTest::data_primitive_{"data"};
const primitive_id GroupNormalizationGPUTest::scale_primitive_{"scale"};
const primitive_id GroupNormalizationGPUTest::bias_primitive_{"bias"};

TEST_P(GroupNormalizationGPUTest, random) {
    Test();
}

const std::vector<cldnn::format> f_planar_4d_formats {
    format::bfyx,
};

const std::vector<cldnn::format> f_blocked_4d_formats {
    format::b_fs_yx_fsv16,
};

const std::vector<cldnn::format> f_planar_5d_formats {
    format::bfzyx,
};

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_planar_layouts_support_4d, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn({std::vector<int32_t>{3, 64, 32, 64}, std::vector<int32_t>{3, 124, 97, 61}, std::vector<int32_t>{1, 1536, 151, 1}, std::vector<int32_t>{1, 12, 2175, 1}}),
        ::testing::ValuesIn(std::vector<size_t>{1, 4}),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_planar_4d_formats),
        ::testing::ValuesIn({padding(), padding({0, 0, 1, 1})})));

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_blocked_layouts_support_4d, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn({std::vector<int32_t>{3, 64, 32, 64}, std::vector<int32_t>{3, 124, 97, 61}, std::vector<int32_t>{1, 1536, 151, 1}, std::vector<int32_t>{1, 12, 2175, 1}}),
        ::testing::ValuesIn(std::vector<size_t>{1, 2, 4}),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_blocked_4d_formats),
        ::testing::ValuesIn({padding(), padding({0, 16, 0, 0})})));

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationGPUTest_planar_layouts_support_5d, GroupNormalizationGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn({std::vector<int32_t>{3, 64, 28, 32, 12}, std::vector<int32_t>{3, 124, 10, 97, 61}, std::vector<int32_t>{1, 1536, 9, 151, 1}, std::vector<int32_t>{1, 12, 8, 2175, 1}}),
        ::testing::ValuesIn(std::vector<size_t>{1, 4}),
        ::testing::Values(0.0025),
        ::testing::ValuesIn(f_planar_5d_formats),
        ::testing::ValuesIn({padding(), padding({0, 0, 1, 1})})));

class group_norm_quantize_gpu_tests: public ::testing::Test {
public:
    void test_basic(bool is_caching_test) {
        std::vector<std::int32_t> input_shape = {3, 64, 32, 64};
        std::vector<std::size_t> data_shape_;
        std::copy(std::begin(input_shape), std::end(input_shape), std::back_inserter(data_shape_));
        size_t num_groups_ = 4;
        double epsilon_ = 0.0025;
        format format_ = format::bfyx;
        const primitive_id data_primitive_ = "data";
        const primitive_id scale_primitive_ = "scale";
        const primitive_id bias_primitive_ = "bias";

        tests::random_generator rg{"GroupNormalizationGPUTest"};
        std::vector<float> data_ = rg.generate_random_1d<float>(ov::shape_size(input_shape), -1, 1);
        std::vector<float> scale_ = rg.generate_random_1d<float>(input_shape[1], -1, 1);
        std::vector<float> bias_ = rg.generate_random_1d<float>(input_shape[1], -1, 1);
        const auto planar_format = format::dimension(format_) == 4 ? format::bfyx : format::bfzyx;

        topology tp;
        auto &engine = get_test_engine();
        layout data_layout_ = layout{data_types::f32, planar_format, tensor{input_shape}};
        layout scale_bias_layout_ = layout{data_types::f32, planar_format, tensor{1,
            static_cast<std::int32_t>(scale_.size()), 1, 1}};

        primitive_id reordered_data_primitive = data_primitive_ + "_reordered";
        tp.add(input_layout{data_primitive_, data_layout_});
        tp.add(input_layout{scale_primitive_, scale_bias_layout_});
        tp.add(input_layout{bias_primitive_, scale_bias_layout_});
        tp.add(reorder{reordered_data_primitive, data_primitive_, format_, data_types::f32});

        std::vector<optional_data_type> hoho = {optional_data_type(data_types::f16), optional_data_type(data_types::f32)};

        auto g = group_norm_quantize{
            "group_norm_quantize_output",
            input_info{reordered_data_primitive},
            input_info{scale_primitive_},
            input_info{bias_primitive_},
            static_cast<std::int64_t>(num_groups_),
            epsilon_,
            hoho
        };
        // g.output_paddings = {output_pad_};
        tp.add(g);
        tp.add(reorder{"output_quantized", input_info("group_norm_quantize_output", 0), planar_format, data_types::f32});
        tp.add(reorder{"output_scale", input_info("group_norm_quantize_output", 1), planar_format, data_types::f32});
        tp.add(eltwise{"output", input_info("output_quantized"), input_info("output_scale"), eltwise_mode::prod});

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network_ = std::make_shared<cldnn::network>(engine, tp, cfg);

        auto data_gpu_mem = engine.allocate_memory(data_layout_);
        auto scale_gpu_mem = engine.allocate_memory(scale_bias_layout_);
        auto bias_gpu_mem = engine.allocate_memory(scale_bias_layout_);
        set_values(data_gpu_mem, data_);
        set_values(scale_gpu_mem, scale_);
        set_values(bias_gpu_mem, bias_);
        network_->set_input_data(data_primitive_, data_gpu_mem);
        network_->set_input_data(scale_primitive_, scale_gpu_mem);
        network_->set_input_data(bias_primitive_, bias_gpu_mem);
        auto outputs = network_->execute();
        auto output = outputs.at("output").get_memory();
        cldnn::mem_lock<float> output_gpu_mem(output, get_test_stream());
        // auto output_scale = outputs.at("output_scale").get_memory();
        // cldnn::mem_lock<float> output_scale_gpu_mem(output_scale, get_test_stream());

        std::vector<float> temp_output(data_.size());
        std::vector<float> reference_output(data_.size());
        ov::reference::group_normalization(data_.data(), scale_.data(), bias_.data(), temp_output.data(),
                                           ov::Shape{data_shape_}, num_groups_, epsilon_);
        ov::reference::swish(temp_output.data(), 1.f, reference_output.data(), ov::shape_size(data_shape_));
        float max_ref = reference_output[0];
        float min_ref = max_ref;
        for (std::size_t i = 1; i < reference_output.size(); i++) {
            if (max_ref < reference_output[i])
                max_ref = reference_output[i];
            if (min_ref > reference_output[i])
                min_ref = reference_output[i];
        }
        std::cout << "min_ref: " << min_ref << std::endl;
        std::cout << "max_ref: " << max_ref << std::endl;

        // std::cout << "output_scale_gpu_mem: " << output_scale_gpu_mem.size() << std::endl;

        ASSERT_EQ(output_gpu_mem.size(), reference_output.size());
        for (std::size_t i = 0; i < reference_output.size(); i++) {
            ASSERT_NEAR(output_gpu_mem[i], reference_output[i], 0.001);
        }
    }
};

TEST_F(group_norm_quantize_gpu_tests, basic) {
    this->test_basic(false);
}

} // anonymous namespace
