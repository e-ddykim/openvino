// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "quantize_inst.h"
#include "reorder_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>
#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace {

topology make_i8_dequantize_topology(engine& engine, float dequantization_scale) {
    auto make_scalar = [&](float value) {
        auto memory = engine.allocate_memory({ov::PartialShape{1}, data_types::f32, format::bfyx});
        set_values<float>(memory, {value});
        return memory;
    };

    auto input_low = make_scalar(-128.0f);
    auto input_high = make_scalar(127.0f);
    auto output_low = make_scalar(-128.0f);
    auto output_high = make_scalar(127.0f);
    auto input_scale = make_scalar(1.0f);
    auto input_shift = make_scalar(0.0f);
    auto output_scale = make_scalar(1.0f);
    auto output_shift = make_scalar(-1.0f);
    auto dequantize_scale = make_scalar(dequantization_scale);

    auto quantize_prim = quantize("quantize",
                                  {input_info("input"),
                                   input_info("input_low"),
                                   input_info("input_high"),
                                   input_info("output_low"),
                                   input_info("output_high"),
                                   input_info("input_scale"),
                                   input_info("input_shift"),
                                   input_info("output_scale"),
                                   input_info("output_shift")},
                                  256,
                                  data_types::i8);
    quantize_prim.scale_shift_opt = true;
    quantize_prim.need_post_shift = true;
    quantize_prim.per_tensor_input_range = true;
    quantize_prim.per_tensor_input_scale = true;
    quantize_prim.per_tensor_output_range = true;
    quantize_prim.per_tensor_output_scale = true;
    quantize_prim.per_tensor_output_shift = true;
    quantize_prim.in_lo = -128.0f;
    quantize_prim.in_hi = 127.0f;
    quantize_prim.in_scale = 1.0f;
    quantize_prim.out_lo = -128.0f;
    quantize_prim.out_hi = 127.0f;
    quantize_prim.out_scale = 1.0f;
    quantize_prim.out_shift = -1.0f;

    topology topology;
    topology.add(input_layout("input", layout{ov::PartialShape{1, 1, 1, 12}, data_types::f32, format::bfyx}));
    topology.add(data("input_low", input_low));
    topology.add(data("input_high", input_high));
    topology.add(data("output_low", output_low));
    topology.add(data("output_high", output_high));
    topology.add(data("input_scale", input_scale));
    topology.add(data("input_shift", input_shift));
    topology.add(data("output_scale", output_scale));
    topology.add(data("output_shift", output_shift));
    topology.add(data("dequantize_scale", dequantize_scale));
    topology.add(quantize_prim);
    topology.add(eltwise("dequantize", {input_info("quantize"), input_info("dequantize_scale")}, eltwise_mode::prod, data_types::f16));
    topology.add(reorder("output", input_info("dequantize"), format::bfyx, data_types::f32));
    return topology;
}

}  // namespace

TEST(prepare_quantization, program_replace_check_num_of_nodes) {
    auto& engine = get_test_engine();
    auto data0_layout = engine.allocate_memory({ ov::PartialShape{1}, data_types::f32, format::bfyx });
    auto data1_layout = engine.allocate_memory({ ov::PartialShape{1}, data_types::f32, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(0), data_types::f32, format::bfyx };

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("input_low", data0_layout));
    topology.add(data("input_high", data1_layout));
    topology.add(quantize("quantize", input_info("input"), input_info("input_low"), input_info("input_high"), input_info("input_low"), input_info("input_high"), 256, data_types::f32));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(prog->get_node("quantize").get_dependencies().size() == 5);

    program_wrapper::apply_opt_pass<prepare_quantization>(*prog);

    ASSERT_TRUE(prog->get_node("quantize").get_dependencies().size() == 9);
}

TEST(prepare_quantization, fuse_i8_dequantize_multiply_to_f16_quantize) {
    auto& engine = get_test_engine();
    auto topology = make_i8_dequantize_topology(engine, 0.25f);
    auto input = engine.allocate_memory({ov::PartialShape{1, 1, 1, 12}, data_types::f32, format::bfyx});
    set_values<float>(input, {-200.0f, -127.5f, -126.5f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 126.5f, 127.5f, 200.0f});

    ExecutionConfig reference_config = get_test_default_config(engine);
    reference_config.set_property(ov::intel_gpu::optimize_data(false));
    network reference_network(engine, topology, reference_config);
    reference_network.set_input_data("input", input);
    auto reference_outputs = reference_network.execute();

    ExecutionConfig optimized_config = get_test_default_config(engine);
    optimized_config.set_property(ov::intel_gpu::optimize_data(true));
    network optimized_network(engine, topology, optimized_config);
    optimized_network.set_input_data("input", input);
    auto optimized_outputs = optimized_network.execute();

    auto optimized_program = optimized_network.get_program();
    ASSERT_FALSE(has_node(*optimized_program, "dequantize"));
    ASSERT_TRUE(has_node(*optimized_program, "quantize"));
    const auto& quantize_node = optimized_program->get_node("quantize").as<cldnn::quantize>();
    EXPECT_EQ(quantize_node.get_output_layout().data_type, data_types::f16);
    EXPECT_TRUE(quantize_node.get_output_round_to_even());
    EXPECT_FLOAT_EQ(quantize_node.get_output_scale_val(), 0.25f);
    EXPECT_FLOAT_EQ(quantize_node.get_output_shift_val(), -1.0f);
    EXPECT_FLOAT_EQ(quantize_node.get_output_lo_val(), -32.0f);
    EXPECT_FLOAT_EQ(quantize_node.get_output_hi_val(), 31.75f);

    mem_lock<float> reference_output(reference_outputs.at("output").get_memory(), get_test_stream());
    mem_lock<float> optimized_output(optimized_outputs.at("output").get_memory(), get_test_stream());
    const std::vector<float> expected = {-32.0f, -32.0f, -32.0f, -1.0f, -0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 31.5f, 31.5f, 31.75f};
    ASSERT_EQ(reference_output.size(), expected.size());
    ASSERT_EQ(optimized_output.size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_FLOAT_EQ(reference_output[index], expected[index]);
        EXPECT_FLOAT_EQ(optimized_output[index], reference_output[index]);
    }
}

TEST(prepare_quantization, negative_dequantize_scale_no_fusion) {
    auto& engine = get_test_engine();
    auto topology = make_i8_dequantize_topology(engine, -0.25f);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_quantization>(*prog);

    ASSERT_TRUE(has_node(*prog, "dequantize"));
    const auto& quantize_node = prog->get_node("quantize").as<cldnn::quantize>();
    EXPECT_EQ(quantize_node.get_output_layout().data_type, data_types::i8);
    EXPECT_FALSE(quantize_node.get_output_round_to_even());
}
