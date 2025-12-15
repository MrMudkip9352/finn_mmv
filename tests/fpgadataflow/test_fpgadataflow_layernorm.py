###################################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
###################################################################################

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe

test_fpga_part = "xcv80-lsva4737-2MHP-e-s"
target_clk_ns = 5


def create_layernorm_model(idt, ishape, epsilon):
    scale_bias_shape = [ishape[-1]]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, scale_bias_shape)
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, scale_bias_shape)

    ln_node = helper.make_node(
        "LayerNormalization",
        inputs=["inp", "scale", "bias"],
        outputs=["outp"],
        name="Layernorm_0",
        epsilon=epsilon,
        axis=-1,
        stash_type=1,
    )

    # Create model
    graph = helper.make_graph(
        nodes=[ln_node], name="LayerNorm_graph", inputs=[inp, scale, bias], outputs=[outp]
    )
    model = qonnx_make_model(graph, producer_name="LayerNorm_graph")
    model = ModelWrapper(model)

    # Tensor initializers
    # set scale and bias to 1 or zero for now
    model.set_initializer("scale", np.ones(scale_bias_shape, dtype=np.float32))
    model.set_initializer("bias", np.zeros(scale_bias_shape, dtype=np.float32))

    # Tensor data types
    model.set_tensor_datatype("inp", idt)

    return model


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.parametrize("idt", [DataType["FLOAT32"]])
@pytest.mark.parametrize("ishape", [[1, 16, 48], [1, 32]])
def test_fpgadataflow_layernorm(idt, ishape):
    model = create_layernorm_model(idt, ishape, epsilon=9.999999960041972e-13)

    # reference calculation
    input = gen_finn_dt_tensor(DataType["FLOAT32"], ishape)
    input_t = {model.graph.input[0].name: input}

    y_ref = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]
    print(y_ref)

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
