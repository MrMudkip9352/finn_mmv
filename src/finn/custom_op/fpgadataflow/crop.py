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

import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Crop(HWCustomOp):
    """Abstraction layer for Crop layers."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "data_type": ("s", True, ""),
            "height": ("i", True, []),
            "width": ("i", True, []),
            "crop_north": ("i", True, []),
            "crop_east": ("i", True, []),
            "crop_west": ("i", True, []),
            "crop_south": ("i", True, []),
            "simd": ("i", False, 1),
            "numInputVectors": ("ints", True, []),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        num_vec = self.get_nodeattr("numInputVectors")
        width = self.get_nodeattr("width")
        height = self.get_nodeattr("height")
        return num_vec + [height, width]

    def get_normal_output_shape(self, ind=0):
        num_vec = self.get_nodeattr("numInputVectors")
        width = self.get_nodeattr("width")
        height = self.get_nodeattr("height")
        crop_north = self.get_nodeattr("crop_north")
        crop_east = self.get_nodeattr("crop_east")
        crop_west = self.get_nodeattr("crop_west")
        crop_south = self.get_nodeattr("crop_south")
        owidth = width - (crop_west + crop_east)
        oheight = height - (crop_north + crop_south)
        return num_vec + [oheight, owidth]

    def execute_node(self, context, graph):
        node = self.onnx_node
        crop_north = self.get_nodeattr("crop_north")
        crop_east = self.get_nodeattr("crop_east")
        crop_west = self.get_nodeattr("crop_west")
        crop_south = self.get_nodeattr("crop_south")
        h = self.get_nodeattr("height")
        w = self.get_nodeattr("width")
        inp = context[node.input[0]]
        cropped_slice = inp[:, crop_north : h - crop_south, crop_west : w - crop_east]
        assert cropped_slice.shape == tuple(self.get_normal_output_shape())
        context[node.output[0]] = cropped_slice

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("data_type")]

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dt = model.get_tensor_datatype(node.input[0])
        if dt != self.get_input_datatype():
            warn_str = (
                f"data_type changing for {node.name}: {str(self.get_input_datatype())} -> {str(dt)}"
            )
            warnings.warn(warn_str)
        self.set_nodeattr("data_type", dt.name)

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("simd")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("simd")
        return obits * simd

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("data_type")]

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        simd = self.get_nodeattr("simd")
        assert normal_oshape[-1] % simd == 0, "SIMD must divid into output dimension"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("simd")
        assert normal_ishape[-1] % simd == 0, "SIMD must divid into input dimension"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_exp_cycles(self):
        simd = self.get_nodeattr("simd")
        num_vec = self.get_nodeattr("numInputVectors")
        width = self.get_nodeattr("width")
        height = self.get_nodeattr("height")

        return np.prod(num_vec) * height * (width // simd)
