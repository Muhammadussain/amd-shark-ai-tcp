# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import numpy as np
import torch
from parameterized import parameterized
from gguf import GGMLQuantizationType as GT
from gguf.quants import quantize as gguf_quantize

from iree.turbine import aot
from amdsharktank import kernels, ops
from amdsharktank.types.gguf_interop import layouts as gl


def _planes(n, k, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n, k)) * 0.05).astype(np.float32)
    raw = torch.from_numpy(gguf_quantize(x, GT.Q5_0).copy())
    t = gl.Q5_0(raw=raw, shape=[n, k], name="w")
    return t, t.unpack()


class mmt_block_scaled_q5_0_test(unittest.TestCase):
    @parameterized.expand(
        [
            (torch.float32, 1e-4, 1e-4),
            (torch.float16, 2e-2, 2e-2),
        ]
    )
    def testMatchesDequantMatmul(self, a_dtype, atol, rtol):
        torch.manual_seed(42)
        n, k = 96, 256
        _, layout = _planes(n, k)
        a = torch.randn([3, 5, k], dtype=a_dtype)
        result = kernels.mmt_block_scaled_q5_0(a, layout.d, layout.qh, layout.qs)
        w = layout.dequant(torch.float32)  # [n, k]
        ref = torch.matmul(a.to(torch.float32), w.T).to(a_dtype)
        self.assertEqual(result.shape, (3, 5, n))
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

    def testOpOverrideDispatchesToKernel(self):
        torch.manual_seed(1)
        n, k = 64, 128
        t, layout = _planes(n, k, seed=3)
        a = torch.randn([2, 4, k], dtype=torch.float16)
        via_op = ops.matmul(a, t, transpose_rhs=True)
        direct = kernels.mmt_block_scaled_q5_0(a, layout.d, layout.qh, layout.qs)
        torch.testing.assert_close(via_op, direct, atol=0, rtol=0)

    def testExportDynamicDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, d, qh, qs):
                return kernels.mmt_block_scaled_q5_0(a, d, qh, qs)

        n, k = 64, 128
        _, layout = _planes(n, k)
        batch = torch.export.Dim("batch")
        m = torch.export.Dim("m")
        ep = torch.export.export(
            MyModule(),
            args=(torch.rand([4, 16, k], dtype=torch.float16), layout.d, layout.qh, layout.qs),
            dynamic_shapes={"a": {0: batch, 1: m}, "d": {}, "qh": {}, "qs": {}},
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@amdsharktank_mmt_block_scaled_q5_0_3d_64_128_f16", asm)
        # Dequantization is fused into the multiply-accumulate: only one
        # linalg.generic, no [N, K] weight tensor.
        self.assertEqual(asm.count("linalg.generic"), 1)
        self.assertNotIn("tensor<64x128xf16>", asm)
        self.assertNotIn("tensor<64x128xf32>", asm)


if __name__ == "__main__":
    unittest.main()
