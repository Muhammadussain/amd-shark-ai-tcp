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

from amdsharktank import kernels
from amdsharktank.types.gguf_interop import layouts as gl


def _planes(n, k, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n, k)) * 0.05).astype(np.float32)
    raw = torch.from_numpy(gguf_quantize(x, GT.Q5_0).copy())
    t = gl.Q5_0(raw=raw, shape=[n, k], name="w")
    return t, t.unpack()


class mmt_block_scaled_q5_0_tiled_test(unittest.TestCase):
    @parameterized.expand(
        [
            # (m0, n0, k0): includes tiles that do not divide M, N or K.
            (1, 1, 32),
            (4, 8, 32),
        ]
    )
    def testMatchesUntiledAndReference(self, m0, n0, k0):
        torch.manual_seed(42)
        n, k = 96, 256
        _, layout = _planes(n, k)
        a = torch.randn([3, 5, k], dtype=torch.float16)
        tiled = kernels.mmt_block_scaled_q5_0_tiled(
            a, layout.d, layout.qh, layout.qs, m0, n0, k0
        )
        untiled = kernels.mmt_block_scaled_q5_0(a, layout.d, layout.qh, layout.qs)
        ref = torch.matmul(a.float(), layout.dequant(torch.float32).T).to(torch.float16)
        self.assertEqual(tiled.shape, (3, 5, n))
        torch.testing.assert_close(tiled, ref, atol=2e-2, rtol=2e-2)
        # Same math as the untiled kernel; only summation order may differ.
        torch.testing.assert_close(tiled, untiled, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()