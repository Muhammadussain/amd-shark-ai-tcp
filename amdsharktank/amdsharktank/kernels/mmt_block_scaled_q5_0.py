# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import *

import torch

__all__ = [
    "mmt_block_scaled_q5_0",
]


@CustomOp.register(library=LIBRARY)
class mmt_block_scaled_q5_0(CustomOp):
    """Block scaled Q5_0 matmul with transposed RHS (untiled).

    This corresponds to `BlockScaledQ5_0Layout` and operates on its planar
    `d`, `qh` and `qs` tensors exactly as stored in an IRPA file:

    * `d`:  `[N, K // 32, 1]`  (f16/f32)
    * `qh`: `[N, K // 32, 4]`  (uint8, 32 high bits, bit j = element j)
    * `qs`: `[N, K // 32, 16]` (uint8, byte j = element j | element j + 16 << 4)

    The LHS is expected to be a 3d tensor of shape [B, M, K]. The kernel
    dequantizes inline (`w = d * (q - 16)`) and accumulates in f32. It will be
    specialized for all values of N, K and LHS dtype.
    """

    signature = (
        "mmt_block_scaled_q5_0(Tensor a, Tensor d, Tensor qh, Tensor qs) -> (Tensor)"
    )

    def select(self, ksel: KernelSelection):
        a_desc = ksel.arg_tensor(0)  # Shape [b, m, k]
        d_desc = ksel.arg_tensor(1)  # Shape [N, K // 32, 1]
        qh_desc = ksel.arg_tensor(2)  # Shape [N, K // 32, 4]
        qs_desc = ksel.arg_tensor(3)  # Shape [N, K // 32, 16]

        # a arg
        *batch_dims, a_m, a_k = a_desc.t.shape
        torch._check(
            a_desc.t.dtype.is_floating_point,
            lambda: f"mmt_block_scaled_q5_0 arg 'a': Expected floating point (got {a_desc.t.dtype})",
        )
        torch._check(
            len(batch_dims) == 1,
            lambda: f"mmt_block_scaled_q5_0 arg 'a': Expected 3d tensor (got {a_desc.t.shape})",
        )

        # qs arg
        qs_n, qs_group0, qs_bs, *rest = qs_desc.t.shape
        torch._check(
            len(rest) == 0 and qs_bs == 16 and (qs_group0 * 32) == a_k,
            lambda: f"mmt_block_scaled_q5_0 arg 'qs': Incorrect shape (got {qs_desc.t.shape})",
        )

        # qh arg
        qh_n, qh_group0, qh_bs, *rest = qh_desc.t.shape
        torch._check(
            len(rest) == 0 and qh_bs == 4 and qh_group0 == qs_group0 and qh_n == qs_n,
            lambda: f"mmt_block_scaled_q5_0 arg 'qh': Incorrect shape (got {qh_desc.t.shape})",
        )

        # d arg
        d_n, d_group0, d_one, *rest = d_desc.t.shape
        torch._check(
            len(rest) == 0 and d_group0 == qs_group0 and d_one == 1 and d_n == qs_n,
            lambda: f"mmt_block_scaled_q5_0 arg 'd': Incorrect shape (got {d_desc.t.shape})",
        )

        # Specialize on K, N, block count.
        a_desc.specialize_dims(-1)
        d_desc.specialize_all_dims()
        qh_desc.specialize_all_dims()
        qs_desc.specialize_all_dims()

        # Shape batch..., m, n
        c_desc = ksel.return_new_tensor(batch_dims + [a_m, d_n], dtype=a_desc.t.dtype)
        c_desc.specialize_dims(-1)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        a = kb.arg_value(0)
        a_tensor_type = RankedTensorType(a.type)
        d = kb.arg_value(1)
        d_tensor_type = RankedTensorType(d.type)
        qs = kb.arg_value(3)
        qs_tensor_type = RankedTensorType(qs.type)

        rank = a_tensor_type.rank
        k = a_tensor_type.get_dim_size(rank - 1)
        n, group0, _ = qs_tensor_type.shape
        a_type_str = str(a_tensor_type.element_type)
        scale_type_str = str(d_tensor_type.element_type)

        template_file = "mmt_block_scaled_q5_0_3d.mlir"
        target_function_name = f"amdsharktank_mmt_block_scaled_q5_0_3d_{n}_{k}_{a_type_str}"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            n=n,
            k=k,
            group0=group0,
            a_type=a_type_str,
            scale_type=scale_type_str,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
