# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0

"""Task 1.1: Q5_0 type, GGUF loading, planar IRPA storage and dequantization."""

import numpy as np
import pytest
import torch

import gguf
from gguf import GGMLQuantizationType as GT
from gguf.quants import dequantize as gguf_dequantize, quantize as gguf_quantize

from amdsharktank.types import *
from amdsharktank.types import gguf_interop
from amdsharktank.types.gguf_interop import layouts as gl
from amdsharktank.types.tensors import REGISTERED_LAYOUT_CLASSES


def _raw_block(d: float, qh_bits: list[int], qs_bytes: list[int]) -> torch.Tensor:
    """Builds one 22 byte block_q5_0 by hand."""
    d_bytes = list(np.array([d], dtype=np.float16).view(np.uint8))
    qh = 0
    for j in qh_bits:
        qh |= 1 << j
    qh_bytes = list(np.array([qh], dtype=np.uint32).view(np.uint8))
    assert len(qs_bytes) == 16
    return torch.tensor(d_bytes + qh_bytes + qs_bytes, dtype=torch.uint8)


def test_layout_registered():
    assert (
        REGISTERED_LAYOUT_CLASSES[BlockScaledQ5_0Layout.serialized_name()]
        is BlockScaledQ5_0Layout
    )


def test_bit_mapping_is_the_ggml_one():
    # Element j: low nibble of qs[j] (j < 16) or high nibble of qs[j - 16],
    # 5th bit = bit j of the little endian uint32 `qh`.
    qs = [0] * 16
    qs[3] = 0xA5  # element 3 -> 5, element 19 -> 10
    raw = _raw_block(d=1.0, qh_bits=[5, 19, 20], qs_bytes=qs)
    layout = gl.Q5_0(raw=raw, shape=[1, 32]).unpack()
    q = layout.unpack_q()[0, 0].tolist()
    assert q[3] == 5
    assert q[19] == 10 + 16  # qh bit 19 is set
    assert q[5] == 16  # qh bit 5 (byte 0, bit 5) is W5's high bit
    assert q[20] == 16  # qh bit 20 (byte 2, bit 4) is W20's high bit
    others = [i for i in range(32) if i not in (3, 19, 5, 20)]
    assert all(q[i] == 0 for i in others)
    # w = d * (q - 16)
    w = layout.dequant(torch.float32)[0].tolist()
    assert w[5] == 0.0 and w[0] == -16.0 and w[19] == 10.0

@pytest.mark.parametrize("n,k", [(1, 32), (8, 64), (16, 256)])
def test_matches_gguf_reference_exactly(n, k):
    rng = np.random.default_rng(1234)
    x = (rng.standard_normal((n, k)) * 0.05).astype(np.float32)
    raw_np = gguf_quantize(x, GT.Q5_0)  # [n, k // 32 * 22] uint8
    ref = gguf_dequantize(raw_np, GT.Q5_0)  # f32 [n, k]
    t = gl.Q5_0(raw=torch.from_numpy(raw_np.copy()), shape=[n, k], name="w")
    layout = t.unpack()
    assert layout.d.shape == (n, k // 32, 1) and layout.d.dtype == torch.float16
    assert layout.qh.shape == (n, k // 32, 4) and layout.qh.dtype == torch.uint8
    assert layout.qs.shape == (n, k // 32, 16) and layout.qs.dtype == torch.uint8
    got = layout.dequant(torch.float32).numpy()
    assert got.shape == (n, k)
    np.testing.assert_array_equal(got, ref)  # bit-exact, same element order
    # The generic helper used by ops must agree too.
    np.testing.assert_array_equal(t.unpack().dequant(torch.float32).numpy(), ref)


def _write_q5_0_gguf(path, tensors: dict[str, np.ndarray]):
    w = gguf.GGUFWriter(str(path), "llama")
    for name, arr in tensors.items():
        w.add_tensor(name, gguf_quantize(arr, GT.Q5_0), raw_dtype=GT.Q5_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_gguf_to_irpa_keeps_planes_and_roundtrips(tmp_path):
    rng = np.random.default_rng(7)
    arrays = {
        f"blk.0.{n}.weight": (rng.standard_normal((256, 1024)) * 0.05).astype(
            np.float32
        )
        for n in ["attn_q", "attn_k", "ffn_up", "ffn_down"]
    }
    gguf_path = tmp_path / "m.gguf"
    irpa_path = tmp_path / "m.irpa"
    _write_q5_0_gguf(gguf_path, arrays)

    # GGUF -> Dataset: every tensor is recognised as Q5_0 (no "unsupported" error).
    ds = gguf_interop.load_file(gguf_path)
    for name, t in ds.root_theta.flatten().items():
        assert isinstance(t, gl.Q5_0), (name, type(t))

    ds.save(irpa_path)
    ds2 = Dataset.load(irpa_path)

    # IRPA holds d / qh / qs planes, never an f16/f32 weight.
    n_elems = 256 * 1024
    for name, t in ds2.root_theta.flatten().items():
        assert isinstance(t, PlanarQuantizedTensor)
        assert t.layout_type is BlockScaledQ5_0Layout
        planes = t.unpack().planes
        assert planes["d"].numel() == n_elems // 32 and planes["d"].dtype == torch.float16
        assert planes["qh"].numel() == n_elems // 8 and planes["qh"].dtype == torch.uint8
        assert planes["qs"].numel() == n_elems // 2 and planes["qs"].dtype == torch.uint8
        # Values survive the save / load roundtrip.
        orig = ds.root_theta.tensor(name).unpack().dequant(torch.float32)
        torch.testing.assert_close(t.unpack().dequant(torch.float32), orig, rtol=0, atol=0)

    # Size: about the same as the GGUF, and ~3x smaller than a f16 copy.
    irpa_size = irpa_path.stat().st_size
    gguf_size = gguf_path.stat().st_size
    f16_size = len(arrays) * n_elems * 2
    assert irpa_size < gguf_size * 1.10
    assert f16_size / irpa_size > 2.7
