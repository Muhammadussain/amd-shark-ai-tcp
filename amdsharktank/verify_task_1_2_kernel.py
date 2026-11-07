"""Task 1.2 — direct kernel math verification."""

import torch
from gguf import GGMLQuantizationType as GT
from gguf.quants import quantize as gguf_quantize, dequantize as gguf_dequantize

from amdsharktank.types import PlanarQuantizedTensor
from amdsharktank.types.gguf_interop import layouts as gl
from amdsharktank import ops

torch.manual_seed(0)

B, M, N, K = 2, 4, 64, 128
w_f32 = (torch.randn(N, K, dtype=torch.float32) * 0.02).numpy()
raw_np = gguf_quantize(w_f32, GT.Q5_0)
ref_w = gguf_dequantize(raw_np, GT.Q5_0)

q5 = gl.Q5_0(raw=torch.from_numpy(raw_np.copy()), shape=[N, K], name="w")
rhs = PlanarQuantizedTensor(name="w", shape=[N, K], layout=q5.unpack())

a = torch.randn(B, M, K, dtype=torch.float16) * 0.1
got = ops.matmul(a, rhs, transpose_rhs=True)
expected = torch.matmul(a.float(), torch.from_numpy(ref_w).T.float()).to(torch.float16)

torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)
print("✓ Kernel math matches PyTorch reference")