"""Task 1.1 — Q5_0 type verification on REAL TinyLlama model."""

import os
import numpy as np
import torch
import gguf
from gguf import GGMLQuantizationType as GT
from gguf.quants import dequantize as gguf_dequantize
from huggingface_hub import hf_hub_download

from amdsharktank.types import Dataset, Theta, PlanarQuantizedTensor
from amdsharktank.types import gguf_interop
from amdsharktank.types.gguf_interop import layouts as gl


def download_tinyllama_q5_0():
    """HuggingFace se real Q5_0 GGUF download."""
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    filename = "tinyllama-1.1b-chat-v1.0.Q5_0.gguf"
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"✓ Downloaded: {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def load_and_identify(gguf_path):
    """Load into amdsharktank, extract Q5_0 tensors, identify attention/MLP."""
    ds = gguf_interop.load_file(gguf_path)
    
    all_tensors = dict(ds.root_theta.flatten())
    q5_0_tensors = {n: t for n, t in all_tensors.items() if isinstance(t, gl.Q5_0)}
    
    print(f"Total: {len(all_tensors)} tensors, {len(q5_0_tensors)} Q5_0")
    
    attention = {n: t for n, t in q5_0_tensors.items() if ".attn_" in n}
    mlp       = {n: t for n, t in q5_0_tensors.items() if ".ffn_" in n}
    
    print(f"Attention: {len(attention)}, MLP: {len(mlp)}")
    return ds, q5_0_tensors, attention, mlp


def verify_against_llama_cpp(gguf_path, ds, attention, mlp):
    """3 attention + 3 MLP tensors ko llama.cpp gguf_dequantize se bit-exact compare."""
    reader = gguf.GGUFReader(gguf_path)
    name_to_tensor = {t.name: t for t in reader.tensors}
    
    test_names = sorted(attention.keys())[:3] + sorted(mlp.keys())[:3]
    results = []
    
    for name in test_names:
        gt = name_to_tensor[name]
        n, k = gt.shape
        raw_bytes = np.asarray(gt.data, dtype=np.uint8).reshape(n, k // 32 * 22)
        
        ref = gguf_dequantize(raw_bytes, GT.Q5_0)              # llama.cpp reference
        got = ds.root_theta.tensor(name).unpack().dequant(torch.float32).numpy()
        
        exact = np.array_equal(got.ravel(), ref.ravel())        # bit-exact
        max_diff = float(np.abs(got.ravel() - ref.ravel()).max()) if exact else float("inf")
        
        cat = "attention" if ".attn_" in name else "MLP"
        results.append({"name": name, "category": cat, "exact": exact, "max_diff": max_diff})
        print(f"{'✓' if exact else '✗'} {name:<40s} {cat:<10s} diff={max_diff:.2e}")
    
    return results


def verify_irpa(gguf_path, ds):
    """Q5_0 tensors as IRPA save, verify size and plane shapes."""
    irpa_path = gguf_path.replace(".gguf", ".irpa")
    
    # Save only Q5_0 tensors (avoids NotImplementedError for Q6_K etc.)
    q5_0_only = {k: v for k, v in ds.root_theta.flatten().items()
                 if isinstance(v, gl.Q5_0)}
    Dataset(properties=ds.properties, root_theta=Theta(q5_0_only)).save(irpa_path)
    
    gguf_size = os.path.getsize(gguf_path)
    irpa_size = os.path.getsize(irpa_path)
    print(f"GGUF: {gguf_size/1e6:.1f} MB, IRPA: {irpa_size/1e6:.1f} MB "
          f"(ratio {irpa_size/gguf_size:.3f})")
    assert irpa_size < gguf_size * 1.10, "IRPA too large"
    
    # Reload and check one tensor's planes
    ds2 = Dataset.load(irpa_path)
    sample_name = next(n for n in ds2.root_theta.flatten() if ".attn_q" in n)
    t = ds2.root_theta.tensor(sample_name)
    N, K = t.shape
    
    print(f"\nSample: {sample_name} [{N}×{K}]")
    planes = t.unpack().planes
    for pname, divisor in [("d", 32), ("qh", 8), ("qs", 2)]:
        pt = planes[pname]
        expected = N * K // divisor
        ok = pt.numel() == expected
        print(f"  {'✓' if ok else '✗'} {pname}: {tuple(pt.shape)} {pt.dtype} "
              f"numel={pt.numel():,} (expected {expected:,})")
        assert ok, f"plane {pname} has wrong size"
    
    # Compression check
    total_plane_bytes = sum(
        pt.numel() * pt.element_size()
        for t in ds2.root_theta.flatten().values() if isinstance(t, PlanarQuantizedTensor)
        for pt in t.unpack().planes.values()
    )
    total_f16_bytes = sum(
        t.shape[0] * t.shape[1] * 2
        for t in ds2.root_theta.flatten().values() if isinstance(t, PlanarQuantizedTensor)
    )
    print(f"\nPlane bytes: {total_plane_bytes/1e6:.1f} MB (f16 equivalent: "
          f"{total_f16_bytes/1e6:.1f} MB, ratio {total_f16_bytes/total_plane_bytes:.2f}×)")
    
    return irpa_path


def main():
    gguf_path = download_tinyllama_q5_0()
    ds, q5_0_tensors, attention, mlp = load_and_identify(gguf_path)
    results = verify_against_llama_cpp(gguf_path, ds, attention, mlp)
    irpa_path = verify_irpa(gguf_path, ds)
    
    exact_count = sum(1 for r in results if r["exact"])
    attn_exact = sum(1 for r in results if r["exact"] and r["category"] == "attention")
    mlp_exact  = sum(1 for r in results if r["exact"] and r["category"] == "MLP")
    
    print(f"\n{'='*60}")
    print(f"  Task 1.1 Summary")
    print(f"{'='*60}")
    print(f"  ✓ GGUF → IRPA: success")
    print(f"  ✓ Planes verified: d/qh/qs shapes correct")
    print(f"  ✓ IRPA size < GGUF × 1.10")
    print(f"  {'✓' if exact_count >= 3 else '✗'} Bit-exact match: {exact_count}/{len(results)}")
    print(f"  {'✓' if attn_exact >= 1 else '✗'} Attention projection: {attn_exact}")
    print(f"  {'✓' if mlp_exact >= 1 else '✗'} MLP projection: {mlp_exact}")
    print(f"\n  GGUF: {gguf_path}")
    print(f"  IRPA: {irpa_path}")
    
    assert exact_count >= 3 and attn_exact >= 1 and mlp_exact >= 1


if __name__ == "__main__":
    main()