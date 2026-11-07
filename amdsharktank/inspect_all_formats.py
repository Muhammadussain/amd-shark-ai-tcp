"""
Inspect GGUF, Dataset, and IRPA in human-readable format.

Chalayein: python inspect_all_formats.py
"""

import os
import warnings
import numpy as np
import torch
import gguf
from gguf import GGMLQuantizationType as GT

from amdsharktank.types import *
from amdsharktank.types import gguf_interop
from amdsharktank.types.gguf_interop import layouts as gl

# Suppress the harmless "not writable" warning from memmap
warnings.filterwarnings("ignore", message=".*not writable.*")

BANNER = "=" * 90

GGUF_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/"
    "snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/"
    "tinyllama-1.1b-chat-v1.0.Q5_0.gguf"
)
IRPA_PATH = GGUF_PATH.replace(".gguf", ".irpa")


def section(title):
    print(f"\n{BANNER}\n  {title}\n{BANNER}")


# ══════════════════════════════════════════════════════════════════════
# PART 1: GGUF FILE
# ══════════════════════════════════════════════════════════════════════
section("PART 1: GGUF FILE — binary container from llama.cpp")

reader = gguf.GGUFReader(GGUF_PATH)

print(f"\nFile: {GGUF_PATH}")
print(f"Size: {os.path.getsize(GGUF_PATH) / 1024**2:.1f} MB")

print(f"\n  ─── HEADER ───")
magic = "N/A"
if "general.architecture" in reader.fields:
    magic = reader.fields['general.architecture'].parts[-1].tobytes().decode('utf-8')
print(f"  Magic:         {magic}")
print(f"  Total tensors: {len(reader.tensors)}")
print(f"  Metadata keys: {len(reader.fields)}")

print(f"\n  ─── METADATA (showing first 5 of {len(reader.fields)}) ───")
for i, (key, field) in enumerate(reader.fields.items()):
    if i >= 5:
        print(f"  ... and {len(reader.fields) - 5} more")
        break
    try:
        if field.types[0] == gguf.GGUFValueType.STRING:
            value = field.parts[-1].tobytes().decode("utf-8")[:50]
        elif field.types[0] == gguf.GGUFValueType.ARRAY:
            value = f"<array of {len(field.data)} items>"
        else:
            value = str(field.parts[-1][0])
    except Exception:
        value = "<complex>"
    print(f"  {key:40s} = {value}")

print(f"\n  ─── TENSOR INFO (showing first 5 of {len(reader.tensors)}) ───")
print(f"  {'Name':<40s} {'Shape':<20s} {'Type':<8s} {'Offset':<12s}")
print(f"  {'-'*40} {'-'*20} {'-'*8} {'-'*12}")
for t in reader.tensors[:5]:
    shape_str = str([int(d) for d in t.shape])
    print(f"  {t.name:<40s} {shape_str:<20s} {t.tensor_type.name:<8s} {int(t.data_offset):<12d}")
print(f"  ... and {len(reader.tensors) - 5} more tensors")

print(f"\n  ─── RAW BYTES of one Q5_0 tensor (blk.0.attn_q.weight) ───")
target = next(t for t in reader.tensors if t.name == "blk.0.attn_q.weight")
raw = np.asarray(target.data, dtype=np.uint8).flatten()
print(f"  Total bytes: {raw.size:,}")
print(f"  First 22 bytes (one block):")
byte_hex = " ".join(f"{int(b):02X}" for b in raw[:22])
print(f"    {byte_hex}")
print(f"    └─┬─┘ └─────┬───────┘ └──────────────────┬────────────┘")
print(f"     d(2B)    qh(4B)                          qs(16B)")


# ══════════════════════════════════════════════════════════════════════
# PART 2: DATASET
# ══════════════════════════════════════════════════════════════════════
section("PART 2: DATASET — Python object in memory")

ds = gguf_interop.load_file(GGUF_PATH)

print(f"\nDataset type: {type(ds).__name__}")
print(f"  Properties keys: {len(ds.properties)}")
print(f"  root_theta: {type(ds.root_theta).__name__}")

print(f"\n  ─── PROPERTIES (first 5) ───")
for i, (k, v) in enumerate(ds.properties.items()):
    if i >= 5:
        print(f"  ... and {len(ds.properties) - 5} more")
        break
    v_str = str(v)[:60] + ("..." if len(str(v)) > 60 else "")
    print(f"  {k:40s} = {v_str}")

all_tensors = dict(ds.root_theta.flatten())
print(f"\n  ─── TENSORS (first 5 of {len(all_tensors)}) ───")
print(f"  {'Name':<40s} {'Type':<24s} {'Shape':<20s}")
print(f"  {'-'*40} {'-'*24} {'-'*20}")
for i, (name, t) in enumerate(all_tensors.items()):
    if i >= 5:
        print(f"  ... and {len(all_tensors) - 5} more")
        break
    shape_str = str(list(t.shape))
    print(f"  {name:<40s} {type(t).__name__:<24s} {shape_str:<20s}")

print(f"\n  ─── ZOOM IN: one Q5_0 tensor (blk.0.attn_q.weight) ───")
t = ds.root_theta.tensor("blk.0.attn_q.weight")
print(f"  Type:         {type(t).__name__}")
print(f"  Shape:        {t.shape}")
print(f"  Layout:       {t.layout_type.__name__}")
print(f"  Raw shape:    {tuple(t.raw.shape)}  (lazy memmap)")
print(f"  Raw dtype:    {t.raw.dtype}")
print(f"  Raw bytes:    {t.raw.numel():,}")

layout = t.unpack()
print(f"\n  After unpack() → BlockScaledQ5_0Layout:")
d_shape_str = str(tuple(layout.d.shape))
qh_shape_str = str(tuple(layout.qh.shape))
qs_shape_str = str(tuple(layout.qs.shape))
print(f"    d:  {d_shape_str:<22s} {str(layout.d.dtype):<18s} ({layout.d.numel():,} elements)")
print(f"    qh: {qh_shape_str:<22s} {str(layout.qh.dtype):<18s} ({layout.qh.numel():,} elements)")
print(f"    qs: {qs_shape_str:<22s} {str(layout.qs.dtype):<18s} ({layout.qs.numel():,} elements)")

print(f"\n  ─── FIRST BLOCK: row 0, block 0 ───")
print(f"  d value:   {layout.d[0, 0, 0].item():.6f}  (scale)")
print(f"  qh bytes:  {layout.qh[0, 0].tolist()}  (4 bytes = 32 high bits)")
print(f"  qs bytes:  {layout.qs[0, 0].tolist()}  (16 bytes = 32 low nibbles)")

q = layout.unpack_q()
print(f"\n  First 8 five-bit values (q[0, 0, 0:8]):  {q[0, 0, :8].tolist()}")

w = layout.dequant(dtype=torch.float32)
print(f"\n  First 8 dequantized floats (w[0, :8]):   {w[0, :8].tolist()}")


# ══════════════════════════════════════════════════════════════════════
# PART 3: IRPA FILE
# ══════════════════════════════════════════════════════════════════════
section("PART 3: IRPA FILE — amdsharktank's format")

if not os.path.exists(IRPA_PATH):
    print(f"  IRPA file not found. Creating a sample...")
    sample_names = ["blk.0.attn_q.weight", "blk.0.ffn_up.weight"]
    sample_dict = {n: ds.root_theta.tensor(n) for n in sample_names
                   if n in all_tensors}
    sample_ds = Dataset(properties=ds.properties, root_theta=Theta(sample_dict))
    sample_path = "/tmp/sample_q5_0.irpa"
    sample_ds.save(sample_path)
    print(f"  Sample IRPA created: {sample_path}")
    IRPA_PATH = sample_path

print(f"\nFile: {IRPA_PATH}")
print(f"Size: {os.path.getsize(IRPA_PATH) / 1024**2:.1f} MB")

ds_irpa = Dataset.load(IRPA_PATH)

print(f"\n  ─── METADATA ───")
print(f"  Properties: {len(ds_irpa.properties)} keys")
print(f"  Tensors:    {len(ds_irpa.root_theta.flatten())}")

irpa_tensors = dict(ds_irpa.root_theta.flatten())
print(f"\n  ─── TENSORS (first 3) ───")
print(f"  {'Name':<40s} {'Type':<24s} {'Layout':<24s}")
print(f"  {'-'*40} {'-'*24} {'-'*24}")
for i, (name, t) in enumerate(irpa_tensors.items()):
    if i >= 3:
        print(f"  ... and {len(irpa_tensors) - 3} more")
        break
    print(f"  {name:<40s} {type(t).__name__:<24s} {t.layout_type.__name__:<24s}")

print(f"\n  ─── ZOOM IN: one tensor inside IRPA ───")
sample_name = next(iter(irpa_tensors.keys()))
t = ds_irpa.root_theta.tensor(sample_name)

print(f"  Tensor name: {sample_name}")
print(f"  Type:        {type(t).__name__}")
print(f"  Shape:       {t.shape}")
print(f"  Layout:      {t.layout_type.__name__}")

if hasattr(t, 'layout'):
    layout_irpa = t.layout
else:
    layout_irpa = t.unpack()

planes = layout_irpa.planes
print(f"\n  Planes (already materialized, NOT lazy):")
for pname in ["d", "qh", "qs"]:
    pt = planes[pname]
    shape_str = str(tuple(pt.shape))
    print(f"    {pname:3s}: shape={shape_str:<22s} dtype={str(pt.dtype):<18s} numel={pt.numel():,}")

print(f"\n  First block values (from IRPA):")
print(f"    d[0, 0, 0]:  {planes['d'][0, 0, 0].item():.6f}")
print(f"    qh[0, 0]:    {planes['qh'][0, 0].tolist()}")
print(f"    qs[0, 0]:    {planes['qs'][0, 0].tolist()}")


# ══════════════════════════════════════════════════════════════════════
# PART 4: SIDE-BY-SIDE
# ══════════════════════════════════════════════════════════════════════
section("PART 4: SIDE-BY-SIDE — GGUF vs Dataset vs IRPA")

print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │                          GGUF FILE                                │
  │  Format: Binary container (llama.cpp)                            │
  │  Layout: Header | Metadata | Tensor Info | Padding | Tensor Data │
  │  Raw bytes: 22-byte blocks, interleaved (d + qh + qs in one block)│
  │  Total tensors: 201 (mixed Q5_0, Q6_K, F32)                      │
  │  Size: 731.5 MB                                                   │
  ├──────────────────────────────────────────────────────────────────┤
  │                           DATASET                                 │
  │  Format: Python object (in-memory, Lazy)                         │
  │  Contains: properties dict + Theta (dict of tensors)             │
  │  201 tensors, mostly Q5_0 objects (loader classes)               │
  │  Bytes in RAM: ~1 MB (just metadata; bytes still on disk)        │
  ├──────────────────────────────────────────────────────────────────┤
  │                          IRPA FILE                                │
  │  Format: Binary container (amdsharktank)                         │
  │  Layout: Header | Metadata | Tensor Data (planes separated)      │
  │  Planes: d, qh, qs stored as SEPARATE tensors                    │
  │  Naming: "blk.0.attn_q.weight:d", ":qh", ":qs"                   │
  │  155 Q5_0 tensors + 46 non-Q5_0 (or skipped)                     │
  │  Size: ~678 MB (Q5_0-only) or ~731 MB (all)                      │
  └──────────────────────────────────────────────────────────────────┘
""")

print(f"  ─── One tensor across three formats ───")
print(f"  Tensor name: blk.0.attn_q.weight  (shape [2048, 2048])")
print(f"")
print(f"  ┌─ In GGUF file ─────────────────────────────────────────┐")
print(f"  │  Raw: 2,883,584 bytes = 131,072 blocks × 22 bytes     │")
print(f"  │  Each block: [d(2B) qh(4B) qs(16B)]                    │")
print(f"  │  Interleaved: d, qh, qs, d, qh, qs, ...                │")
print(f"  └────────────────────────────────────────────────────────┘")
print(f"")
print(f"  ┌─ In Dataset (in-memory, lazy) ─────────────────────────┐")
print(f"  │  Q5_0 object:                                          │")
print(f"  │    .raw = lazy memmap → 2,883,584 bytes on disk        │")
print(f"  │    .shape = [2048, 2048]                               │")
print(f"  │  0 bytes of raw data in RAM yet                        │")
print(f"  └────────────────────────────────────────────────────────┘")
print(f"")
print(f"  ┌─ In IRPA file (saved) ─────────────────────────────────┐")
print(f"  │  Three SEPARATE tensors:                               │")
print(f"  │    name:d   → 131,072 × 2 bytes  = 262,144 bytes      │")
print(f"  │    name:qh  → 524,288 × 1 byte   = 524,288 bytes      │")
print(f"  │    name:qs  → 2,097,152 × 1 byte = 2,097,152 bytes    │")
print(f"  │  Total: 2,883,584 bytes (same as GGUF block)          │")
print(f"  │  BUT: not interleaved — separated by plane            │")
print(f"  └────────────────────────────────────────────────────────┘")

print(f"\n✓ Inspection complete.")