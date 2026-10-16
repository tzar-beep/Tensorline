"""Weight-only quantisation: W8A16 (int8, per output channel) and W4A16 (uint4, group-wise).

Formats (for a Linear weight W of shape [N out, K in]):
  int8  q[N, K] int8, scale[N]:            W ~= q * scale[n]            (symmetric, |q| <= 127)
  int4  q[N, K/2] uint8, scale/minv[N, K/G]: W ~= q * scale + minv     (asymmetric, 0 <= q <= 15)
        G = 128 (one group = one Triton K-tile). Packing within each 128-wide group: byte j holds
        k = j in its low nibble and k = j + 64 in its high nibble, so a kernel can unpack a tile
        into its two contiguous halves with a mask and a shift, with no interleave needed.

Activations stay fp16; only the weights shrink. Decode is memory-bandwidth-bound (every step
streams all weights), so fewer weight bytes means faster steps, *if* dequantisation is fused into
the matmul (kernels.py). Unfused, `dequantize() + F.linear` saves memory but costs speed.
Used for large-M (prefill) calls, where the matmul is compute-bound and cuBLAS wins anyway, and
on CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 128
# Linear layers inside each decoder block that get quantised. Embeddings, norms and lm_head
# (tied to the embedding in small Qwen models) stay in fp16.
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# Mode "int4" is mixed precision (in the spirit of llama.cpp's Q4_K_M): the most quantisation-
# sensitive projections stay int8. On Qwen2.5-0.5B this cuts the WikiText-2 perplexity loss from
# +22.8% (all-int4) to +9.2% for 4% more weight memory, which beats shrinking the int4 group
# size to 32 (+11.6%) and keeps the fast 128-wide kernel. "int4-all" quantises everything to 4 bits.
SENSITIVE = ("v_proj", "down_proj")
MODES = ("int8", "int4", "int4-all")


def quantize_int8(w: torch.Tensor):
    w = w.float()
    scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def quantize_int4(w: torch.Tensor, group: int = GROUP):
    N, K = w.shape
    assert K % group == 0 and group % 2 == 0
    g = w.float().view(N, K // group, group)
    mn, mx = g.amin(dim=2), g.amax(dim=2)
    scale = ((mx - mn) / 15.0).clamp(min=1e-8)
    q = torch.round((g - mn[..., None]) / scale[..., None]).clamp(0, 15).to(torch.uint8)
    half = group // 2
    packed = q[..., :half] | (q[..., half:] << 4)              # [N, K/G, G/2]
    return packed.reshape(N, K // 2).contiguous(), scale, mn


def dequantize_int8(q, scale, dtype):
    return (q.to(dtype) * scale.to(dtype)[:, None])


def dequantize_int4(packed, scale, mn, dtype, group: int = GROUP):
    N = packed.shape[0]
    p = packed.view(N, -1, group // 2)
    q = torch.cat((p & 0xF, p >> 4), dim=2).to(dtype)          # [N, K/G, G]
    return (q * scale.to(dtype)[..., None] + mn.to(dtype)[..., None]).reshape(N, -1)


class QuantLinear(nn.Module):
    """Drop-in replacement for nn.Linear holding int8 or packed-int4 weights."""

    def __init__(self, linear: nn.Linear, mode: str, dtype: torch.dtype, group: int = GROUP):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.mode, self.dtype, self.group = mode, dtype, group
        w = linear.weight.data
        if mode == "int8":
            q, s = quantize_int8(w)
            self.register_buffer("qweight", q)
            self.register_buffer("scale", s.to(dtype))
            self.register_buffer("minv", None)
        elif mode == "int4":
            q, s, m = quantize_int4(w, group)
            self.register_buffer("qweight", q)
            self.register_buffer("scale", s.to(dtype))
            self.register_buffer("minv", m.to(dtype))
        else:
            raise ValueError(mode)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data.to(dtype), requires_grad=False)

    def dequantize(self) -> torch.Tensor:
        if self.mode == "int8":
            return dequantize_int8(self.qweight, self.scale, self.dtype)
        return dequantize_int4(self.qweight, self.scale, self.minv, self.dtype, self.group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        if x2.is_cuda and x2.shape[0] <= 64:
            from . import kernels
            if kernels.available():
                y = kernels.qmatmul(x2, self)
                if self.bias is not None:
                    y = y + self.bias
                return y.view(*shape[:-1], self.out_features)
        return F.linear(x, self.dequantize(), self.bias)

    def extra_repr(self):
        return f"{self.in_features}->{self.out_features}, {self.mode}"


def weight_bytes(module: nn.Module) -> int:
    seen, total = set(), 0
    for t in list(module.parameters()) + list(module.buffers()):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    return total


@torch.no_grad()
def quantize_model(model: nn.Module, mode: str, device: str, dtype: torch.dtype) -> int:
    """Replace target Linears in-place, layer by layer on `device`, so a model too big for VRAM in
    fp16 never has to be there in fp16. Returns the number of layers quantised."""
    assert mode in MODES, mode
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and child_name in TARGETS:
                layer_mode = "int8" if mode == "int8" or (mode == "int4" and child_name in SENSITIVE) else "int4"
                if layer_mode == "int4" and child.in_features % GROUP:
                    continue                                      # shape not supported: keep fp16
                child.to(device)
                setattr(mod, child_name, QuantLinear(child, layer_mode, dtype))
                child.to("meta")                                  # drop the fp16 copy right away
                n += 1
    return n
