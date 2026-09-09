"""Discussion point 3 — experiment: does a Triton TMA descriptor accept a
*runtime* (data-dependent) page index into a paged KV cache, and does it lower
to cp.async.bulk.tensor (TMA hardware)?

  load_plain : tl.load(base + page*stride + ...)   <- baseline style
  load_tma   : desc.load([page*128, 0])            page from a runtime tensor
"""
import os
import sys

HARNESS = os.path.join(os.path.dirname(__file__), "..", "harness")
sys.path.insert(0, HARNESS)

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

OUT = "profiles"


@triton.jit
def load_plain(kv_ptr, pages_ptr, out_ptr, s_pg, s_pos, s_d,
               PAGE: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    page = tl.load(pages_ptr + pid).to(tl.int64)
    off_n = tl.arange(0, PAGE)
    off_d = tl.arange(0, D)
    x = tl.load(kv_ptr + page * s_pg + off_n[:, None] * s_pos + off_d[None, :] * s_d)
    tl.store(out_ptr + pid * PAGE * D + off_n[:, None] * D + off_d[None, :], x)


@triton.jit
def load_tma(desc, pages_ptr, out_ptr, PAGE: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    page = tl.load(pages_ptr + pid).to(tl.int32)
    x = desc.load([page * PAGE, 0])          # runtime row-band offset
    off_n = tl.arange(0, PAGE)
    off_d = tl.arange(0, D)
    tl.store(out_ptr + pid * PAGE * D + off_n[:, None] * D + off_d[None, :], x)


def dump(compiled, key):
    ptx = compiled.asm.get("ptx", "")
    sass = compiled.asm.get("sass", "")
    open(f"{OUT}/tma_{key}.ptx", "w").write(ptx)
    open(f"{OUT}/tma_{key}.sass", "w").write(sass)
    pk = ("cp.async.bulk", "bulk.tensor", "mbarrier", "tensormap",
          "cp.async.ca", "cp.async.cg")
    p = [l.strip() for l in ptx.splitlines() if any(s in l for s in pk)]
    sk = ("UTMALDG", "LDGSTS", "LDG.E", "UBLKCP", "TMA")
    s = [l.strip() for l in sass.splitlines() if any(x in l for x in sk)]
    print(f"\n=== {key}: PTX async/TMA lines ({len(p)}) ===")
    for l in p[:14]:
        print("  ", l)
    seen, u = set(), []
    for l in s:
        parts = l.split()
        tok = parts[1] if len(parts) > 1 else l
        if tok not in seen:
            seen.add(tok); u.append(l)
    print(f"=== {key}: distinct SASS load ops ({len(u)}) ===")
    for l in u[:14]:
        print("  ", l)


def main():
    torch.manual_seed(0)
    P, PAGE, D = 64, 128, 128
    kv = torch.randn(P, PAGE, D, device="cuda", dtype=torch.bfloat16)
    kv2 = kv.reshape(P * PAGE, D)
    pages = torch.randint(0, P, (8,), device="cuda", dtype=torch.int32)
    out = torch.empty(8, PAGE, D, device="cuda", dtype=torch.bfloat16)

    cp = load_plain.warmup(kv, pages, out, kv.stride(0), kv.stride(1), kv.stride(2),
                           PAGE, D, grid=(8,))
    load_plain[(8,)](kv, pages, out, kv.stride(0), kv.stride(1), kv.stride(2), PAGE, D)
    torch.cuda.synchronize()
    ok = all(torch.equal(out[i], kv[pages[i].item()]) for i in range(8))
    print("plain correct:", ok)
    dump(cp, "plain")

    try:
        desc = TensorDescriptor.from_tensor(kv2, [PAGE, D])
        ct = load_tma.warmup(desc, pages, out, PAGE, D, grid=(8,))
        load_tma[(8,)](desc, pages, out, PAGE, D)
        torch.cuda.synchronize()
        ok = all(torch.equal(out[i], kv[pages[i].item()]) for i in range(8))
        print("tma   correct:", ok)
        dump(ct, "tma")
    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
