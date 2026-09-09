"""CUDA-graph timing: removes CPU/Triton launch overhead so we see the true
steady-state GPU cost and the real batch crossover. Run in a GPU allocation.
    python p1_graph.py
"""
import torch
from lib import make_case, plan, sa


def graph_time(fns, iters=500, warmup=50):
    """Capture fns() (a list of callables) into one CUDA graph, time replays."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for f in fns:
                f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1e3  # us


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"# {p.name} SMs={p.multi_processor_count} cc={p.major}.{p.minor}")
    print(f"\n{'batch':>6} {'chunks':>7} {'e2e_us':>8} {'decode_us':>10} "
          f"{'merge_us':>9} {'d+m':>7} {'graph_gap':>10}")
    for n in (1, 2, 4, 8, 16, 24, 32, 48, 64):
        case = make_case(num_reqs=n, seq_range=(8192, 8192), seed=0)
        pl = plan(case)
        t_e2e = graph_time([pl["run_decode"], pl["run_merge"]])
        t_dec = graph_time([pl["run_decode"]])
        t_mrg = graph_time([pl["run_merge"]])
        print(f"{n:>6} {pl['num_topk_chunks']:>7} {t_e2e:>8.2f} {t_dec:>10.2f} "
              f"{t_mrg:>9.2f} {t_dec+t_mrg:>7.2f} {t_e2e-t_dec-t_mrg:>10.2f}")


if __name__ == "__main__":
    main()
