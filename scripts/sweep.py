#!/usr/bin/env python3
"""WarpSim parameter sweeps: measured IPC vs analytical prediction."""
import subprocess, csv, io, os
import matplotlib
matplotlib.use("Agg")            # no display needed; write files
import matplotlib.pyplot as plt

CLI = os.path.join("build", "warpsim_cli")
RESULTS = "results"

def run(**kwargs):
    """Run one CLI invocation, return the parsed CSV row as a dict."""
    args = [CLI]
    for k, v in kwargs.items():
        args += ["--" + k, str(v)]
    args += ["--header"]         # emit header so DictReader can parse
    out = subprocess.check_output(args, text=True)
    rows = list(csv.DictReader(io.StringIO(out)))
    return rows[0]

def occupancy_sweep():
    L = 4
    warps = [1, 2, 4, 8, 16, 32]
    measured, predicted = [], []
    for w in warps:
        row = run(workload="depchain", warps=w, n=5000, fp_latency=L)
        measured.append(float(row["ipc"]))
        predicted.append(min(1.0, w / L))

    plt.figure(figsize=(7, 4.5))
    plt.plot(warps, predicted, "-", color="#7f8c8d", linewidth=6, alpha=0.5,
             label=f"predicted  min(1, W/L),  L={L}")
    plt.plot(warps, measured, "o", color="#c0392b", markersize=8,
             markerfacecolor="white", markeredgewidth=2, label="measured")
    plt.xscale("log", base=2)
    plt.xticks(warps, [str(w) for w in warps])
    plt.xlabel("resident warps (W)")
    plt.ylabel("IPC")
    plt.title("Occupancy hides latency up to the issue-slot ceiling")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS, "occupancy.png")
    plt.savefig(path, dpi=130)
    print(f"wrote {path}")
    for w, m, p in zip(warps, measured, predicted):
        print(f"  W={w:2d}  measured={m:.4f}  predicted={p:.4f}")


def mshr_sweep():
    """Stream workload, L1 off: IPC rises with MSHRs, saturates at bandwidth."""
    W = 200            # memory latency
    B = 4              # dram txns/cycle (bandwidth ceiling for 1-txn loads)
    mshrs = [1, 4, 16, 64, 256, 512, 1024, 2048]
    measured, predicted = [], []
    for m in mshrs:
        row = run(workload="stream", warps=32, n=500,
                  l1_enabled=0, memory_latency=W, max_outstanding=m, dram_txns=B)
        measured.append(float(row["ipc"]))
        # Little's Law: throughput = M/W, capped by bandwidth B (1 txn/load).
        predicted.append(min(1.0, m / W))  # serialised channel caps at 1 access/cycle

    plt.figure(figsize=(7, 4.5))
    plt.plot(mshrs, predicted, "-", color="#7f8c8d", linewidth=6, alpha=0.5,
             label=f"predicted  min(1, M/W),  W={W}")
    plt.plot(mshrs, measured, "o", color="#c0392b", markersize=8,
             markerfacecolor="white", markeredgewidth=2, label="measured")
    plt.xscale("log", base=2)
    plt.xticks(mshrs, [str(m) for m in mshrs])
    plt.xlabel("outstanding requests / MSHRs (M)")
    plt.ylabel("IPC")
    plt.title("Memory-level parallelism: IPC = M/W until the channel saturates")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS, "mshr.png")
    plt.savefig(path, dpi=130)
    print(f"wrote {path}")
    for m, meas, p in zip(mshrs, measured, predicted):
        print(f"  M={m:3d}  measured={meas:.4f}  predicted={p:.4f}")


def execwidth_sweep():
    """Same knob (FP latency), two workloads: latency-bound responds, memory-bound doesn't."""
    latencies = [1, 2, 4, 8, 16]
    dep_meas, dep_pred, stream_meas = [], [], []
    for L in latencies:
        d = run(workload="depchain", warps=1, n=5000, fp_latency=L)
        dep_meas.append(float(d["ipc"]))
        dep_pred.append(1.0 / L)                       # dependent chain: IPC = 1/L
        s = run(workload="stream", warps=8, n=2000, fp_latency=L,
                l1_enabled=0, memory_latency=200, max_outstanding=8)
        stream_meas.append(float(s["ipc"]))            # memory-bound: independent of L

    plt.figure(figsize=(7, 4.5))
    plt.plot(latencies, dep_pred, "-", color="#7f8c8d", linewidth=6, alpha=0.5,
             label="predicted  1/L  (latency-bound)")
    plt.plot(latencies, dep_meas, "o", color="#c0392b", markersize=8,
             markerfacecolor="white", markeredgewidth=2,
             label="depchain measured (compute-bound)")
    plt.plot(latencies, stream_meas, "s", color="#2980b9", markersize=8,
             markerfacecolor="white", markeredgewidth=2,
             label="stream measured (memory-bound)")
    plt.xscale("log", base=2)
    plt.xticks(latencies, [str(x) for x in latencies])
    plt.xlabel("FP execution latency (L, cycles)")
    plt.ylabel("IPC")
    plt.title("The bottleneck decides: faster compute helps only compute-bound work")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS, "execwidth.png")
    plt.savefig(path, dpi=130)
    print(f"wrote {path}")
    for L, dm, dp, sm in zip(latencies, dep_meas, dep_pred, stream_meas):
        print(f"  L={L:2d}  depchain={dm:.4f} (pred {dp:.4f})   stream={sm:.4f}")

if __name__ == "__main__":
    occupancy_sweep()
    mshr_sweep()
    execwidth_sweep()
