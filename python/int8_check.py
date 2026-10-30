#!/usr/bin/env python3
"""
INT8 vs FP32 accuracy check for Talbot.

Modes (combinable):
  default            MinMax calibration, full report (policy, value, value BIAS).
  --methods ...      Compare calibration methods: minmax, percentile, entropy.
  --sensitivity      Leave one layer group in FP32 at a time (stem, each residual
                     block, all SE layers, policy_conv1) and report which exclusion
                     recovers the most value accuracy.

All inference runs on ONNX Runtime CPU (QDQ-simulated INT8). The deployed TensorRT
engine can differ; treat results as a measure of the quantization recipe, not of
the engine itself.

Examples:
  python int8_check.py --pth model.pth --model-yaml model.yaml --db replay_memory.lmdb
  python int8_check.py ... --methods minmax percentile entropy
  python int8_check.py ... --sensitivity --sens-n-calib 1024
"""
import argparse, os, sys, re, struct, tempfile, time, warnings
import numpy as np
sys.path.insert(0, "/app/python")
import yaml, lmdb, zstandard as zstd, torch, onnx
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      CalibrationMethod, QuantType, QuantFormat)
from model import ChessAIModel, fuse_bn_for_export

warnings.filterwarnings("ignore")
ort.set_default_logger_severity(3)

HEAD_ENTRY = ("/policy_conv2/", "/value_conv/")
METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "percentile": CalibrationMethod.Percentile,
    "entropy": CalibrationMethod.Entropy,
}
# Bins over the FP32 expected score, to see whether error is systematic by position type.
E_BINS = [(-1.01, -0.5, "losing"), (-0.5, -0.1, "worse"), (-0.1, 0.1, "level"),
          (0.1, 0.5, "better"), (0.5, 1.01, "winning")]


# ----------------------------------------------------------------------------- data

def head_node_names(graph, entry_prefixes):
    consumers = {}
    for n in graph.node:
        for t in n.input:
            consumers.setdefault(t, []).append(n)
    stack = [n for n in graph.node if n.name.startswith(entry_prefixes)]
    seen = set()
    while stack:
        n = stack.pop()
        if n.name in seen:
            continue
        seen.add(n.name)
        for t in n.output:
            stack.extend(consumers.get(t, []))
    return sorted(seen)


def load_positions(db_path, n, planes, dim, moves, buffer_gb, seed):
    # NOTE: decodes the board as bit-packed planes. If any plane is non-binary in the
    # real input (e.g. the normalised 50-move clock), it must be reconstructed here
    # the same way the trainer does, or the eval inputs are off-distribution.
    env = lmdb.open(db_path, map_size=1024**3 * buffer_gb, readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        count = struct.unpack('QQdQQQ', txn.get(b"__CPP_STATE"))[3]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(count, size=min(n * 2, count), replace=False)
    board_bytes = (planes * dim * dim + 7) // 8
    mask_bytes = (moves + 7) // 8
    total_in = planes * dim * dim
    dec = zstd.ZstdDecompressor()
    B, M = [], []
    with env.begin(write=False, buffers=True) as txn:
        for idx in idxs:
            blob = txn.get(f"{int(idx):020d}".encode("ascii"))
            if blob is None:
                continue
            buf = dec.decompress(blob)
            off_board = 2 + board_bytes
            off_mask = off_board + mask_bytes
            B.append(np.unpackbits(np.frombuffer(buf[2:off_board], dtype=np.uint8))[:total_in]
                     .reshape(planes, dim, dim).astype(np.float16))
            M.append(np.unpackbits(np.frombuffer(buf[off_board:off_mask], dtype=np.uint8))[:moves].astype(bool))
            if len(B) == n:
                break
    env.close()
    print(f"Loaded {len(B)} positions from buffer (count={count}).")
    return np.stack(B), np.stack(M)


class Reader(CalibrationDataReader):
    def __init__(self, x, bs):
        self.it = iter([{"input": x[i:i + bs]} for i in range(0, len(x), bs)])

    def get_next(self):
        return next(self.it, None)


# ----------------------------------------------------------------------------- models

def export_fp32(cfg, pth, out_path, planes, dim):
    model = ChessAIModel(cfg)
    ck = torch.load(pth, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    model = fuse_bn_for_export(model)
    orig = model.forward

    def fwd(inp):  # FP16 I/O, FP32 internals -- same as production export
        p, v = orig(inp.float())
        return p.half(), v.half()

    model.forward = fwd
    torch.onnx.export(model, torch.zeros(1, planes, dim, dim, dtype=torch.float16), out_path,
                      export_params=True, opset_version=17, do_constant_folding=True,
                      input_names=["input"], output_names=["policy", "value"],
                      dynamic_axes={"input": {0: "b"}, "policy": {0: "b"}, "value": {0: "b"}},
                      dynamo=False)


def quantize(fp32_path, out_path, x_cal, method, exclude, calib_batch, percentile):
    extra = {"WeightSymmetric": True, "ActivationSymmetric": True, "QuantizeBias": False}
    if method == "percentile":
        extra["CalibPercentile"] = percentile
    t0 = time.time()
    quantize_static(model_input=fp32_path, model_output=out_path,
                    calibration_data_reader=Reader(x_cal, calib_batch),
                    quant_format=QuantFormat.QDQ, calibrate_method=METHODS[method],
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
                    nodes_to_exclude=list(exclude), calibration_providers=["CPUExecutionProvider"],
                    extra_options=extra)
    return time.time() - t0


PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def run(path, x, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    # FP32 and INT8 always use the same provider list, so both run on the same backend.
    s = ort.InferenceSession(path, so, providers=PROVIDERS)
    P, V = [], []
    for i in range(0, len(x), 256):
        p, v = s.run(None, {"input": x[i:i + 256]})
        P.append(p)
        V.append(v)
    return np.concatenate(P).astype(np.float64), np.concatenate(V).astype(np.float64)


# ----------------------------------------------------------------------------- metrics

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metrics(P32, V32, P8, V8, mask):
    n = len(P32)
    ar = np.arange(n)
    q32 = softmax(np.where(mask, P32, -1e9))
    q8 = softmax(np.where(mask, P8, -1e9))
    best = q32.argmax(1)
    E32 = V32[:, 0] - V32[:, 2]
    E8 = V8[:, 0] - V8[:, 2]
    return {
        "n": n,
        "top1": (q8.argmax(1) == best).mean(),
        "top3": (np.argsort(-q8, 1)[:, :3] == best[:, None]).any(1).mean(),
        "kl": (q32 * (np.log(q32 + 1e-12) - np.log(q8 + 1e-12))).sum(1),
        "tv": 0.5 * np.abs(q32 - q8).sum(1),
        "pbest8": q8[ar, best].mean(),
        "pbest32": q32[ar, best].mean(),
        "E32": E32,
        "dE": E8 - E32,          # signed: INT8 minus FP32
        "dWDL": V8 - V32,        # signed, per component
    }


def summary(r):
    adE = np.abs(r["dE"])
    return (r["top1"], r["top3"], r["kl"].mean(), adE.mean(), np.percentile(adE, 95), r["dE"].mean())


def print_report(name, r):
    pct = np.percentile
    kl, tv, dE, dWDL = r["kl"], r["tv"], r["dE"], r["dWDL"]
    adE = np.abs(dE)
    n = r["n"]
    ci = 1.96 * dE.std(ddof=1) / np.sqrt(n)

    print(f"\n=== {name}: INT8 vs FP32 on {n} held-out positions ===")
    print("Policy, legal moves only:")
    print(f"  top-1 agreement              : {r['top1']:.3f}")
    print(f"  FP32 best in INT8 top-3      : {r['top3']:.3f}")
    print(f"  KL(FP32||INT8) mean/p95/max  : {kl.mean():.4f} / {pct(kl, 95):.4f} / {kl.max():.4f}")
    print(f"  prob mass moved mean/p95     : {tv.mean():.4f} / {pct(tv, 95):.4f}")
    print(f"  INT8 prob on FP32 best (mean): {r['pbest8']:.3f}  vs FP32's own {r['pbest32']:.3f}")

    print("Value (expected score E = W - L, range -1..1):")
    print(f"  |dE|  mean/p95/max           : {adE.mean():.4f} / {pct(adE, 95):.4f} / {adE.max():.4f}")
    print(f"  signed dE mean (INT8-FP32)   : {dE.mean():+.4f}  (95% CI +/- {ci:.4f})"
          + ("   <-- systematic bias" if abs(dE.mean()) > ci else "   (no detectable bias)"))
    print(f"  signed mean dW / dD / dL     : {dWDL[:, 0].mean():+.4f} / {dWDL[:, 1].mean():+.4f} / {dWDL[:, 2].mean():+.4f}")

    print("  By FP32 evaluation:            n      signed dE    |dE| mean")
    for lo, hi, label in E_BINS:
        sel = (r["E32"] >= lo) & (r["E32"] < hi)
        if sel.sum() == 0:
            continue
        print(f"    {label:<8} [{lo:+.1f},{hi:+.1f})   {sel.sum():>5}    {dE[sel].mean():+.4f}      {adE[sel].mean():.4f}")
    print("  Read: signed dE pulling toward 0 in 'winning'/'losing' bins = INT8 compresses")
    print("  evaluations toward a draw; a bias that search averaging cannot remove.")


def print_table(title, rows, base_adE=None):
    print(f"\n=== {title} ===")
    hdr = f"  {'config':<24}{'top1':>7}{'top3':>7}{'KL mean':>9}{'|dE| mean':>11}{'|dE| p95':>10}{'signed dE':>11}"
    if base_adE is not None:
        hdr += f"{'recovered':>11}"
    print(hdr)
    for name, s in rows:
        line = f"  {name:<24}{s[0]:>7.3f}{s[1]:>7.3f}{s[2]:>9.4f}{s[3]:>11.4f}{s[4]:>10.4f}{s[5]:>+11.4f}"
        if base_adE is not None:
            line += f"{(base_adE - s[3]) / base_adE:>10.0%}"
        print(line)


def sensitivity_groups(graph):
    names = [n.name for n in graph.node]
    groups = {"stem": [n for n in names if n.startswith("/initial_conv/")]}
    blocks = sorted({int(m.group(1)) for n in names if (m := re.match(r"^/residual_blocks\.(\d+)/", n))})
    for b in blocks:
        groups[f"block {b}"] = [n for n in names if n.startswith(f"/residual_blocks.{b}/")]
    groups["all SE layers"] = [n for n in names if "/se_fc" in n]
    groups["policy_conv1"] = [n for n in names if n.startswith("/policy_conv1/")]
    return {k: v for k, v in groups.items() if v}


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--model-yaml", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--n-calib", type=int, default=4096)
    ap.add_argument("--n-eval", type=int, default=4096)
    ap.add_argument("--calib-batch", type=int, default=128)
    ap.add_argument("--buffer-gb", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["minmax"], choices=list(METHODS))
    ap.add_argument("--percentile", type=float, default=99.999)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--sens-method", default="minmax", choices=list(METHODS))
    ap.add_argument("--sens-n-calib", type=int, default=1024,
                    help="smaller calibration set for sensitivity runs (one quantization per group)")
    ap.add_argument("--cpu", action="store_true", help="run both models on CPU instead of GPU")
    a = ap.parse_args()

    global PROVIDERS
    if a.cpu:
        PROVIDERS = ["CPUExecutionProvider"]

    torch.set_num_threads(a.threads)
    cfg = yaml.safe_load(open(a.model_yaml))
    m = cfg["model"]
    planes, dim, moves = m["input_planes"], m["board_dim"], m["total_policy_moves"]

    x, mask = load_positions(a.db, a.n_calib + a.n_eval, planes, dim, moves, a.buffer_gb, a.seed)
    x_cal, x_ev, m_ev = x[:a.n_calib], x[a.n_calib:], mask[a.n_calib:]
    if len(x_ev) == 0:
        sys.exit("Not enough positions for an eval set.")

    tmp = tempfile.mkdtemp(prefix="int8chk_")
    fp32_path = os.path.join(tmp, "fp32.onnx")
    export_fp32(cfg, a.pth, fp32_path, planes, dim)
    graph = onnx.load(fp32_path).graph
    head_excl = head_node_names(graph, HEAD_ENTRY)
    print(f"Excluding {len(head_excl)} head nodes from INT8 in all runs.")

    P32, V32 = run(fp32_path, x_ev, a.threads)

    # --- calibration method comparison
    rows = []
    for meth in a.methods:
        out = os.path.join(tmp, f"int8_{meth}.onnx")
        print(f"\nQuantizing with {meth} on {len(x_cal)} positions...")
        dt = quantize(fp32_path, out, x_cal, meth, head_excl, a.calib_batch, a.percentile)
        P8, V8 = run(out, x_ev, a.threads)
        r = metrics(P32, V32, P8, V8, m_ev)
        print_report(f"{meth} (calibration {dt:.0f}s)", r)
        rows.append((f"{meth} ({dt:.0f}s)", summary(r)))
        os.remove(out)
    if len(rows) > 1:
        print_table("Calibration method comparison", rows)

    # --- per-group sensitivity
    if a.sensitivity:
        xs = x_cal[:a.sens_n_calib]
        groups = sensitivity_groups(graph)
        print(f"\nSensitivity: {len(groups) + 1} quantizations with {a.sens_method} "
              f"on {len(xs)} calibration positions...")
        base_out = os.path.join(tmp, "int8_base.onnx")
        quantize(fp32_path, base_out, xs, a.sens_method, head_excl, a.calib_batch, a.percentile)
        P8, V8 = run(base_out, x_ev, a.threads)
        base = summary(metrics(P32, V32, P8, V8, m_ev))
        os.remove(base_out)

        srows = []
        for gname, gnodes in groups.items():
            out = os.path.join(tmp, "int8_sens.onnx")
            quantize(fp32_path, out, xs, a.sens_method, set(head_excl) | set(gnodes),
                     a.calib_batch, a.percentile)
            P8, V8 = run(out, x_ev, a.threads)
            srows.append((f"FP32: {gname}", summary(metrics(P32, V32, P8, V8, m_ev))))
            os.remove(out)
            print(f"  done: {gname}")
        srows.sort(key=lambda r: r[1][3])
        print_table("Sensitivity (each row keeps one group in FP32; sorted by |dE| mean)",
                    [("baseline (heads only)", base)] + srows, base_adE=base[3])
        print("  'recovered' = share of baseline |dE| removed by keeping that group in FP32.")

    print(f"\nNote: ONNX Runtime ({PROVIDERS[0]}), not the TensorRT engine.")


if __name__ == "__main__":
    main()