#!/usr/bin/env python3
"""
run_sweep.py -- one driver for every local tournament:
                model vs model, model-size comparisons, and one-parameter sweeps.

USAGE
    python3 run_sweep.py spec.yaml             # run
    python3 run_sweep.py spec.yaml --dry-run   # print what would run; export/run nothing

SPEC FILE
    config_file: config/play_local_tournament.yaml

    tournament:                      # how the matches are played
      schedule: round_robin          # round_robin | gauntlet
      baseline: s100k                # gauntlet: everyone plays this one (required)
                                     # round_robin: Elo anchor only (optional)
                                     # sweep mode: a VALUE from sweep.values, e.g. 100
      search_depths: [800]           # optional; default mcts.search_depth from config
      fixed_overrides: []            # optional; applied to BOTH sides of every pairing
      match_timeout_s: 0             # optional; 0 = no timeout

    then EITHER
    players:                         # explicit list
      - name: s080k
        checkpoint: /app/train_dir/models/checkpoints/step_080000_model.pth
      - name: big
        checkpoint: /app/train_dir/models/big/step_050000_model.pth
        model_file: /app/config/model_big.yaml   # arch for ONNX export (default: global.model_file)
        node_scale: 0.5                          # plays 0.5 x depth nodes
      - name: s100k_sh
        onnx: /app/some/exported.onnx            # already-exported model
        overrides: [mcts.search=halving]         # applied after fixed_overrides
    OR
    sweep:                           # one model, one parameter, several values
      model: {checkpoint: /app/train_dir/models/checkpoints/step_100000_model.pth}
      param: mcts.gumbel.sigma_cap
      label: cap
      values: [25, 50, 100, 200, 400]

OUTPUT (one timestamped folder per run, under global.log_dir from the config)
    <log_dir>/<timestamp>/
        run.log                         this script's progress + results
        config/                         spec, tournament config (the copy actually run),
                                        model arch yaml(s), git_commit.txt
        onnx_cache/                       models exported for this run (+ TRT engines)
        depth_<depth>/<A>_vs_<B>/         games.pgn, summary.txt, C++ logs
        depth_<depth>/all_games.pgn       all pairings at that depth, for Ordo
        depth_<depth>/ratings.txt         Ordo table

NOTE: talbot_tournament reads input_planes from the config's global.model_file
for BOTH sides. Models of different SIZE are fine; a different INPUT ENCODING
is not supported.
"""

import argparse
import hashlib
import itertools
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

log = logging.getLogger("run_sweep")

SPEC_KEYS       = {"config_file", "tournament", "players", "sweep"}
TOURNAMENT_KEYS = {"schedule", "baseline", "search_depths", "fixed_overrides", "match_timeout_s"}
SWEEP_KEYS      = {"model", "param", "label", "values"}
PLAYER_KEYS     = {"name", "checkpoint", "onnx", "model_file", "overrides", "node_scale"}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def fmt_value(v):
    """Compact, filesystem- and PGN-safe rendering of a parameter value."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def check_keys(where, d, allowed):
    unknown = set(d) - allowed
    if unknown:
        sys.exit(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


def check_override(ov):
    key = ov.split("=", 1)[0]
    if "=" not in ov or "." not in key or key.split(".")[0] not in ("mcts", "selection"):
        sys.exit(f"override must look like mcts.<path>=<value> or selection.<path>=<value> (e.g. mcts.gumbel.explore_c=0.2), got {ov!r}")


def read_summary(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def setup_logging(log_file):
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="w"))
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)


# -----------------------------------------------------------------------------
# who plays
# -----------------------------------------------------------------------------
def players_from_spec(spec, baseline):
    """Return (players, baseline_name). Each player:
       {name, model: {checkpoint|onnx, model_file?}, overrides, node_scale}"""
    if ("players" in spec) == ("sweep" in spec):
        sys.exit("spec needs exactly one of 'players:' or 'sweep:'")

    if "sweep" in spec:
        sw = spec["sweep"]
        check_keys("sweep", sw, SWEEP_KEYS)
        param, values = sw["param"], sw["values"]
        label = sw.get("label", param.split(".")[-1])
        check_override(f"{param}=0")
        if len(set(map(fmt_value, values))) != len(values):
            sys.exit("sweep values must be distinct")
        players = [{"name": f"{label}{fmt_value(v)}",
                    "model": sw["model"],
                    "overrides": [f"{param}={fmt_value(v)}"],
                    "node_scale": 1.0} for v in values]
        if baseline is not None:
            if baseline not in values:
                sys.exit(f"tournament.baseline {baseline!r} must be one of sweep.values {values!r}")
            baseline = f"{label}{fmt_value(baseline)}"
        return players, baseline

    players = []
    for i, p in enumerate(spec["players"]):
        check_keys(f"players[{i}]", p, PLAYER_KEYS)
        if ("checkpoint" in p) == ("onnx" in p):
            sys.exit(f"players[{i}]: give exactly one of 'checkpoint' or 'onnx'")
        src = p.get("checkpoint") or p.get("onnx")
        ovs = list(p.get("overrides") or [])
        for ov in ovs:
            check_override(ov)
        players.append({"name": p.get("name", Path(src).stem),
                        "model": {k: p[k] for k in ("checkpoint", "onnx", "model_file") if k in p},
                        "overrides": ovs,
                        "node_scale": float(p.get("node_scale", 1.0))})

    names = [p["name"] for p in players]
    if len(set(names)) != len(names):
        sys.exit(f"player names must be distinct: {names}")
    if baseline is not None and baseline not in names:
        sys.exit(f"tournament.baseline {baseline!r} is not a player name {names}")
    return players, baseline


def build_pairs(names, schedule, baseline):
    if schedule == "gauntlet":
        if baseline is None:
            sys.exit("tournament.schedule: gauntlet requires tournament.baseline")
        return [(baseline, n) for n in names if n != baseline]
    if schedule == "round_robin":
        return list(itertools.combinations(names, 2))
    sys.exit(f"tournament.schedule must be 'gauntlet' or 'round_robin', got {schedule!r}")


# -----------------------------------------------------------------------------
# checkpoint -> ONNX (cached in <log_dir>/onnx_cache across runs)
# -----------------------------------------------------------------------------
def export_pth_to_onnx(pth_path, onnx_path, arch_cfg):
    # Imported lazily so runs over existing ONNX files don't need torch.
    import warnings
    import torch
    from model import ChessAIModel, fuse_bn_for_export

    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    model = ChessAIModel(arch_cfg)
    state = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    model = fuse_bn_for_export(model).cuda().half()

    planes = arch_cfg["model"]["input_planes"]
    dims = arch_cfg["model"]["board_dim"]
    dummy = torch.zeros(1, planes, dims, dims, dtype=torch.float16, device="cuda")

    tmp = onnx_path.with_suffix(".onnx.tmp")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            model, dummy, str(tmp),
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=["input"], output_names=["policy", "value"],
            dynamic_axes={"input": {0: "batch_size"},
                          "policy": {0: "batch_size"},
                          "value": {0: "batch_size"}},
            dynamo=False)
    tmp.replace(onnx_path)   # atomic: a crash mid-export never leaves a half-written .onnx

    del model, dummy
    torch.cuda.empty_cache()


def resolve_onnx(model, default_arch, cache_dir, dry_run, memo):
    """model: {checkpoint|onnx, model_file?} -> onnx path (str)."""
    if "onnx" in model:
        p = Path(model["onnx"])
        if not dry_run and not p.exists():
            sys.exit(f"onnx not found: {p}")
        return str(p)

    pth = Path(model["checkpoint"])
    arch_file = model.get("model_file", default_arch)
    key = (str(pth.resolve()), str(arch_file))
    if key in memo:
        return memo[key]

    # Hash path + arch so two checkpoints with the same file name (or one
    # checkpoint exported with two arch files) never share a cache entry.
    tag = hashlib.sha1("|".join(key).encode()).hexdigest()[:8]
    onnx = cache_dir / f"{pth.stem}_{tag}.onnx"
    memo[key] = str(onnx)

    if dry_run:
        log.info(f"  [dry-run] {pth} -> {onnx}")
        return str(onnx)
    if not pth.exists():
        sys.exit(f"checkpoint not found: {pth}")

    if onnx.exists() and onnx.stat().st_size > 0 and onnx.stat().st_mtime >= pth.stat().st_mtime:
        log.info(f"  cached ONNX: {onnx.name}")
        return str(onnx)

    log.info(f"  exporting {pth.name} -> {onnx.name}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # talbot_tournament treats any existing <name>.engine as valid, so a stale
    # engine from an older export must be removed or it would be silently reused.
    onnx.with_suffix(".engine").unlink(missing_ok=True)
    export_pth_to_onnx(pth, onnx, load_yaml(arch_file))
    return str(onnx)


def snapshot_configs(run_dir, spec_path, config_path, cfg, players):
    """Copy every config this run depends on into <run_dir>/config/ and
    return the path of the copied tournament config. The C++ binary is run
    on that copy, so editing the original mid-run cannot change later
    pairings, and the saved file is exactly what was played."""
    cfg_dir = run_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(spec_path, cfg_dir / Path(spec_path).name)
    config_copy = cfg_dir / Path(config_path).name
    shutil.copy2(config_path, config_copy)

    # Architecture configs: the shared one plus any per-player overrides.
    arch_files = {cfg["global"]["model_file"]}
    arch_files |= {p["model"]["model_file"] for p in players if "model_file" in p["model"]}
    for f in sorted(arch_files):
        src = Path(f)
        if src.exists():
            dst = cfg_dir / src.name
            if dst.exists():                       # two arch files with the same name
                dst = cfg_dir / f"{src.parent.name}_{src.name}"
            shutil.copy2(src, dst)

    # Code version, if this is a git checkout.
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
        (cfg_dir / "git_commit.txt").write_text(
            head + ("\n(uncommitted changes present)\n" if dirty else "\n"))
    except (OSError, subprocess.CalledProcessError):
        pass

    return config_copy


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="players/sweep YAML")
    ap.add_argument("--dry-run", action="store_true", help="print commands; export and run nothing")
    args = ap.parse_args()

    spec = load_yaml(args.spec)
    check_keys(args.spec, spec, SPEC_KEYS)
    if "config_file" not in spec:
        sys.exit(f"{args.spec}: config_file is required")
    t = spec.get("tournament") or {}
    check_keys("tournament", t, TOURNAMENT_KEYS)

    config_path = spec["config_file"]
    cfg = load_yaml(config_path)
    g = cfg["global"]

    exe       = g["exe_path"]
    log_dir   = Path(g["log_dir"])
    run_dir   = log_dir / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    cache_dir = run_dir / "onnx_cache"
    ordo      = g.get("ordo_path", "ordo")

    depths   = t.get("search_depths") or [cfg["mcts"]["search_depth"]]
    fixed    = list(t.get("fixed_overrides") or [])
    timeout  = int(t.get("match_timeout_s", 0)) or None
    for ov in fixed:
        check_override(ov)

    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(None if args.dry_run else run_dir / "run.log")

    players, baseline = players_from_spec(spec, t.get("baseline"))
    if len(players) < 2:
        sys.exit(f"need >= 2 players, have {len(players)}")
    if not args.dry_run:
        config_path = str(snapshot_configs(run_dir, args.spec, config_path, cfg, players))
    schedule = t.get("schedule", "gauntlet" if baseline else "round_robin")
    pairs = build_pairs([p["name"] for p in players], schedule, baseline)
    by_name = {p["name"]: p for p in players}

    log.info(f"=== {datetime.now():%Y-%m-%d %H:%M:%S}  spec={args.spec}  run_dir={run_dir}")
    log.info(f"config={config_path}  exe={exe}")
    log.info(f"{len(players)} players, {schedule}, depths={depths} -> "
             f"{len(pairs) * len(depths)} pairings")

    # Phase 1: ONNX for every distinct model.
    log.info("=== Phase 1: models")
    memo = {}
    for p in players:
        p["onnx"] = resolve_onnx(p["model"], g["model_file"], cache_dir, args.dry_run, memo)
        log.info(f"  {p['name']:<20} {p['onnx']}  overrides={p['overrides']}  "
                 f"node_scale={p['node_scale']}")

    # Phase 2: matches.
    log.info("=== Phase 2: matches")
    total, idx, failed = len(pairs) * len(depths), 0, []
    for depth in depths:
        depth_dir = run_dir / f"depth_{depth}"
        for a, b in pairs:
            idx += 1
            pa, pb = by_name[a], by_name[b]
            name_a, name_b = a, b   # depth is already in the folder; names stay short
            pairing_dir = depth_dir / f"{name_a}_vs_{name_b}"
            tag = f"[{idx}/{total}] {name_a} vs {name_b}"

            cmd = [exe, "--config_file", config_path,
                   "--model_a", pa["onnx"], "--model_b", pb["onnx"],
                   "--run_dir", str(depth_dir),
                   "--name_a", name_a, "--name_b", name_b]
            for side, p in (("a", pa), ("b", pb)):
                nodes = max(1, round(depth * p["node_scale"]))
                for ov in [f"mcts.search_depth={nodes}", *fixed, *p["overrides"]]:
                    cmd += [f"--override_{side}", ov]

            if args.dry_run:
                log.info(f"{tag}\n  " + " ".join(cmd))
                continue

            log.info(f"{tag}: running")
            pairing_dir.mkdir(parents=True, exist_ok=True)
            try:
                # C++ output goes straight to this console (it only prints on
                # fatal errors; everything else is in its own log files).
                rc = subprocess.run(cmd, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                log.error(f"{tag}: TIMED OUT after {timeout}s (killed)")
                failed.append(tag)
                continue
            if rc != 0 or not (pairing_dir / "summary.txt").exists():
                log.error(f"{tag}: FAILED (exit {rc}); see the error above and {pairing_dir}/tournament_main.log")
                failed.append(tag)
                continue

            s = read_summary(pairing_dir / "summary.txt")
            wa, d, wb = int(s.get("a_wins", 0)), int(s.get("draws", 0)), int(s.get("b_wins", 0))
            n = wa + d + wb
            pct = 100.0 * (wa + 0.5 * d) / n if n else float("nan")
            log.info(f"{tag}: +{wa} ={d} -{wb}  A scores {pct:.1f}%  ({s.get('wall_seconds')}s)")

    if args.dry_run:
        return

    # Phase 3: pool + rate per depth (successful pairings only).
    log.info("=== Phase 3: ratings")
    ordo_bin = shutil.which(str(ordo)) if ordo else None
    for depth in depths:
        depth_dir = run_dir / f"depth_{depth}"
        pgns = sorted(p.parent / "games.pgn" for p in depth_dir.glob("*_vs_*/summary.txt"))
        if not pgns:
            continue
        combined = depth_dir / "all_games.pgn"
        with combined.open("w") as out:
            for p in pgns:
                out.write(p.read_text())
                out.write("\n")
        log.info(f"depth_{depth}: pooled {len(pgns)} pairings -> {combined}")

        if ordo_bin is None:
            log.info(f"  Ordo not found at {ordo!r} -- set global.ordo_path in the config.")
            continue
        ratings = depth_dir / "ratings.txt"
        # Same flags as the external-engine ratings: -W / -D estimate white
        # advantage and draw rate from the games; -s 1000 simulations for the
        # error bars. Internal runs anchor on the baseline player (-A) instead
        # of an anchor file (-m).
        ordo_cmd = [ordo_bin, "-p", str(combined), "-o", str(ratings), "-W", "-D", "-s", "1000"]
        if baseline is not None:
            ordo_cmd += ["-A", baseline, "-a", "0"]
        rc = subprocess.run(ordo_cmd, stdout=subprocess.DEVNULL).returncode
        if rc == 0:
            log.info(ratings.read_text())
        else:
            log.error(f"  Ordo exited {rc}: {' '.join(ordo_cmd)}")

    if failed:
        log.error(f"{len(failed)} pairing(s) failed:")
        for t_ in failed:
            log.error(f"  {t_}")
        sys.exit(1)


if __name__ == "__main__":
    main()