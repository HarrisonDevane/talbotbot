"""
main_local_tournament.py -- orchestrates a round-robin between checkpoint models.

Flow:
  Phase 1: export every checkpoint (.pth) to ONNX in a per-run tmp directory.
  Phase 2: for each pairing (A, B), launch talbot_tournament.exe. The exe
           handles TRT engine building itself (shells out to
           talbot_trt_compile.exe, caches the .engine next to the .onnx). The
           SAME model in multiple pairings only compiles once because the
           tmp_dir is shared across all pairings in this run.

The C++ exe writes into <run_dir>/<A>_vs_<B>/:
  games.pgn    -- all games concatenated; consume this with your Elo script.
  summary.txt  -- W/D/L counts, mode, wall time. Grep-friendly key=value.
  main.log / game_worker_*.log / batcher_*.log -- diagnostics.

Elo/ratings are NOT computed here or in the C++ exe. Run your rating script
over the games.pgn files after this completes.
"""

import os
import yaml
import subprocess
import itertools
import shutil
import torch
import warnings
import logging
import psutil
from datetime import datetime

from model import ChessAIModel, fuse_bn_for_export


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def export_pth_to_onnx(pth_path, onnx_path, model_cfg, logger):
    """Export a checkpoint to ONNX. Skips if the ONNX already exists.
    The TRT engine (.engine) is built later by the C++ exe on demand."""
    if os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0:
        logger.info(f"  cached ONNX: {os.path.basename(onnx_path)}")
        return True

    logger.info(f"  exporting {os.path.basename(pth_path)} -> ONNX")

    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=True)
    model = ChessAIModel(model_cfg)

    state_dict = checkpoint['model_state_dict']
    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict)

    model.eval()
    model = fuse_bn_for_export(model)
    model = model.cuda().half()

    planes = model_cfg['model']['input_planes']
    dims = model_cfg['model']['board_dim']

    dummy_input = torch.zeros(
        1, planes, dims, dims,
        dtype=torch.float16, device='cuda'
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=['input'], output_names=['policy', 'value'],
            dynamic_axes={
                'input':  {0: 'batch_size'},
                'policy': {0: 'batch_size'},
                'value':  {0: 'batch_size'}
            },
            dynamo=False
        )

    logger.info(f"  ONNX export complete: {onnx_path}")

    del model
    del dummy_input
    torch.cuda.empty_cache()
    return True


def run_match(exe_path, config_path, onnx_a, onnx_b, run_dir, timeout_s, logger):
    """Launch talbot_tournament.exe for one pairing. The exe:
       - checks for a cached <name>.engine next to each <name>.onnx, and
         invokes talbot_trt_compile.exe if missing;
       - plays games_per_match games (config.tournament.games_per_match),
         each opening played twice with sides swapped;
       - writes games.pgn + summary.txt into <run_dir>/<A>_vs_<B>/."""
    name_a = os.path.basename(onnx_a).replace(".onnx", "")
    name_b = os.path.basename(onnx_b).replace(".onnx", "")
    logger.info(f"  running: {name_a} vs {name_b}")

    cmd = [
        exe_path,
        "--tournament",                       # legacy flag, harmless
        "--config_file", config_path,
        "--model_a", onnx_a,
        "--model_b", onnx_b,
        "--run_dir", run_dir,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        for line in proc.stdout:
            logger.info(f"    [{name_a}_vs_{name_b}] {line.rstrip()}")
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        logger.error(f"  ERROR: match {name_a} vs {name_b} timed out "
                     f"after {timeout_s}s -- killed.")
        return False

    if proc.returncode != 0:
        logger.error(f"  ERROR: match {name_a} vs {name_b} "
                     f"exited with code {proc.returncode}")
        return False
    return True


def main():
    config_path = "config/play_local_tournament.yaml"
    tournament_cfg = load_config(config_path)
    arch_cfg = load_config(tournament_cfg['global']['model_file'])

    # --- CPU Pinning ---
    main_cores = tournament_cfg['evaluation']['main_cores']
    p = psutil.Process()
    p.cpu_affinity(main_cores)

    source_dir    = tournament_cfg['global']['checkpoint_dir']
    exe_path      = tournament_cfg['global']['exe_path']
    target_models = tournament_cfg['tournament_models']

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(tournament_cfg['global']['log_dir'], timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # tmp_dir is SHARED across all pairings in this run so the C++ exe hits
    # its .engine cache on the second-and-later appearance of any model.
    # We clean it up at the end of the run; it's not persisted across runs.
    tmp_dir = os.path.join(run_dir, "tmp_engines")
    os.makedirs(tmp_dir, exist_ok=True)

    # --- Logging Setup ---
    log_path = os.path.join(run_dir, "main.log")
    logger = logging.getLogger("tournament_orchestrator")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    logger.info(f"Tournament run directory: {run_dir}")
    logger.info(f"Pinned orchestrator to core(s): {main_cores}")
    logger.info(f"Mode: {tournament_cfg['tournament'].get('mode', 'fixed')}")
    logger.info("")
    logger.info("=== Phase 1: exporting ONNX ===")

    onnx_export = []

    for pth_filename in target_models:
        pth_path = os.path.join(source_dir, pth_filename)
        if not os.path.exists(pth_path):
            logger.warning(f"  WARNING: {pth_path} not found. Skipping.")
            continue

        base_name = os.path.splitext(pth_filename)[0]
        onnx_path = os.path.join(tmp_dir, base_name + ".onnx")

        export_pth_to_onnx(pth_path, onnx_path, arch_cfg, logger)
        onnx_export.append(onnx_path)

    if len(onnx_export) < 2:
        logger.error(f"\nFatal: only {len(onnx_export)} onnx file(s) built; "
                     f"need >= 2 for a tournament.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    logger.info("")
    logger.info("=== Phase 2: match play ===")
    logger.info("(TRT engines will be built by the C++ exe on first use; "
                "cached in tmp_engines/ for subsequent pairings.)")
    pairings = list(itertools.combinations(onnx_export, 2))
    logger.info(f"Found {len(onnx_export)} engines. Running {len(pairings)} total matches.")

    match_timeout_s = int(tournament_cfg['tournament'].get('match_timeout_s', 7200))

    for onnx_a, onnx_b in pairings:
        run_match(exe_path, config_path, onnx_a, onnx_b, run_dir,
                  match_timeout_s, logger)

    logger.info("")
    logger.info("Tournament execution complete.")
    logger.info(f"Per-pairing results in: {run_dir}/<A>_vs_<B>/")
    logger.info("  - games.pgn   -> feed to your Elo/rating script")
    logger.info("  - summary.txt -> grep-friendly W/D/L counts")
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()