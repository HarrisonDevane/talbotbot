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
    logger.info(f"  exporting {os.path.basename(pth_path)} -> ONNX")

    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=True)

    model = ChessAIModel(model_cfg)

    state_dict = checkpoint['model_state_dict']

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_state_dict[k.replace('module.', '')] = v

    model.load_state_dict(cleaned_state_dict)

    model.eval()
    model = fuse_bn_for_export(model)
    model = model.cuda().half()

    planes = model_cfg['chess']['input_planes']
    dims = model_cfg['chess']['board_dim']

    dummy_input = torch.zeros(
        1,
        planes,
        dims,
        dims,
        dtype=torch.float16,
        device='cuda'
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['policy', 'value'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'policy': {0: 'batch_size'},
                'value': {0: 'batch_size'}
            },
            dynamo=False
        )

    logger.info(f"  ONNX export complete: {onnx_path}")

    del model
    del dummy_input
    torch.cuda.empty_cache()

    return True

def run_match(exe_path, config_path, onnx_a, onnx_b, run_dir, timeout_s, logger):
    name_a = os.path.basename(onnx_a).replace(".onnx", "")
    name_b = os.path.basename(onnx_b).replace(".onnx", "")
    logger.info(f"  running: {name_a} vs {name_b}")

    cmd = [
        exe_path,
        "--tournament",
        "--config_file", config_path,
        "--model_a", onnx_a,
        "--model_b", onnx_b,
        "--run_dir", run_dir,
    ]

    # The C++ writes its own structured logs into <run_dir>/<A>_vs_<B>/.
    # Its stdout/stderr (fatal messages, TRT warnings) are folded into the
    # orchestrator's main.log via the logger -- no separate per-pairing file.
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
    
    source_dir = tournament_cfg['global']['checkpoint_dir']
    exe_path = tournament_cfg['global']['exe_path']
    target_models = tournament_cfg['tournament_models']
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(tournament_cfg['global']['log_dir'], timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
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
    logger.info(f"Pinned orchestrator to core(s): {main_cores}\n")
    logger.info("=== Phase 1: building engines ===")
    
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
        logger.error(f"\nFatal: only {len(onnx_export)} onnx file(s) built; need >= 2 for a tournament.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    logger.info("\n=== Phase 2: match play ===")
    pairings = list(itertools.combinations(onnx_export, 2))
    logger.info(f"Found {len(onnx_export)} engines. Running {len(pairings)} total matches.")
    
    match_timeout_s = int(tournament_cfg['tournament'].get('match_timeout_s', 7200))

    for onnx_a, onnx_b in pairings:
        run_match(exe_path, config_path, onnx_a, onnx_b, run_dir,
                  match_timeout_s, logger)

    logger.info("\nTournament execution complete. Temporary engines cleaned up.")
    shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()