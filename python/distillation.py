import os
import time
import yaml
import struct
import lmdb
import datetime
import torch
import random
import warnings
from trainer import TrainTask
from model import ChessAIModel, fuse_bn_for_export

current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

RL_DIR = os.path.abspath(os.path.join(root_dir, "rl_dir"))
RL_PARAMS_FILE = os.path.abspath(os.path.join(root_dir, "config", "rl_training.yaml"))
MODEL_FILE = os.path.abspath(os.path.join(root_dir, "config", "model.yaml"))
DB_PATH = os.path.abspath(os.path.join(RL_DIR, "replay_memory.lmdb"))

def run_distillation():
    with open(RL_PARAMS_FILE, 'r') as f:
        params_config = yaml.safe_load(f)
    with open(MODEL_FILE, 'r') as f:
        raw_model_config = yaml.safe_load(f)

    student_model_config = {
        'model': raw_model_config['student'],
        'chess': raw_model_config['chess']
    }

    base_path = os.path.join(root_dir, params_config['global']['model_path'])
    main_model_pth = os.path.abspath(base_path + ".pth")
    temp_seed_pth = os.path.abspath(base_path + "_temp_seed.pth")
    onnx_path = main_model_pth.replace(".pth", ".onnx")

    seed = params_config['training'].get('seed', 42)
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"Initializing brand new student model using seed: {seed}")

    fresh_model = ChessAIModel(student_model_config)
    os.makedirs(os.path.dirname(temp_seed_pth), exist_ok=True)
    torch.save({'model_state_dict': fresh_model.state_dict()}, temp_seed_pth)

    env = lmdb.open(DB_PATH, readonly=True, lock=False)
    with env.begin() as txn:
        cpp_blob = txn.get(b"__CPP_STATE")
        if not cpp_blob:
            raise RuntimeError("LMDB is empty. Cannot run distillation.")
        buffer_count = struct.unpack('QQdQQQ', cpp_blob)[3]
    env.close()

    batch_size = params_config['training']['batch_size']
    distill_passes = params_config['global']['distill_passes']
    
    target_steps = int((buffer_count / batch_size) * distill_passes)
    
    print(f"--- Starting Offline Distillation (New Architecture) ---")
    print(f"Buffer Size: {buffer_count} | Batch Size: {batch_size} | Passes: {distill_passes}")
    print(f"Target Training Steps: {target_steps}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    distill_log_dir = os.path.join(RL_DIR, f"distill_run_{timestamp}")
    os.makedirs(distill_log_dir, exist_ok=True)

    dummy_state = {'lifetime': {'training_steps': 0}}

    task = TrainTask(
        model_path=temp_seed_pth,
        model_config=student_model_config,
        training_config=params_config['training'],
        state_config=dummy_state,
        global_config=params_config['global'],
        db_path=DB_PATH
    )

    start_time = time.time()

    for step in range(target_steps):
        dummy_state['lifetime']['training_steps'] = step
        task.run_single_step(distill_log_dir, dummy_state)
        
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600.0

    print(f"\nDistillation complete in {elapsed_hours:.4f} hours.")
    print(f"Overwriting main model file at: {main_model_pth}")
    task.save_checkpoint(main_model_pth)
    task.cleanup()

    if os.path.exists(temp_seed_pth):
        os.remove(temp_seed_pth)

    # --- Export to ONNX ---
    print(f"Exporting new student model to ONNX at: {onnx_path}")
    export_model = ChessAIModel(student_model_config)
    checkpoint = torch.load(main_model_pth, map_location='cpu', weights_only=True)
    export_model.load_state_dict(checkpoint['model_state_dict'])
    export_model.eval()
    
    export_model = fuse_bn_for_export(export_model)
    export_model = export_model.cuda().half()

    dummy_input = torch.zeros(
        1, 
        student_model_config['chess']['input_planes'], 
        student_model_config['chess']['board_dim'], 
        student_model_config['chess']['board_dim'], 
        dtype=torch.float16
    ).cuda()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            export_model, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['policy', 'value'],
            dynamic_axes={'input': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}},
            dynamo=False
        )
    print("ONNX export successful.")

    # --- Write state and trigger TRT build ---
    print("Updating LMDB state and triggering C++ TRT build...")
    env_write = lmdb.open(DB_PATH, write=True, lock=True)
    with env_write.begin(write=True) as txn:
        # 1. Update ONLY the hours in PY_STATE
        py_blob = txn.get(b"__PY_STATE")
        if py_blob:
            current_steps, current_hours = struct.unpack('Qd', py_blob)
            new_hours = current_hours + elapsed_hours
            new_py_blob = struct.pack('Qd', current_steps, new_hours)
            txn.put(b"__PY_STATE", new_py_blob)
            
        # 2. Trigger TRT rebuild by incrementing the signal count directly
        export_blob = txn.get(b"__TRT_EXPORT_SIGNAL")
        if export_blob:
            last_signal = struct.unpack('Q', export_blob)[0]
            txn.put(b"__TRT_EXPORT_SIGNAL", struct.pack('Q', last_signal + 1))
            
    env_write.close()
    print("Distillation pipeline complete. The C++ engine should now intercept the signal and rebuild the engine.")

if __name__ == "__main__":
    run_distillation()