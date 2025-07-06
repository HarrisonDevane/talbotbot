import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader, random_split
# Only import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm # For progress bars
from datetime import datetime
import os
import time

# Assuming your model and dataset are in these files
from model import ChessAIModel
# Import both ChessDataset and the worker_init_fn from data_loader
from data_loader import ChessDataset, _worker_init_fn

# Define constants (should match those in your model.py)
BOARD_DIM = 8
POLICY_CHANNELS = 73
TOTAL_POLICY_MOVES = BOARD_DIM * BOARD_DIM * POLICY_CHANNELS


def train_model(
    hdf5_path: str,
    log_dir: str,
    num_input_planes: int,
    num_residual_blocks: int = 10,
    num_filters: int = 128,
    batch_size: int = 256,
    learning_rate: float = 1e-5,
    num_epochs: int = 10,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
    log_interval: int = 100,
    save_interval: int = 5,
    checkpoint_dir: str = "checkpoints",
    validation_split: float = 0.5,
    resume_checkpoint_path: str = None,
    weight_decay: float = 1e-4,
    # Kept as arguments as requested
    scheduler_type: str = 'CosineAnnealingLR',
    cosine_eta_min: float = 1e-6
):
    # --- 1. Setup Logging ---
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"training_run_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # --- 2. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA is available, but no CUDA devices found!")
    else:
        logger.warning("CUDA is NOT available. Falling back to CPU.")

    # --- 3. Dataset and DataLoader ---
    logger.info(f"Loading data from HDF5 file: {hdf5_path}")
    full_dataset = ChessDataset(hdf5_path=hdf5_path)

    total_samples = len(full_dataset)
    val_samples = int(total_samples * validation_split)
    train_samples = total_samples - val_samples

    # Using a fixed seed for random_split to ensure reproducibility
    train_dataset, val_dataset = random_split(full_dataset, [train_samples, val_samples],
                                              generator=torch.Generator().manual_seed(42))

    num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
    if num_workers == 0:
        logger.warning("No CPU cores detected or setting num_workers to 0. Data loading might be slower.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )

    logger.info(f"Total dataset size: {total_samples} samples")
    logger.info(f"Training set size: {len(train_dataset)} samples ({len(train_loader)} batches)")
    logger.info(f"Validation set size: {len(val_dataset)} samples ({len(val_loader)} batches)")

    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # --- 4. Model Initialization ---
    model = ChessAIModel(
        num_input_planes=num_input_planes,
        num_residual_blocks=num_residual_blocks,
        num_filters=num_filters
    ).to(device)
    logger.info("Model initialized:")
    logger.info(model)

    # --- 5. Loss Functions ---
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # --- 6. Optimizer and Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # CosineAnnealingLR is now the only scheduler type, regardless of scheduler_type arg
    total_training_steps = num_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps, eta_min=cosine_eta_min)
    logger.info(f"Using CosineAnnealingLR with T_max={total_training_steps} (steps), eta_min={cosine_eta_min}")

    # --- 7. Resume Training Logic (Start Epoch and States) ---
    start_epoch = 0
    best_val_loss = float('inf') # Track best validation loss for scheduler/saving

    if resume_checkpoint_path:
        if os.path.exists(resume_checkpoint_path):
            logger.info(f"Resuming training from checkpoint: {resume_checkpoint_path}")
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # *** MODIFICATION START ***
            # Remove the line that loads the scheduler state:
            # if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            #     logger.info("Scheduler state loaded from checkpoint.")
            # else:
            #     logger.warning("No scheduler state found in checkpoint. Starting scheduler from scratch.")
            logger.info("Scheduler will be re-initialized from scratch, not loaded from checkpoint.")
            # *** MODIFICATION END ***

            # start_epoch = checkpoint['epoch'] + 1

            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                logger.info(f"Loaded best validation loss: {best_val_loss:.4f}")

            logger.info(f"Resumed training. Last completed epoch: {checkpoint['epoch']}. Next epoch to run: {start_epoch}")
        else:
            logger.warning(f"Resume checkpoint path '{resume_checkpoint_path}' does not exist. Starting new training.")
            resume_checkpoint_path = None # Reset to None to ensure new training starts

    # --- 8. Training Loop ---
    logger.info(f"\nStarting training from epoch {start_epoch + 1} for {num_epochs} total epochs...")
    for epoch in range(start_epoch, num_epochs):
        # --- Training Phase ---
        model.train()
        running_policy_loss = 0.0
        running_value_loss = 0.0
        running_total_loss = 0.0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", unit="batch")
        for batch_idx, (board_tensors, policy_indices, value_targets) in enumerate(pbar_train):
            batch_start_time = time.perf_counter()

            transfer_to_gpu_start = time.perf_counter()
            board_tensors = board_tensors.to(device, non_blocking=True)
            policy_indices = policy_indices.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            transfer_to_gpu_end = time.perf_counter()

            optimizer.zero_grad()

            forward_pass_start = time.perf_counter()
            policy_logits, value_outputs = model(board_tensors)
            value_outputs = value_outputs.squeeze(1) # Ensure value_outputs is 1D

            if device.type == 'cuda':
                torch.cuda.synchronize()
            forward_pass_end = time.perf_counter()

            # Ensure policy_indices is Long for CrossEntropyLoss
            policy_loss = policy_criterion(policy_logits, policy_indices.long())
            value_loss = value_criterion(value_outputs, value_targets)
            total_loss = (policy_loss * policy_loss_weight) + (value_loss * value_loss_weight)

            backward_pass_start = time.perf_counter()
            total_loss.backward()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            backward_pass_end = time.perf_counter()

            optimizer.step()
            scheduler.step() # Scheduler step is still performed after optimizer step

            running_policy_loss += policy_loss.item()
            running_value_loss += value_loss.item()
            running_total_loss += total_loss.item()

            batch_end_time = time.perf_counter()
            current_lr = optimizer.param_groups[0]['lr']
            pbar_train.set_postfix({
                'P_Loss': f'{policy_loss.item():.4f}',
                'V_Loss': f'{value_loss.item():.4f}',
                'T_Loss': f'{total_loss.item():.4f}',
                'LR': f'{current_lr:.6f}', # Added LR here
                'GPU_Xfer_ms': f'{(transfer_to_gpu_end - transfer_to_gpu_start)*1000:.2f}',
                'FW_ms': f'{(forward_pass_end - forward_pass_start)*1000:.2f}',
                'BW_ms': f'{(backward_pass_end - backward_pass_start)*1000:.2f}',
                'Batch_Total_ms': f'{(batch_end_time - batch_start_time)*1000:.2f}'
            })

            if batch_idx % log_interval == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: P_Loss={policy_loss.item():.4f}, "
                            f"V_Loss={value_loss.item():.4f}, T_Loss={total_loss.item():.4f}, "
                            f"LR={current_lr:.6f}, " # Log current LR
                            f"GPU Xfer: {(transfer_to_gpu_end - transfer_to_gpu_start)*1000:.2f}ms, "
                            f"FW: {(forward_pass_end - forward_pass_start)*1000:.2f}ms, "
                            f"BW: {(backward_pass_end - backward_pass_start)*1000:.2f}ms, "
                            f"Total Batch Time: {(batch_end_time - batch_start_time)*1000:.2f}ms")


        avg_policy_loss_train = running_policy_loss / len(train_loader)
        avg_value_loss_train = running_value_loss / len(train_loader)
        avg_total_loss_train = running_total_loss / len(train_loader)

        logger.info(f"--- Epoch {epoch+1} Train Summary ---")
        logger.info(f"Average Policy Loss: {avg_policy_loss_train:.4f}")
        logger.info(f"Average Value Loss: {avg_value_loss_train:.4f}")
        logger.info(f"Average Total Loss: {avg_total_loss_train:.4f}")
        logger.info(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}") # LR after all training batch steps

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        running_policy_loss_val = 0.0
        running_value_loss_val = 0.0
        running_total_loss_val = 0.0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", unit="batch")
        with torch.no_grad(): # Disable gradient calculations during validation
            for board_tensors, policy_indices, value_targets in pbar_val:
                board_tensors = board_tensors.to(device, non_blocking=True)
                policy_indices = policy_indices.to(device, non_blocking=True)
                value_targets = value_targets.to(device, non_blocking=True)

                policy_logits, value_outputs = model(board_tensors)
                value_outputs = value_outputs.squeeze(1)

                policy_loss = policy_criterion(policy_logits, policy_indices.long())
                value_loss = value_criterion(value_outputs, value_targets)
                total_loss = (policy_loss * policy_loss_weight) + (value_loss * value_loss_weight)

                running_policy_loss_val += policy_loss.item()
                running_value_loss_val += value_loss.item()
                running_total_loss_val += total_loss.item()

                pbar_val.set_postfix({
                    'P_Loss': f'{policy_loss.item():.4f}',
                    'V_Loss': f'{value_loss.item():.4f}',
                    'T_Loss': f'{total_loss.item():.4f}'
                })

        avg_policy_loss_val = running_policy_loss_val / len(val_loader)
        avg_value_loss_val = running_value_loss_val / len(val_loader)
        avg_total_loss_val = running_total_loss_val / len(val_loader)

        logger.info(f"--- Epoch {epoch+1} Validation Summary ---")
        logger.info(f"Average Policy Loss: {avg_policy_loss_val:.4f}")
        logger.info(f"Average Value Loss: {avg_value_loss_val:.4f}")
        logger.info(f"Average Total Loss: {avg_total_loss_val:.4f}")

        # Update best_val_loss based on the VALIDATION loss
        if avg_total_loss_val < best_val_loss:
            best_val_loss = avg_total_loss_val
            logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving best model...")
            # Save the BEST model (not just any periodic one)
            best_model_path = os.path.join(checkpoint_dir, "best_chess_ai_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # Still save scheduler state for full checkpointing if needed elsewhere
                'best_val_loss': best_val_loss,
            }, best_model_path)


        # Save checkpoint periodically
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"chess_ai_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch, # This is the epoch that was just COMPLETED
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # Still save scheduler state
                'best_val_loss': best_val_loss, # Save the current best_val_loss too
            }, checkpoint_path)
            logger.info(f"Model checkpoint saved to {checkpoint_path}")

    logger.info("\nTraining complete!")

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Using the provided paths from your original script
    hdf5_file_path = os.path.abspath(os.path.join(current_script_dir, "v4_hqgames_with_tactics_opmcts/data/combined_data.h5"))
    checkpoint_output_dir = os.path.abspath(os.path.join(current_script_dir, "v4_hqgames_with_tactics_opmcts/model"))
    log_dir_path = os.path.abspath(os.path.join(current_script_dir, "v4_hqgames_with_tactics_opmcts/logs"))

    os.makedirs(checkpoint_output_dir, exist_ok=True)

    train_model(
        hdf5_path=hdf5_file_path,
        log_dir=log_dir_path,
        num_input_planes=18,
        num_residual_blocks=20,
        num_filters=128,
        batch_size=512,
        learning_rate=1e-7,
        num_epochs=20,
        policy_loss_weight=1.0,
        value_loss_weight=5.0,
        save_interval=1,
        checkpoint_dir=checkpoint_output_dir,
        validation_split=0.5,
        resume_checkpoint_path=os.path.abspath(os.path.join(current_script_dir, "v4_hqgames_with_tactics_opmcts/model/best_chess_ai_model_v3.pth")), # Set to a checkpoint path if you want to resume
        # Kept as arguments as requested, even if their values aren't conditionally used inside train_model right now
        scheduler_type='CosineAnnealingLR',
        cosine_eta_min=1e-9
    )