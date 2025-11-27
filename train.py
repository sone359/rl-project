"""
Command-line training script for Walker2d-v5 with custom reward functions.

Usage examples
--------------

# 1) Train with reward "speed_energy" and default parameters:
python train_cli.py --reward speed_energy --timesteps 100000

# 2) Train with "target_speed" and custom hyperparameters:
python train_cli.py --reward target_speed \
    --reward-param v_target=1.5 \
    --reward-param alpha=1.0 \
    --reward-param beta=0.001 \
    --timesteps 150000

# 3) Train with "posture_stability" and record a video after training:
python train_cli.py --reward posture_stability \
    --reward-param h_target=1.25 \
    --reward-param w_h=5.0 \
    --reward-param w_angle=1.0 \
    --timesteps 150000 \
    --video
"""

import argparse
from typing import Dict, Any

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from walker2d_env import make_walker2d_env
from reward_function import REWARD_FNS

import numpy as np
import torch
import random
import os
import csv
import re
import shutil
from pathlib import Path

class LogMerger:
    def __init__(self, monitor_path: str, noise_path: str, enabled: bool, start_timesteps: int, end_timesteps: int = None):
        self.monitor_path = monitor_path
        self.noise_path = noise_path
        self.start_timesteps = start_timesteps
        self.end_timesteps = end_timesteps
        self.enabled = enabled
        self.temp_paths = {}
    
    def __enter__(self):
        if not self.enabled:
            return self

        # Archive existing files to .temp
        for path in [self.monitor_path, self.noise_path]:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                temp_path = path + ".temp"
                try:
                    shutil.move(path, temp_path)
                    self.temp_paths[path] = temp_path
                    print(f"[SmartLogMerger] Archived temporary log: {os.path.basename(path)} -> .temp")
                except OSError as e:
                    print(f"[SmartLogMerger] Error moving file {path}: {e}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        print("[LogMerger] Merging and cleaning up logs...")

        if self.monitor_path in self.temp_paths:
            self._merge_monitor()
            # Cleanup temp file
            if os.path.exists(self.temp_paths[self.monitor_path]):
                os.remove(self.temp_paths[self.monitor_path])

        # 2. Merge Noise CSV (Requires parsing 'timestep' column)
        if self.noise_path in self.temp_paths:
            self._merge_noise()
            # Cleanup temp file
            if os.path.exists(self.temp_paths[self.noise_path]):
                os.remove(self.temp_paths[self.noise_path])
    
    def _merge_csv_log(self):
        """
        Merge SB3 Monitor logs.
        Format:
            Line 1: Metadata (JSON)
            Line 2: Header (r, l, t)
            Line 3+: Data
        Logic: Sum 'l' (episode length) to determine the cumulative timestep.
        """
        if not os.path.exists(self.monitor_path):
            # If the script crashed before creating a new file, restore the old one
            if os.path.exists(self.temp_paths[self.monitor_path]):
                shutil.move(self.temp_paths[self.monitor_path], self.monitor_path)
            return

        kept_lines = []
        cumulative_steps = 0
        
        # 1. Read and filter old file
        if os.path.exists(self.temp_paths[self.monitor_path]):
            with open(self.temp_paths[self.monitor_path], 'r') as f:
                # Read Metadata
                meta_line = f.readline()
                if meta_line: kept_lines.append(meta_line)
                
                # Read Header
                header_line = f.readline()
                if header_line: kept_lines.append(header_line)
                
                reader = csv.DictReader(f, fieldnames=['r', 'l', 't'])
                
                for row in reader:
                    try:
                        l = int(row['l'])
                    except (ValueError, TypeError):
                        continue # Skip corrupted lines
                    
                    # Keep line if the episode ended at or before the checkpoint step and within start/end timesteps
                    if cumulative_steps + l <= self.start_timesteps and (self.end_timesteps is None or cumulative_steps < self.end_timesteps):
                        # Reconstruct CSV line
                        kept_lines.append(f"{row['r']},{row['l']},{row['t']}\n")
                        cumulative_steps += l
                    else:
                        # Stop reading once we pass the checkpoint step
                        break
        
        # 2. Read new file (skip headers)
        new_content = []
        if os.path.exists(self.monitor_path):
            with open(self.monitor_path, 'r') as f:
                # Skip the first two lines (Metadata + Header) of the new file
                f.readline() 
                f.readline()
                for row in reader:
                    try:
                        l = int(row['l'])
                    except (ValueError, TypeError):
                        continue # Skip corrupted lines
                    
                    # Only keep episodes that start after the checkpoint step and within end timesteps
                    if self.end_timesteps is None or cumulative_steps < self.end_timesteps:
                        new_content.append(f"{row['r']},{row['l']},{row['t']}\n")
                        cumulative_steps += l
                    else:
                        break

        # 3. Write combined content
        with open(self.monitor_path, 'w') as f:
            f.writelines(kept_lines)
            f.writelines(new_content)
        
        print(f"  -> Monitor merged. Preserved {max(0, len(kept_lines)-2)} historical episodes.")

    def _merge_noise(self):
        """
        Merge custom Noise logs.
        Format:
            Line 1: Header (timestep, ...)
            Line 2+: Data
        Logic: Filter based on the first column (timestep).
        """
        if not os.path.exists(self.noise_path):
            if os.path.exists(self.temp_paths[self.noise_path]):
                shutil.move(self.temp_paths[self.noise_path], self.noise_path)
            return

        kept_lines = []
        
        # 1. Read and filter old file
        if os.path.exists(self.temp_paths[self.noise_path]):
            with open(self.temp_paths[self.noise_path], 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    kept_lines.append(",".join(header) + "\n")
                
                for row in reader:
                    if not row: continue
                    try:
                        # Assume 1st column is 'timestep' or 'global_step'
                        ts = int(row[0]) 
                        if ts <= self.start_step:
                            kept_lines.append(",".join(row) + "\n")
                    except ValueError:
                        continue

        # 2. Read new file (skip header)
        new_content = []
        if os.path.exists(self.noise_path):
            with open(self.noise_path, 'r') as f:
                f.readline() # Skip header
                new_content = f.readlines()

        # 3. Write combined content
        with open(self.noise_path, 'w') as f:
            f.writelines(kept_lines)
            f.writelines(new_content)

        print(f"  -> Noise data merged. Cutoff at step <= {self.start_step}.")    

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_reward_params(param_list) -> Dict[str, Any]:
    """
    Parse a list of 'key=value' strings into a dict.

    Example:
        ["w_forward=1.0", "w_ctrl=0.5"] -> {"w_forward": 1.0, "w_ctrl": 0.5}

    Values are cast to float when possible, otherwise left as strings.
    """
    params: Dict[str, Any] = {}
    if param_list is None:
        return params

    for item in param_list:
        if "=" not in item:
            raise ValueError(f"Invalid reward-param '{item}'. Expected 'key=value'.")
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Try casting to float
        try:
            val_cast = float(val)
        except ValueError:
            val_cast = val
        params[key] = val_cast

    return params


def main():
    parser = argparse.ArgumentParser(description="Train SAC on Walker2d-v5 with custom rewards.")
    parser.add_argument(
        "--reward",
        type=str,
        required=True,
        help=f"Reward name. Available: {list(REWARD_FNS.keys())}",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Number of training timesteps.",
    )
    parser.add_argument(
        "--reward-param",
        action="append",
        help="Reward hyperparameter in the form key=value. "
             "Can be used multiple times. Example: --reward-param w_forward=1.0",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="If set, record a video of one evaluation episode after training.",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="./videos_cli",
        help="Folder where evaluation videos are saved (if --video is used).",
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to observations in the wrapper.",
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to actions in the wrapper.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Folder where TensorBoard logs will be saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this specific run (useful for grouping in plots).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="If set, resume training from the latest checkpoint if available.",
    )

    args = parser.parse_args()
    reward_params = parse_reward_params(args.reward_param)

    if args.run_name is None:
        args.run_name = f"walker2d-SAC-{args.reward}-ts{args.timesteps}-seed{args.seed}-obsnoise{args.obs_noise_std}-actnoise{args.action_noise_std}-rewardparams{'_'.join(f'{k}{v}' for k,v in reward_params.items())}"

    # Set global seed for reproducibility
    set_global_seed(args.seed)

    # Paths for logs
    monitor_file = f"{args.log_dir}/monitor-{args.run_name}.csv"
    noise_file = f"{args.log_dir}/noise-{args.run_name}.csv"
    checkpoint_dir = Path(f"{args.log_dir}/checkpoints")
    checkpoint_timesteps = 0
    checkpoint_path = ""

    if args.resume and checkpoint_dir.exists():
        # Determine the latest checkpoint step
        checkpoint_files = list(checkpoint_dir.glob(f"checkpoint-{args.run_name}_*.zip"))
        if checkpoint_files:
            latest_checkpoint = max(
                checkpoint_files,
                key=os.path.getctime
            )
            match = re.search(rf"checkpoint-{re.escape(args.run_name)}_(\d+)\.zip", latest_checkpoint.name)
            if match:
                checkpoint_timesteps = int(match.group(1))
                checkpoint_path = str(latest_checkpoint)
                print(f"[INFO] Resuming from checkpoint at step {checkpoint_timesteps}.")
            else:
                print("[WARNING] Could not parse step from checkpoint filename. Skipping log merge.")
        else:
            print("[WARNING] No checkpoint files found. Starting fresh training.")

    with LogMerger(
        monitor_path=monitor_file,
        noise_path=noise_file,
        enabled=args.resume and checkpoint_timesteps > 0,
        start_timesteps=checkpoint_timesteps,
        end_timesteps=args.timesteps
    ):
        # -----------------------
        # 1) Training environment
        # -----------------------
        # NOTE:
        # We do NOT enable video during training, to avoid overhead and
        # known issues with black videos in some MuJoCo + RecordVideo setups.
        train_env = make_walker2d_env(
            reward_name=args.reward,
            reward_params=reward_params,
            record_video=False,
            obs_noise_std=args.obs_noise_std,
            action_noise_std=args.action_noise_std,
            log_dir=args.log_dir,
            run_name=args.run_name,
        )
        train_env.reset(seed=args.seed)
        train_env.action_space.seed(args.seed)
        train_env = Monitor(train_env, filename=monitor_file)

        # Save a checkpoint every 100k steps
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=str(checkpoint_dir),
            name_prefix=f"checkpoint-{args.run_name}"
        )

        # Train SAC on the selected reward
        if args.resume and checkpoint_path:
            model = SAC.load(
                checkpoint_path,
                env=train_env,
                verbose=1,
                seed=args.seed,
                tensorboard_log=args.log_dir,
                device="auto",
            )
            print(f"[INFO] Resumed training from checkpoint.")
        else:
            model = SAC(
                "MlpPolicy",
                train_env,
                verbose=1,
                seed=args.seed,
                tensorboard_log=args.log_dir,
                device="auto",
            )

        model.learn(
            total_timesteps=args.timesteps - checkpoint_timesteps,
            tb_log_name=args.run_name,
            callback=checkpoint_callback,
            reset_num_timesteps=not args.resume,
        )

        model_path = f"{args.log_dir}/final_model_{args.run_name}.zip"
        model.save(model_path)
        print(f"[INFO] Model saved to {model_path}")

    # -----------------------
    # 2) Optional evaluation + video
    # -----------------------
    if args.video:
        eval_env = make_walker2d_env(
            reward_name=args.reward,
            reward_params=reward_params,
            record_video=True,
            video_folder=args.video_folder,
            video_prefix=f"video-{args.run_name}",
            obs_noise_std=args.obs_noise_std,
            action_noise_std=args.action_noise_std,
        )

        obs, info = eval_env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

        eval_env.close()
        print(f"[INFO] Evaluation video saved to: {args.video_folder}")


if __name__ == "__main__":
    main()
