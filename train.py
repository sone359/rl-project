"""
Command-line training script for Walker2d-v5 with custom reward functions.
Now supports VecNormalize (enabled by default).
"""

import argparse
from typing import Dict, Any, Optional, Tuple

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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


# ==============================================================================
# Logging utilities
# ==============================================================================

class LogMerger:
    """
    When resuming, SB3 Monitor + your noise CSV will be re-created from scratch.
    This helper archives previous logs to *.temp, then merges them back after training
    so you keep ONE continuous CSV.

    NOTE: enabled=False => this context manager is a no-op.
    """
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
                    print(f"[LogMerger] Archived: {os.path.basename(path)} -> {os.path.basename(temp_path)}")
                except OSError as e:
                    print(f"[LogMerger] Error moving file {path}: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        print("[LogMerger] Merging and cleaning up logs...")

        # Merge Monitor
        if self.monitor_path in self.temp_paths:
            self._merge_monitor_csv()
            if os.path.exists(self.temp_paths[self.monitor_path]):
                os.remove(self.temp_paths[self.monitor_path])

        # Merge Noise
        if self.noise_path in self.temp_paths:
            self._merge_noise_csv()
            if os.path.exists(self.temp_paths[self.noise_path]):
                os.remove(self.temp_paths[self.noise_path])

    def _merge_monitor_csv(self):
        """
        Merge SB3 Monitor logs.
        Format:
            Line 1: Metadata (JSON)
            Line 2: Header (r,l,t)
            Line 3+: Data

        Logic:
            We estimate "global timestep" by cumulative sum of episode lengths `l`.
            We keep old episodes until cumulative <= start_timesteps.
        """
        temp_path = self.temp_paths.get(self.monitor_path)
        if not temp_path or not os.path.exists(temp_path):
            return

        # If new file doesn't exist (crash), restore old and exit
        if not os.path.exists(self.monitor_path):
            shutil.move(temp_path, self.monitor_path)
            return

        kept_lines = []
        cumulative_steps = 0

        # --- Read old file (temp) ---
        with open(temp_path, "r") as f_old:
            meta_line = f_old.readline()
            header_line = f_old.readline()
            if meta_line:
                kept_lines.append(meta_line)
            if header_line:
                kept_lines.append(header_line)

            reader_old = csv.DictReader(f_old, fieldnames=["r", "l", "t"])
            for row in reader_old:
                try:
                    l = int(row["l"])
                except (ValueError, TypeError, KeyError):
                    continue
                if cumulative_steps + l <= self.start_timesteps and (self.end_timesteps is None or cumulative_steps < self.end_timesteps):
                    kept_lines.append(f"{row['r']},{row['l']},{row['t']}\n")
                    cumulative_steps += l
                else:
                    break

        # --- Read new file (skip 2 first lines) ---
        new_lines = []
        with open(self.monitor_path, "r") as f_new:
            # skip metadata + header
            f_new.readline()
            f_new.readline()
            reader_new = csv.DictReader(f_new, fieldnames=["r", "l", "t"])
            for row in reader_new:
                try:
                    l = int(row["l"])
                except (ValueError, TypeError, KeyError):
                    continue
                if self.end_timesteps is None or cumulative_steps < self.end_timesteps:
                    new_lines.append(f"{row['r']},{row['l']},{row['t']}\n")
                    cumulative_steps += l
                else:
                    break

        # --- Write merged ---
        with open(self.monitor_path, "w") as f_out:
            f_out.writelines(kept_lines)
            f_out.writelines(new_lines)

        print(f"[LogMerger] Monitor merged. Preserved {max(0, len(kept_lines)-2)} old episodes.")

    def _merge_noise_csv(self):
        """
        Merge custom Noise logs.
        Format:
            Header (timestep, obs_noise_0..., action_noise_0...)
            Data rows

        Logic:
            Keep rows from old file with timestep <= start_timesteps, then append new file rows.
        """
        temp_path = self.temp_paths.get(self.noise_path)
        if not temp_path or not os.path.exists(temp_path):
            return

        # If new file doesn't exist (crash), restore old and exit
        if not os.path.exists(self.noise_path):
            shutil.move(temp_path, self.noise_path)
            return

        kept_lines = []

        # --- Old (temp) ---
        with open(temp_path, "r") as f_old:
            reader = csv.reader(f_old)
            header = next(reader, None)
            if header:
                kept_lines.append(",".join(header) + "\n")

            for row in reader:
                if not row:
                    continue
                try:
                    ts = int(row[0])
                except ValueError:
                    continue
                if ts <= self.start_timesteps:
                    kept_lines.append(",".join(row) + "\n")
                else:
                    break

        # --- New ---
        with open(self.noise_path, "r") as f_new:
            # skip header
            f_new.readline()
            new_content = f_new.readlines()

        # --- Write merged ---
        with open(self.noise_path, "w") as f_out:
            f_out.writelines(kept_lines)
            f_out.writelines(new_content)

        print(f"[LogMerger] Noise merged. Cutoff at timestep <= {self.start_timesteps}.")


# ==============================================================================
# VecNormalize helpers
# ==============================================================================

class VecNormalizeCheckpointCallback(BaseCallback):
    """
    Save VecNormalize statistics periodically alongside model checkpoints.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vecnormalize", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.save_freq > 0 and (self.n_calls % self.save_freq == 0):
            env = self.model.get_env()
            if isinstance(env, VecNormalize):
                file_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}.pkl")
                env.save(file_path)
                if self.verbose:
                    print(f"[VecNormalize] Saved stats to: {file_path}")
        return True


def find_latest_vecnormalize(checkpoint_dir: Path, run_name: str) -> Tuple[Optional[str], int]:
    """
    Return (path, timestep) for latest vecnormalize-{run_name}_STEP.pkl, else (None, 0).
    """
    files = list(checkpoint_dir.glob(f"vecnormalize-{run_name}_*.pkl"))
    if not files:
        return None, 0

    best_path = None
    best_step = -1
    pattern = re.compile(rf"vecnormalize-{re.escape(run_name)}_(\d+)\.pkl$")
    for f in files:
        m = pattern.search(f.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = str(f)
    return best_path, max(best_step, 0)


# ==============================================================================
# Misc helpers
# ==============================================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_reward_params(param_list) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if param_list is None:
        return params
    for item in param_list:
        if "=" not in item:
            raise ValueError(f"Invalid reward-param '{item}'. Expected 'key=value'.")
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            val_cast = float(val)
        except ValueError:
            val_cast = val
        params[key] = val_cast
    return params


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train an RL agent on Walker2d-v5 with custom rewards.")
    parser.add_argument("--algorithm", type=str, default="SAC", help="RL algorithm to use. Available: SAC, PPO")
    parser.add_argument("--reward", type=str, required=True, help=f"Reward name. Available: {list(REWARD_FNS.keys())}")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Number of training timesteps.")
    parser.add_argument(
        "--reward-param",
        action="append",
        help="Reward hyperparameter in the form key=value. Can be used multiple times."
    )
    parser.add_argument("--video", action="store_true", help="Record one evaluation episode video after training.")
    parser.add_argument("--video-folder", type=str, default="./videos_cli", help="Folder where eval videos are saved.")
    parser.add_argument("--obs-noise-std", type=float, default=0.0, help="Std of Gaussian noise added to observations.")
    parser.add_argument("--action-noise-std", type=float, default=0.0, help="Std of Gaussian noise added to actions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Folder where logs will be saved.")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from latest checkpoint if available.")

    # VecNormalize options (enabled by default)
    parser.add_argument(
        "--no-vecnormalize",
        action="store_true",
        help="Disable VecNormalize (by default VecNormalize is ENABLED)."
    )
    parser.add_argument(
        "--norm-reward",
        action="store_true",
        default=False,
        help="Normalize rewards (usually keep False when you want comparable reward scales)."
    )
    parser.add_argument(
        "--clip-obs",
        type=float,
        default=10.0,
        help="Clipping for normalized observations in VecNormalize."
    )

    args = parser.parse_args()
    reward_params = parse_reward_params(args.reward_param)

    if args.run_name is None:
        args.run_name = (
            f"walker2d-{args.algorithm}-{args.reward}-ts{args.timesteps}-seed{args.seed}"
            f"-obsnoise{args.obs_noise_std}-actnoise{args.action_noise_std}"
            f"-rewardparams{'_'.join(f'{k}{v}' for k, v in reward_params.items())}"
        )

    set_global_seed(args.seed)

    # Paths
    os.makedirs(args.log_dir, exist_ok=True)
    monitor_file = f"{args.log_dir}/monitor-{args.run_name}.csv"
    noise_file = f"{args.log_dir}/noise-{args.run_name}.csv"
    checkpoint_dir = Path(f"{args.log_dir}/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vecnorm_final_path = f"{args.log_dir}/vecnormalize_{args.run_name}.pkl"
    use_vecnorm = not args.no_vecnormalize

    checkpoint_timesteps = 0
    checkpoint_path = ""

    # Find model checkpoint
    if args.resume and checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob(f"checkpoint-{args.run_name}_*.zip"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            match = re.search(rf"checkpoint-{re.escape(args.run_name)}_(\d+)\.zip", latest_checkpoint.name)
            if match:
                checkpoint_timesteps = int(match.group(1))
                checkpoint_path = str(latest_checkpoint)
                print(f"[INFO] Resuming from model checkpoint at step {checkpoint_timesteps}: {checkpoint_path}")
            else:
                print("[WARNING] Could not parse step from checkpoint filename. Starting fresh training.")
        else:
            print("[WARNING] No checkpoint files found. Starting fresh training.")

    with LogMerger(
        monitor_path=monitor_file,
        noise_path=noise_file,
        enabled=args.resume and checkpoint_timesteps > 0,
        start_timesteps=checkpoint_timesteps,
        end_timesteps=args.timesteps,
    ):
        # -----------------------
        # 1) Training environment
        # -----------------------

        def make_train_env():
            # NOTE: no video during training
            env = make_walker2d_env(
                reward_name=args.reward,
                reward_params=reward_params,
                record_video=False,
                obs_noise_std=args.obs_noise_std,
                action_noise_std=args.action_noise_std,
                log_dir=args.log_dir,
                run_name=args.run_name,
            )
            # Monitor inside the thunk (because DummyVecEnv expects gym.Env)
            env = Monitor(env, filename=monitor_file)

            # Seed once here (VecEnv reset() has no seed argument)
            # This may cause one extra reset internally, but keeps runs reproducible.
            env.reset(seed=args.seed)
            env.action_space.seed(args.seed)
            return env

        base_train_env = DummyVecEnv([make_train_env])

        # Apply / load VecNormalize
        if use_vecnorm:
            vec_ckpt_path = None
            if args.resume and checkpoint_timesteps > 0:
                vec_ckpt_path, vec_ckpt_step = find_latest_vecnormalize(checkpoint_dir, args.run_name)
                if vec_ckpt_path:
                    print(f"[INFO] Resuming VecNormalize stats from step {vec_ckpt_step}: {vec_ckpt_path}")

            if vec_ckpt_path and os.path.exists(vec_ckpt_path):
                train_env = VecNormalize.load(vec_ckpt_path, base_train_env)
            else:
                if args.resume and checkpoint_timesteps > 0:
                    print("[WARNING] No VecNormalize checkpoint found. Starting new normalization stats (not a perfect resume).")
                train_env = VecNormalize(
                    base_train_env,
                    norm_obs=True,
                    norm_reward=args.norm_reward,
                    clip_obs=float(args.clip_obs),
                )

            train_env.training = True
            train_env.norm_reward = bool(args.norm_reward)
        else:
            train_env = base_train_env

        # -----------------------
        # 2) Callbacks (model + vecnorm checkpoints)
        # -----------------------
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=str(checkpoint_dir),
            name_prefix=f"checkpoint-{args.run_name}",
        )

        callbacks = [checkpoint_callback]
        if use_vecnorm:
            callbacks.append(
                VecNormalizeCheckpointCallback(
                    save_freq=100000,
                    save_path=str(checkpoint_dir),
                    name_prefix=f"vecnormalize-{args.run_name}",
                    verbose=0,
                )
            )
        callback = CallbackList(callbacks)

        # -----------------------
        # 3) Model
        # -----------------------
        if args.resume and checkpoint_path:
            if args.algorithm == "SAC":
                model = SAC.load(
                    checkpoint_path,
                    env=train_env,
                    verbose=1,
                    seed=args.seed,
                    tensorboard_log=args.log_dir,
                    device="auto",
                )
                print("[DEBUG] SAC model loaded for resuming.")
            elif args.algorithm == "PPO":
                model = PPO.load(
                    checkpoint_path,
                    env=train_env,
                    verbose=1,
                    seed=args.seed,
                    tensorboard_log=args.log_dir,
                    device="auto",
                )
                print("[DEBUG] PPO model loaded for resuming.")
            else:
                raise ValueError(f"Unsupported algorithm '{args.algorithm}'. Available: SAC, PPO")
            print("[INFO] Resumed training from checkpoint.")
        else:
            if args.algorithm == "SAC":
                model = SAC(
                    "MlpPolicy",
                    train_env,
                    verbose=1,
                    seed=args.seed,
                    tensorboard_log=args.log_dir,
                    device="auto",
                )
                print("[DEBUG] SAC model created.")
            elif args.algorithm == "PPO":
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    verbose=1,
                    seed=args.seed,
                    tensorboard_log=args.log_dir,
                    device="auto",
                )
                print("[DEBUG] PPO model created.")
            else:
                raise ValueError(f"Unsupported algorithm '{args.algorithm}'. Available: SAC, PPO")

        # -----------------------
        # 4) Train
        # -----------------------
        remaining_steps = max(0, args.timesteps - checkpoint_timesteps)
        model.learn(
            total_timesteps=remaining_steps,
            tb_log_name=args.run_name,
            callback=callback,
            reset_num_timesteps=not args.resume,
        )

        # -----------------------
        # 5) Save final artifacts
        # -----------------------
        model_path = f"{args.log_dir}/final_model_{args.run_name}.zip"
        model.save(model_path)
        print(f"[INFO] Model saved to {model_path}")

        if use_vecnorm and isinstance(train_env, VecNormalize):
            train_env.save(vecnorm_final_path)
            print(f"[INFO] VecNormalize stats saved to {vecnorm_final_path}")

        train_env.close()

    # -----------------------
    # 6) Optional evaluation + video
    # -----------------------
    if args.video:
        # Create a 1-env VecEnv that includes RecordVideo inside the sub-env
        def make_eval_env():
            env = make_walker2d_env(
                reward_name=args.reward,
                reward_params=reward_params,
                record_video=True,
                video_folder=args.video_folder,
                video_prefix=f"video-{args.run_name}",
                obs_noise_std=args.obs_noise_std,
                action_noise_std=args.action_noise_std,
                log_dir=args.log_dir,
                run_name=args.run_name,
            )
            return env

        base_eval_env = DummyVecEnv([make_eval_env])

        if use_vecnorm:
            if not os.path.exists(vecnorm_final_path):
                print("[WARNING] VecNormalize final stats not found. Video will run WITHOUT normalization (policy may behave poorly).")
                eval_env = base_eval_env
            else:
                eval_env = VecNormalize.load(vecnorm_final_path, base_eval_env)
                eval_env.training = False
                eval_env.norm_reward = False
        else:
            eval_env = base_eval_env

        obs = eval_env.reset()
        done = np.array([False])

        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, infos = eval_env.step(action)

        eval_env.close()
        print(f"[INFO] Evaluation video saved to: {args.video_folder}")


if __name__ == "__main__":
    main()
