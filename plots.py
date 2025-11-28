"""
Specific Plotting Script for Walker2d Experiments.

Usage:
    python plot_experiment.py --reward speed_energy --reward-param w_forward=1.0 ...

Behavior:
    1. Reconstructs the experiment filename pattern based on arguments.
    2. Finds all noise variations (Clean, Obs, Act, Combined) for this specific configuration.
    3. Generates comparison plots (Learning Curve, Robustness, Stability).
    4. Saves them in a specific folder for this experiment.
"""

import argparse
import os
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LOG_DIR = "./logs"
PLOTS_DIR = "./plots"
WINDOW_SIZE = 50  # Smoothing window for learning curves

# ==============================================================================
# UTILS
# ==============================================================================

def parse_reward_params(param_list: List[str]) -> Dict[str, Any]:
    """Same parsing logic as train.py to ensure filename consistency."""
    params: Dict[str, Any] = {}
    if param_list is None:
        return params

    for item in param_list:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            val_cast = float(val)
        except ValueError:
            val_cast = val
        params[key] = val_cast
    return params

def get_noise_label(filename: str) -> str:
    """Extracts noise levels and returns a readable label."""
    obs_match = re.search(r"obsnoise([0-9\.]+)", filename)
    act_match = re.search(r"actnoise([0-9\.]+)", filename)

    obs = float(obs_match.group(1)) if obs_match else 0.0
    act = float(act_match.group(1)) if act_match else 0.0

    if obs == 0.0 and act == 0.0:
        return "Baseline (Clean)"
    elif obs > 0.0 and act > 0.0:
        return "Combined (Hard)"
    elif obs > 0.0:
        return "Sensors Noise"
    elif act > 0.0:
        return "Motors Noise"
    return "Unknown"

# ==============================================================================
# MAIN PLOTTING LOGIC
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot results for a specific experiment config.")
    
    # Arguments must match train.py to allow copy-pasting commands
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-param", action="append")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR)
    parser.add_argument("--output-dir", type=str, default=PLOTS_DIR)
    
    # Note: We do NOT ask for noise levels here, because we want to plot ALL noise levels 
    # found for this specific reward configuration.

    args = parser.parse_args()
    
    # 1. Reconstruct the run name pattern (excluding noise)
    reward_params = parse_reward_params(args.reward_param)
    param_str = '_'.join(f'{k}{v}' for k, v in reward_params.items())
    
    # The pattern matches the file naming convention in train.py
    # We use wildcards (*) for noise values
    # Format: monitor-walker2d-SAC-{reward}-ts{ts}-seed{seed}-obsnoise*-actnoise*-rewardparams{params}.csv.monitor.csv
    file_pattern = (
        f"monitor-walker2d-SAC-{args.reward}-"
        f"ts{args.timesteps}-seed{args.seed}-"
        f"obsnoise*-actnoise*-"
        f"rewardparams{param_str}.csv.monitor.csv"
    )
    
    search_path = os.path.join(args.log_dir, file_pattern)
    found_files = glob.glob(search_path)
    
    if not found_files:
        print(f"[ERROR] No files found matching pattern:\n  {search_path}")
        return

    print(f"[INFO] Found {len(found_files)} noise variations for configuration '{args.reward}'.")

    # 2. Load Data
    data_frames = []
    for file_path in found_files:
        try:
            # SB3 monitor files have 2 header lines
            df = pd.read_csv(file_path, skiprows=1)
            if 'r' not in df.columns or 'l' not in df.columns:
                continue
            
            # Add metadata
            df['noise_type'] = get_noise_label(os.path.basename(file_path))
            df['timesteps'] = df['l'].cumsum()
            df['smoothed_reward'] = df['r'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
            
            data_frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {file_path}: {e}")

    if not data_frames:
        print("[ERROR] No valid data extracted.")
        return

    full_df = pd.concat(data_frames, ignore_index=True)

    # 3. Prepare Output Directory
    # We create a subfolder named after the params to keep things organized
    exp_name = f"{args.reward}_{param_str}" if param_str else f"{args.reward}_default"
    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Learning Curve Comparison ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=full_df,
        x="timesteps",
        y="smoothed_reward",
        hue="noise_type",
        style="noise_type",
        linewidth=2
    )
    plt.title(f"Learning Curve: {args.reward}\nParams: {reward_params}")
    plt.xlabel("Timesteps")
    plt.ylabel("Smoothed Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curve.png"), dpi=150)
    plt.close()

    # --- Plot 2: Stability Boxplot (Last 20% of episodes) ---
    # Filter for the end of training
    end_training_df = full_df.groupby('noise_type').tail(100)
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=end_training_df,
        x="noise_type",
        y="l",
        palette="Set2"
    )
    plt.axhline(y=1000, color='r', linestyle='--', alpha=0.5, label="Max Steps")
    plt.title(f"Stability Analysis (Episode Length)\nParams: {reward_params}")
    plt.ylabel("Episode Length")
    plt.xlabel("Noise Scenario")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stability_boxplot.png"), dpi=150)
    plt.close()

    # --- Plot 3: Performance Barplot ---
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=end_training_df,
        x="noise_type",
        y="r",
        palette="viridis",
        errorbar="sd" # Show standard deviation
    )
    plt.title(f"Final Performance (Reward)\nParams: {reward_params}")
    plt.ylabel("Average Reward")
    plt.xlabel("Noise Scenario")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_barplot.png"), dpi=150)
    plt.close()

    print(f"[SUCCESS] Plots saved to: {save_dir}")

if __name__ == "__main__":
    main()