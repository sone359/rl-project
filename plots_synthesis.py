"""
Global Synthesis Plotting Script.

Generates 3 high-level summary figures for the report:
1. Global Stability: Boxplot of episode lengths (focusing on the hardest noise scenario).
2. Relative Robustness: Barplot of % performance drop (Clean vs Combined noise).
3. Learning Dynamics Grid: 2x2 grid showing learning curves per reward type (comparing Algos).

Usage:
    python plot_global_synthesis.py --log-dir ./logs
"""

import os
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LOG_DIR = "./final_logs/logs"
OUTPUT_DIR = "./final_logs/plots/synthesis"
WINDOW_SIZE = 1000  # Strong smoothing for the global grid to make it readable
EVAL_WINDOW = 100   # Last N episodes for barplots/boxplots

# Mapping raw params to readable names (Must match your Latex script)
PARAM_MAPPINGS = {
    # speed_energy
    "w_forward1.0_w_ctrl1.0_w_survive1.0": "Standard",
    "w_forward0.5_w_ctrl1.0_w_survive3.0": "Cautious",
    # target_speed
    "v_target1.5_alpha1.0_beta0.001": "Target 1.5",
    "v_target2.5_alpha1.0_beta0.001": "Target 2.5",
    # posture_stability
    "h_target1.25_w_h5.0_w_angle1.0": "Standard",
    "h_target1.25_w_h5.0_w_angle10.0": "Rigid",
    # dynamic_stability
    "lambda_state0.01": "Dyn 0.01",
    "lambda_state0.001": "Dyn 0.001", # Adjusted based on your bash script
     "lambda_state0.05": "Dyn 0.05",
}

# ==============================================================================
# DATA LOADING
# ==============================================================================

def parse_filename(filename):
    basename = os.path.basename(filename)
    
    # 1. Algorithm
    algo_match = re.search(r"walker2d-([a-zA-Z0-9]+)-", basename)
    algo = algo_match.group(1) if algo_match else "Unknown"

    # 2. Reward
    reward_match = re.search(f"walker2d-{algo}-(.+)-ts", basename)
    reward = reward_match.group(1) if reward_match else "Unknown"

    # 3. Noise
    obs_match = re.search(r"obsnoise([0-9\.]+)", basename)
    act_match = re.search(r"actnoise([0-9\.]+)", basename)
    obs = float(obs_match.group(1)) if obs_match else 0.0
    act = float(act_match.group(1)) if act_match else 0.0

    if obs == 0.0 and act == 0.0: noise = "Clean"
    elif obs > 0.0 and act > 0.0: noise = "Combined"
    elif obs > 0.0: noise = "Obs Only"
    elif act > 0.0: noise = "Act Only"
    else: noise = "Other"

    # 4. Variant
    param_match = re.search(r"rewardparams(.+?)(?:\.csv|$)", basename)
    raw_params = param_match.group(1) if param_match else ""
    if raw_params.endswith(".monitor"): raw_params = raw_params.replace(".monitor", "")
    
    variant = PARAM_MAPPINGS.get(raw_params, "Default")
    
    return algo, reward, variant, noise

def load_all_data(log_dir):
    files = glob.glob(os.path.join(log_dir, "monitor-*.csv"))
    data_frames = []
    
    print(f"[INFO] Loading {len(files)} log files...")
    
    for f in files:
        try:
            algo, reward, variant, noise = parse_filename(f)
            df = pd.read_csv(f, skiprows=1)
            
            if len(df) < 10: continue

            df['timesteps'] = df['l'].cumsum()
            df['Algorithm'] = algo
            df['Reward'] = reward
            df['Variant'] = variant
            df['Noise'] = noise
            df['Config'] = f"{algo} ({variant})" # For legend
            
            # Smoothing for learning curves
            df['smoothed_reward'] = df['r'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
            
            data_frames.append(df)
        except:
            pass
            
    return pd.concat(data_frames, ignore_index=True)

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_global_stability(df, output_dir):
    """
    Boxplot of Episode Lengths for the 'Combined' (Hardest) noise scenario.
    Compares Algorithms side-by-side for each Reward.
    """
    print("[PLOT] Generating Global Stability (Combined Noise)...")
    
    # Filter: Only Combined Noise, Last 100 episodes
    subset = df[df['Noise'] == 'Combined'].groupby(['Algorithm', 'Reward', 'Variant']).tail(EVAL_WINDOW)
    
    # Create label: "Reward\n(Variant)"
    subset['Label'] = subset['Reward'] + "\n(" + subset['Variant'] + ")"
    
    plt.figure(figsize=(14, 7))
    sns.boxplot(
        data=subset,
        x="Label",
        y="l",
        hue="Algorithm",
        palette="Set2"
    )
    
    plt.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label="Max Duration")
    plt.title("Global Stability Stress Test: Episode Duration under Combined Noise")
    plt.ylabel("Episode Length (Max 1000)")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.legend(title="Algorithm", loc="lower right")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "global_stability_combined.png"), dpi=200)
    plt.close()


def plot_relative_robustness(df, output_dir):
    """
    Barplot showing % Drop in Reward between 'Clean' and 'Combined' noise.
    """
    print("[PLOT] Generating Relative Robustness Drops...")
    
    # Calculate means
    means = df.groupby(['Algorithm', 'Reward', 'Variant', 'Noise'])['r'].mean().reset_index()
    
    # Pivot to have Clean and Combined in columns
    pivot = means.pivot(index=['Algorithm', 'Reward', 'Variant'], columns='Noise', values='r').reset_index()
    
    if 'Clean' not in pivot.columns or 'Combined' not in pivot.columns:
        print("[WARN] Missing Clean or Combined data for robustness plot.")
        return

    # Calculate Drop %
    # Formula: (Clean - Combined) / Clean * 100
    # Note: We handle negative rewards by taking absolute value in denominator or simplified diff
    pivot['DropPct'] = (pivot['Clean'] - pivot['Combined']) / pivot['Clean'].abs() * 100
    
    # Clean up labels
    pivot['Label'] = pivot['Reward'] + "\n(" + pivot['Variant'] + ")"

    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=pivot,
        x="Label",
        y="DropPct",
        hue="Algorithm",
        palette="viridis"
    )
    
    plt.axhline(y=0, color='black', linewidth=1)
    plt.title("Relative Robustness: Performance Drop from Clean to Combined Noise (%)")
    plt.ylabel("Performance Drop (%) - Lower is Better")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.legend(title="Algorithm")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "global_robustness_drop.png"), dpi=200)
    plt.close()


def plot_learning_grid(df, output_dir):
    """
    2x2 Grid of Learning Curves.
    One subplot per Reward type.
    Curves = Algorithms + Variants.
    Noise = Clean (to show ideal learning capability).
    """
    print("[PLOT] Generating Learning Curves Grid (Clean Environment)...")
    
    # Filter: Only Clean environment for clear learning curves
    subset = df[df['Noise'] == 'Clean']
    
    rewards = subset['Reward'].unique()
    # We assume 4 rewards for a 2x2 grid
    
    g = sns.FacetGrid(
        subset, 
        col="Reward", 
        col_wrap=2, 
        sharey=False,  # CRITICAL: Allow different Y scales per reward
        height=5, 
        aspect=1.5
    )
    
    g.map_dataframe(
        sns.lineplot, 
        x="timesteps", 
        y="smoothed_reward", 
        hue="Config", # Distinct line per Algo+Variant
        style="Algorithm", # Dashed vs Solid based on Algo
        linewidth=1.5
    )
    
    g.add_legend()
    g.set_titles("{col_name}")
    g.set_axis_labels("Timesteps", "Smoothed Reward")
    
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Learning Dynamics by Reward Type (Clean Environment)")
    
    plt.savefig(os.path.join(output_dir, "global_learning_grid.png"), dpi=200)
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Everything
    df = load_all_data(LOG_DIR)
    
    if df.empty:
        print("[ERROR] No data found.")
        return

    # 2. Generate Plots
    plot_global_stability(df, OUTPUT_DIR)
    plot_relative_robustness(df, OUTPUT_DIR)
    plot_learning_grid(df, OUTPUT_DIR)
    
    print(f"[SUCCESS] Global plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()