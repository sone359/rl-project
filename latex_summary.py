"""
Script to generate a comprehensive LaTeX table from training logs.
Aggregates data across Algorithms, Rewards, VARIANTS, and Noise levels.

Usage:
    python generate_latex_table_v2.py --log-dir ./logs
"""

import os
import glob
import re
import argparse
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EVAL_WINDOW = 100 

# Mapping des chaînes de paramètres brutes vers des noms lisibles pour le rapport
# Basé sur votre pipeline run_campaign.sh
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
    
    # smooth_actions
    "lambda_smooth0.01": "Smooth 0.01",
    "lambda_smooth0.1": "Smooth 0.1",
    
    # dynamic_stability
    "lambda_state0.01": "Dyn 0.01",
    "lambda_state0.05": "Dyn 0.05",
    
    # anti_fall_progressive
    "h_crit1.1_w_h5.0": "Early Detect",
    "h_crit0.8_w_h5.0": "Late Detect",
    
    # robust_econ
    "v_weight1.0_energy_weight0.001": "Balanced",
    "v_weight1.0_energy_weight0.01": "Economical"
}

def parse_filename_info(filename):
    """
    Extracts Algo, Reward, Variant (Config), and Noise info from filename.
    """
    basename = os.path.basename(filename)
    
    # 1. Extract Algorithm
    algo_match = re.search(r"walker2d-([a-zA-Z0-9]+)-", basename)
    algo = algo_match.group(1) if algo_match else "Unknown"

    # 2. Extract Reward Name
    reward_match = re.search(f"walker2d-{algo}-(.+)-ts", basename)
    reward = reward_match.group(1) if reward_match else "Unknown"

    # 3. Extract Noise
    obs_match = re.search(r"obsnoise([0-9\.]+)", basename)
    act_match = re.search(r"actnoise([0-9\.]+)", basename)
    
    obs = float(obs_match.group(1)) if obs_match else 0.0
    act = float(act_match.group(1)) if act_match else 0.0

    if obs == 0.0 and act == 0.0:
        noise_label = "Clean"
    elif obs > 0.0 and act > 0.0:
        noise_label = "Combined"
    elif obs > 0.0:
        noise_label = "Obs Only"
    elif act > 0.0:
        noise_label = "Act Only"
    else:
        noise_label = "Other"

    # 4. Extract Parameters (Variant)
    # On cherche ce qui est après "rewardparams" et avant ".csv" ou ".monitor"
    # Le pattern est souvent : ...rewardparamsXXX.csv...
    param_match = re.search(r"rewardparams(.+?)(?:\.csv|$)", basename)
    raw_params = param_match.group(1) if param_match else ""
    
    # On nettoie le string (parfois des extensions trainent)
    if raw_params.endswith(".monitor"):
        raw_params = raw_params.replace(".monitor", "")

    # Mapping vers un nom lisible
    if raw_params in PARAM_MAPPINGS:
        variant = PARAM_MAPPINGS[raw_params]
    else:
        # Fallback : on affiche les params bruts tronqués si inconnu
        variant = raw_params[:15] + "..." if len(raw_params) > 15 else raw_params
        if not variant: variant = "Default"

    return algo, reward, variant, noise_label

def load_data(log_dir):
    files = glob.glob(os.path.join(log_dir, "monitor-*.csv"))
    records = []

    print(f"[INFO] Found {len(files)} log files. Processing...")

    for f in files:
        try:
            algo, reward, variant, noise = parse_filename_info(f)
            
            df = pd.read_csv(f, skiprows=1)
            
            if len(df) < EVAL_WINDOW:
                continue

            last_n = df.tail(EVAL_WINDOW)
            mean_reward = last_n['r'].mean()
            std_reward = last_n['r'].std()
            mean_len = last_n['l'].mean()

            records.append({
                "Algorithm": algo,
                "Reward": reward,
                "Variant": variant,  # On ajoute la variante ici
                "Noise": noise,
                "MeanR": mean_reward,
                "StdR": std_reward,
                "MeanLen": mean_len
            })
        except Exception as e:
            print(f"[WARN] Error processing {f}: {e}")

    return pd.DataFrame(records)

def format_cell(row):
    if pd.isna(row['MeanR']):
        return "-"
    
    r = int(round(row['MeanR']))
    std = int(round(row['StdR']))
    l = int(round(row['MeanLen']))
    
    len_str = f"{l}"
    if l >= 980:
        len_str = f"\\textbf{{{l}}}"
        
    return f"${r} \\pm {std}$ ({len_str})"

def generate_latex(df):
    if df.empty:
        print("[ERROR] No data to process.")
        return

    # Création d'une colonne combinée pour l'affichage : "Reward (Variant)"
    # Cela permet d'avoir une ligne unique par config dans le tableau
    df['RewardConfig'] = df['Reward'] + " (" + df['Variant'] + ")"
    
    df['Formatted'] = df.apply(format_cell, axis=1)
    
    # Pivot sur Algorithm + RewardConfig
    pivot_df = df.pivot_table(
        index=["Algorithm", "RewardConfig"], 
        columns="Noise", 
        values="Formatted", 
        aggfunc='first'
    )

    desired_order = ["Clean", "Obs Only", "Act Only", "Combined"]
    cols = [c for c in desired_order if c in pivot_df.columns]
    pivot_df = pivot_df[cols]

    latex_code = pivot_df.to_latex(
        escape=False, 
        multicolumn=True,
        multirow=True,
        caption="Comparison of Final Performance (Reward $\\pm$ Std) and Episode Length (in parentheses).",
        label="tab:results_summary"
    )

    latex_code = latex_code.replace("\\begin{table}", "\\begin{table*}[ht]")
    latex_code = latex_code.replace("\\end{table}", "\\end{table*}")
    latex_code = latex_code.replace("\\toprule", "\\toprule\n & & \\multicolumn{4}{c}{Noise Scenarios} \\\\\n \\cmidrule(lr){3-6}")
    
    # Échappement des underscores pour LaTeX (sauf s'ils sont déjà dans des maths)
    # On le fait sur le nom des rewards
    latex_code = latex_code.replace("_", "\\_")

    return latex_code

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    df = load_data(args.log_dir)
    
    if not df.empty:
        print("\n[INFO] Preview of aggregated data:")
        # On groupe par Variant aussi pour vérifier qu'on a bien les doublons
        print(df.groupby(["Algorithm", "Reward", "Variant", "Noise"])["MeanR"].mean().head())

        latex = generate_latex(df)
        
        output_file = "results_table_v2.tex"
        with open(output_file, "w") as f:
            f.write(latex)
        
        print(f"\n[SUCCESS] LaTeX table generated in '{output_file}'.")
    else:
        print("[ERROR] No data found.")

if __name__ == "__main__":
    main()