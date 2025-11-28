#!/bin/bash

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LOG_DIR="./logs"
OUTPUT_DIR="./plots"
PYTHON_SCRIPT="plots.py"
TIMESTEPS=1000000
SEED=42

# Ensure python dependencies
python -c "import pandas, seaborn, matplotlib" 2>/dev/null || { echo "Missing python libs"; exit 1; }

mkdir -p "$OUTPUT_DIR"

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

# Wrapper to call the python plotting script
# Usage: launch_plot <reward_name> <param_list...>
function launch_plot {
    local reward=$1
    shift 1
    local params=("$@")

    local cli_args=""
    for p in "${params[@]}"; do
        cli_args="$cli_args --reward-param $p"
    done

    echo "[PLOT] Generating graphs for: $reward $cli_args"

    python $PYTHON_SCRIPT \
        --reward "$reward" \
        --timesteps $TIMESTEPS \
        --seed $SEED \
        --log-dir "$LOG_DIR" \
        --output-dir "$OUTPUT_DIR" \
        $cli_args
}

# ==============================================================================
# PLOT CAMPAIGN DEFINITION
# ==============================================================================
# NOTE: Noise loops are removed because the python script aggregates
# all noise levels for a given config into one comparison plot.

echo "Starting plot generation..."

# 1. REWARD: speed_energy
# Config A: Standard
launch_plot "speed_energy" "w_forward=1.0" "w_ctrl=1.0" "w_survive=1.0"
# Config B: Cautious
launch_plot "speed_energy" "w_forward=0.5" "w_ctrl=1.0" "w_survive=3.0"


# 2. REWARD: target_speed
# Config A: Slow Walk
launch_plot "target_speed" "v_target=1.5" "alpha=1.0" "beta=0.001"
# Config B: Fast Walk
launch_plot "target_speed" "v_target=2.5" "alpha=1.0" "beta=0.001"


# 3. REWARD: posture_stability
# Config A: Standard
launch_plot "posture_stability" "h_target=1.25" "w_h=5.0" "w_angle=1.0"
# Config B: Rigid Robot
launch_plot "posture_stability" "h_target=1.25" "w_h=5.0" "w_angle=10.0"


# 4. REWARD: smooth_actions
# Config A: Light Smoothing
launch_plot "smooth_actions" "lambda_smooth=0.01"
# Config B: Heavy Smoothing
launch_plot "smooth_actions" "lambda_smooth=0.1"


# 5. REWARD: dynamic_stability
launch_plot "dynamic_stability" "lambda_state=0.01"
launch_plot "dynamic_stability" "lambda_state=0.05"


# 6. REWARD: anti_fall_progressive
# Config A: Early Detection
launch_plot "anti_fall_progressive" "h_crit=1.1" "w_h=5.0"
# Config B: Late Detection
launch_plot "anti_fall_progressive" "h_crit=0.8" "w_h=5.0"


# 7. REWARD: robust_econ
# Config A: Balanced
launch_plot "robust_econ" "v_weight=1.0" "energy_weight=0.001"
# Config B: Very Economical
launch_plot "robust_econ" "v_weight=1.0" "energy_weight=0.01"


echo "-----------------------------------------------------"
echo "Plots generated in: $OUTPUT_DIR"
echo "Each subfolder corresponds to one hyperparameter configuration."
echo "-----------------------------------------------------"