#!/bin/bash

# ==============================================================================
# 1. MACHINE AND EXPERIMENT CONFIGURATION
# ==============================================================================
# Adjust this section to match your hardware and experiment needs.

# Number of CPU cores to use simultaneously.
# Tip: Always leave 2-4 cores free for the OS and background tasks.
# Example: On a 14-core machine, setting this to 10 is ideal.
MAX_JOBS=4

# Duration of training for each model (in timesteps).
# 100,000 is good for quick tests. For final results, aim for 1,000,000+.
TIMESTEPS=800000

# Random Seed to ensure results are reproducible.
SEED=42

# Resume training from last checkpoint if available
# (Set to true to continue previous runs, false to start fresh)
RESUME=true

# Job Counting (for logging purposes)
CURRENT_JOB=0
TOTAL_JOBS=56  # (7 rewards * 2 configs * 4 noise levels)

# Directories and Files
LOG_DIR="./logs"          # Where TensorBoard logs and CSV files will be saved
PYTHON_SCRIPT="train.py"  # The name of your training script

# Automatically create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"


# ==============================================================================
# 2. UTILITY FUNCTIONS (SCRIPT ENGINE)
# ==============================================================================
# This section handles the complex logic (parallelization, file naming).
# You usually don't need to modify this part.

# Function that pauses the script if we are already using too many cores (MAX_JOBS).
function wait_for_slot {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
}

# Main function to launch a single training run.
# It constructs the filename, checks if it already exists, and launches Python.
function launch_exp {
    ((CURRENT_JOB++))

    local reward=$1       # Reward name (e.g., speed_energy)
    local obs_noise=$2    # Observation noise level (sensors)
    local act_noise=$3    # Action noise level (motors)
    shift 3
    local params=("$@")   # List of hyperparameters (e.g., w_forward=1.0)

    # --- Step A: Construct the unique run name ---
    # We transform the parameter list into a text string for the filename.
    local param_str=""
    local cli_args=""

    for p in "${params[@]}"; do
        # Split key and value to build CLI arguments
        cli_args="$cli_args --reward-param $p"
        
        # Build the filename suffix (keyval_keyval...)
        local key=${p%=*}
        local val=${p#*=}
        if [ -z "$param_str" ]; then
            param_str="${key}${val}"
        else
            param_str="${param_str}_${key}${val}"
        fi
    done

    # Final Run Name (Must match the logic inside train.py exactly!)
    local run_name="walker2d-SAC-${reward}-ts${TIMESTEPS}-seed${SEED}-obsnoise${obs_noise}-actnoise${act_noise}-rewardparams${param_str}"
    
    # Sentinel file to check if training is already done
    local target_file="$LOG_DIR/final_model_${run_name}.zip"

    # --- Step B: "Skip Logic" check ---
    # If the CSV file already exists, the training was completed. We skip it.
    if [ -f "$target_file" ]; then
        echo "[$CURRENT_JOB/$TOTAL_JOBS] [SKIP] File already exists: $target_file"
        return
    fi

    # --- Step C: Launch ---
    wait_for_slot # Wait for a CPU core to become free

    echo "[$CURRENT_JOB/$TOTAL_JOBS] [RUN] Launching: $run_name (Active jobs: $(( $(jobs -r | wc -l) + 1 ))/$MAX_JOBS)"
    
    # Launch Python in the background (& at the end).
    # We redirect text output to "void" (/dev/null) to keep the terminal clean.
    # Errors and important info are handled by TensorBoard logs.
    python $PYTHON_SCRIPT \
        --reward "$reward" \
        --timesteps $TIMESTEPS \
        --seed $SEED \
        --obs-noise-std "$obs_noise" \
        --action-noise-std "$act_noise" \
        --log-dir "$LOG_DIR" \
        --run-name "$run_name" \
        --video \
        $( [ "$RESUME" = true ] && echo "--resume" ) \
        $cli_args \
        > "$LOG_DIR/${run_name}.log" 2>&1 &
}


# ==============================================================================
# 3. CAMPAIGN DEFINITION
# ==============================================================================
# This is where you define what you want to test!

echo "Starting campaign on $MAX_JOBS cores..."

# --- Noise Scenarios Definition ---
# Each line represents a scenario: "obs_noise action_noise"
NOISE_CONFIGS=(
    "0.0 0.0"   # Baseline: No noise (Perfect environment)
    "0.01 0.0"   # Noisy Sensors: The robot sees poorly
    "0.0 0.1"   # Noisy Motors: The robot trembles
    "0.01 0.1"   # Hard Mode: Noisy sensors AND noisy motors
)

# --- EXPERIMENT LIST ---
To add an experiment, copy a block and change the parameters.

# 1. REWARD: speed_energy (Classic)
for noise in "${NOISE_CONFIGS[@]}"; do
    obs=${noise% *}
    act=${noise#* }
    
    # Config A: Standard
    launch_exp "speed_energy" $obs $act "w_forward=1.0" "w_ctrl=1.0" "w_survive=1.0"
    
    # Config B: Cautious (High survival bonus)
    launch_exp "speed_energy" $obs $act "w_forward=0.5" "w_ctrl=1.0" "w_survive=3.0"
done

# 2. REWARD: target_speed (Tracking specific speed)
for noise in "${NOISE_CONFIGS[@]}"; do
    obs=${noise% *}
    act=${noise#* }
    
    # Config A: Slow Walk (1.5 m/s)
    launch_exp "target_speed" $obs $act "v_target=1.5" "alpha=1.0" "beta=0.001"
    
    # Config B: Fast Walk (2.5 m/s)
    launch_exp "target_speed" $obs $act "v_target=2.5" "alpha=1.0" "beta=0.001"
done

# 3. REWARD: posture_stability (Elegant robot)
for noise in "${NOISE_CONFIGS[@]}"; do
    obs=${noise% *}
    act=${noise#* }
    
    # Config A: Standard
    launch_exp "posture_stability" $obs $act "h_target=1.25" "w_h=5.0" "w_angle=1.0"
    
    # Config B: Rigid Robot (Heavy angle penalty)
    launch_exp "posture_stability" $obs $act "h_target=1.25" "w_h=5.0" "w_angle=10.0"
done

# # 4. REWARD: smooth_actions (Fluid movements)
# for noise in "${NOISE_CONFIGS[@]}"; do
#     obs=${noise% *}
#     act=${noise#* }
    
#     # Config A: Light Smoothing
#     launch_exp "smooth_actions" $obs $act "lambda_smooth=0.01"
    
#     # Config B: Heavy Smoothing (Potentially sluggish but robust)
#     launch_exp "smooth_actions" $obs $act "lambda_smooth=0.1"
# done

# 5. REWARD: dynamic_stability
for noise in "${NOISE_CONFIGS[@]}"; do
    obs=${noise% *}
    act=${noise#* }
    
    launch_exp "dynamic_stability" $obs $act "lambda_state=0.01"
    launch_exp "dynamic_stability" $obs $act "lambda_state=0.001"
done

# # 6. REWARD: anti_fall_progressive (Fall prevention)
# for noise in "${NOISE_CONFIGS[@]}"; do
#     obs=${noise% *}
#     act=${noise#* }
    
#     # Config A: Early Detection (Punishes as soon as height drops slightly)
#     launch_exp "anti_fall_progressive" $obs $act "h_crit=1.1" "w_h=5.0"
    
#     # Config B: Late Detection (Punishes only near the ground)
#     launch_exp "anti_fall_progressive" $obs $act "h_crit=0.8" "w_h=5.0"
# done

# # 7. REWARD: robust_econ (Energy economy)
# for noise in "${NOISE_CONFIGS[@]}"; do
#     obs=${noise% *}
#     act=${noise#* }
    
#     # Config A: Balanced
#     launch_exp "robust_econ" $obs $act "v_weight=1.0" "energy_weight=0.001"
    
#     # Config B: Very Economical ("Lazy" robot)
#     launch_exp "robust_econ" $obs $act "v_weight=1.0" "energy_weight=0.01"
# done

# ==============================================================================
# END OF SCRIPT
# ==============================================================================

# Wait for the last 10 background processes to finish
wait

echo "-----------------------------------------------------"
echo "Finished! All scenarios have been executed."
echo "To view results:"
echo "  1. Open a terminal"
echo "  2. Run: tensorboard --logdir $LOG_DIR"
echo "  3. Open http://localhost:6006 in your browser"
echo "-----------------------------------------------------"