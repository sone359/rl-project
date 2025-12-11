# Walker2d Reward Shaping Project

This project investigates **reward shaping** and **robustness to perturbations** on the `Walker2d-v5` environment using multiple custom reward functions and a generic training script.

We use:
- [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium) with `Walker2d-v5` (MuJoCo),
- [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3) (SAC algorithm),
- A **generic reward interface** so you can switch reward functions from the command line.

Later, perturbations (noise, random pushes, domain randomization, etc.) can be added at clearly marked places in the environment wrapper.

---

## 1. Project Structure

```text
rl-project/
├── reward_function.py      # 7 custom reward functions + registry REWARD_FNS
├── walker2d_env.py         # Generic Walker2d wrapper with noise injection
├── train.py                # CLI training script (SAC/PPO) with checkpointing
├── train_pipeline.sh       # Batch training across configs and noise levels
├── plots.py                # Per-experiment visualization script
├── plots_pipeline.sh       # Batch plot generation
├── plots_synthesis.py      # Global comparison figures
├── latex_summary.py        # LaTeX table generator for reports
├── manual_merge.py         # Utility for merging interrupted logs
├── README.md               # This file
├── logs/                   # Training outputs (created at runtime)
│   ├── monitor-*.csv       # Episode rewards and lengths
│   ├── noise-*.csv         # Per-step noise values
│   ├── checkpoints/        # Model checkpoints
│   └── tb_logs/            # TensorBoard logs
├── plots/                  # Generated figures (created at runtime)
├── videos_cli/             # Evaluation videos (created at runtime)
└── final_logs/             # Archived final results
```

### 1.1. `reward_function.py`

Contains:

* Type aliases (`Obs`, `Action`, `RewardState`, …),

* A registry:

  ```python
  REWARD_FNS: Dict[str, RewardFn] = {}
  ```

* A decorator:

  ```python
  def register_reward(name: str):
      ...
  ```

* Seven reward functions, each with the **same interface**:

  ```python
  def reward_xxx(
      obs: np.ndarray,
      action: np.ndarray,
      base_reward: float,
      info: dict,
      state: dict,
      params: dict,
  ) -> tuple[float, dict]:
      ...
      return new_reward, new_state
  ```

Each function is registered under a string key:

```python
@register_reward("speed_energy")
def reward_speed_energy(...):
    ...
```

So all rewards are selectable by name through `REWARD_FNS["speed_energy"]`, `REWARD_FNS["target_speed"]`, etc.

---

### 1.2. `walker2d_env.py`

Contains:

* `GenericRewardWrapper` (a single **generic** wrapper for all rewards),
* `make_walker2d_env(...)` to build a Walker2d environment with a selected reward and optional video recording.

Key points:

* The wrapper calls the **base `Walker2d-v5` env** to get:

  ```python
  obs, base_reward, terminated, truncated, info = env.step(action)
  ```

* Then it calls the selected reward function:

  ```python
  new_reward, new_state = self.reward_fn(
      obs=obs_for_reward,
      action=np.array(action_to_env),
      base_reward=float(base_reward),
      info=info,
      state=self.reward_state,
      params=self.reward_params,
  )
  ```

* `self.reward_state` is a small dict used by some rewards to keep history (previous actions, previous observations, etc.).

* There are `# TODO: perturbations` comments where you can later:

  * add noise to actions/observations,
  * apply random pushes,
  * do domain randomization at reset, etc.

---

### 1.3. `train.py`

Command-line script to:

1. Parse arguments (`--reward`, `--timesteps`, `--reward-param key=value`, `--video`, …),
2. Build a **training environment** via `make_walker2d_env(reward_name=...)`,
3. Train a SAC agent,
4. Optionally, create a **separate evaluation env with video** and run one episode to record a `.mp4`.

Example usage:

```bash
python train.py --reward speed_energy --timesteps 100000 --video
```

---

## 2. Installation

### 2.1. Dependencies

You need (typical versions, adjust as needed):

* Python 3.10+ (recommended)
* NumPy 2.x
* gymnasium with MuJoCo support:

  ```bash
  pip install "gymnasium[mujoco]"
  ```
* stable-baselines3:

  ```bash
  pip install stable-baselines3
  ```
* (Optional) other tools: `matplotlib`, `jupyter`, etc.

Also make sure you have MuJoCo correctly installed/configured according to Gymnasium’s documentation.

### 2.2. Setup

Clone or copy the project/code, then from the project root:

```bash
pip install -r requirements.txt   # if you create one
# or install manually:
pip install "gymnasium[mujoco]" stable-baselines3 numpy
```

Check that the reward registry works:

```bash
python -c "from reward_function import REWARD_FNS; print(REWARD_FNS.keys())"
```

You should see something like:

```text
dict_keys([
  'speed_energy',
  'target_speed',
  'posture_stability',
  'smooth_actions',
  'dynamic_stability',
  'anti_fall_progressive',
  'robust_econ'
])
```

---

## 3. How the Reward System Works

All reward functions:

* Share the same signature,
* Are registered in `REWARD_FNS` with a string key,
* May use `state` and `params` differently.

### 3.1. Generic Signature

```python
new_reward, new_state = reward_fn(
    obs: np.ndarray,
    action: np.ndarray,
    base_reward: float,
    info: dict,
    state: dict,
    params: dict,
)
```

* `obs`      – observation from `env.step`,
* `action`   – action applied at this step,
* `base_reward` – original Walker2d reward (not always used),
* `info`     – info dict from env (may contain forward_reward, reward_survive, etc.),
* `state`    – internal state (e.g. previous action, previous observation),
* `params`   – hyperparameters for the reward (weights, targets, etc.).

### 3.2. Selecting a Reward from CLI

`train.py` reads `--reward` and checks that this name is in `REWARD_FNS`.

Example:

```bash
python train.py --reward target_speed ...
```

→ uses `reward_function.reward_target_speed` registered as `"target_speed"`.

You can pass hyperparameters using repeated `--reward-param key=value` options.

---

## 4. Running Training – General Command

General pattern:

```bash
python train.py \
    --reward <reward_name> \
    --reward-param key1=value1 \
    --reward-param key2=value2 \
    --timesteps 100000 \
    --video
```

* `--reward`       – name registered in `REWARD_FNS`,
* `--reward-param` – optional, can be used multiple times to set hyperparameters,
* `--timesteps`    – number of training steps for SAC,
* `--video`        – if set, record a video after training in `./videos_cli/`.

---

## 5. Reward Functions and Example Commands

Below, each reward is briefly described with an example command.

### 5.1. Reward 1 – `speed_energy`

**Goal:** Re-weight the standard Walker2d components:

* `reward_forward` (forward speed),
* `reward_ctrl` (control cost, already negative),
* `reward_survive` (healthy/survival bonus).

Formula:

[
r_t = w_{\text{forward}} \cdot \text{reward_forward}
+ w_{\text{ctrl}}    \cdot \text{reward_ctrl}
+ w_{\text{survive}} \cdot \text{reward_survive}
]

**Example:**

```bash
python train.py --reward speed_energy \
    --reward-param w_forward=1.0 \
    --reward-param w_ctrl=1.0 \
    --reward-param w_survive=1.0 \
    --timesteps 100000 \
    --video
```

---

### 5.2. Reward 2 – `target_speed`

**Goal:** Track a **target walking speed** (v^*) instead of “the faster the better”.

Approximate form:

[
r_t = - \alpha , |v_x - v^*|
- \beta , |a_t|^2
+ w_{\text{survive}} \cdot \text{reward_survive}
]

where (v_x) is the torso horizontal velocity.

**Example:**

```bash
python train.py --reward target_speed \
    --reward-param v_target=1.5 \
    --reward-param alpha=1.0 \
    --reward-param beta=0.001 \
    --reward-param w_survive=1.0 \
    --timesteps 100000 \
    --video
```

You can try different `v_target` (e.g. 1.0 vs 2.0 m/s) to see different gaits.

---

### 5.3. Reward 3 – `posture_stability`

**Goal:** Encourage a **stable upright posture** (high enough torso, not too tilted) in addition to moving forward.

Uses:

* `reward_forward`, `reward_ctrl`, `reward_survive` (base terms),
* `obs[0]` = torso height (h_t),
* `obs[1]` = torso angle (\theta_t).

Typical shaping term:

[
r_t = \text{base_terms}
- w_h , \max(0, h_{\text{target}} - h_t)^2
- w_{\theta} \theta_t^2
]

**Example:**

```bash
python train.py --reward posture_stability \
    --reward-param h_target=1.25 \
    --reward-param w_forward=1.0 \
    --reward-param w_ctrl=1.0 \
    --reward-param w_survive=1.0 \
    --reward-param w_h=5.0 \
    --reward-param w_angle=1.0 \
    --timesteps 100000 \
    --video
```

---

### 5.4. Reward 4 – `smooth_actions`

**Goal:** Penalize **abrupt changes in actions** so the policy doesn’t react nervously to noise.

Uses:

* Previous action stored in `state["prev_action"]`.

Formula:

[
r_t = \text{base_reward}*t
- \lambda*{\text{smooth}} , |a_t - a_{t-1}|^2
]

**Example:**

```bash
python train.py --reward smooth_actions \
    --reward-param lambda_smooth=0.01 \
    --timesteps 100000 \
    --video
```

You can compare different `lambda_smooth` values (e.g. 0.001, 0.01, 0.05).

---

### 5.5. Reward 5 – `dynamic_stability`

**Goal:** Penalize **jerky changes in the state** (approximate acceleration of the observation) to get smoother dynamics.

Uses:

* Previous observations `state["prev_obs"]`, `state["prev_prev_obs"]`.

Approximate second-order difference:

[
\Delta^2 s_t \approx s_t - 2 s_{t-1} + s_{t-2}
]

Reward:

[
r_t = \text{base_reward}*t
- \lambda*{\text{state}} , |\Delta^2 s_t|^2
]

**Example:**

```bash
python train.py --reward dynamic_stability \
    --reward-param lambda_state=0.01 \
    --timesteps 100000 \
    --video
```

---

### 5.6. Reward 6 – `anti_fall_progressive`

**Goal:** Provide a **progressive anti-fall shaping**: penalize states that are close to falling (torso too low or too tilted), not only the actual fall.

Uses:

* Torso height `h_t = obs[0]`,
* Torso angle `θ_t = obs[1]`,
* Thresholds `h_crit`, `angle_crit`.

Reward:

[
r_t = \text{base_reward}*t
- w_h , \max(0, h*{\text{crit}} - h_t)^2
- w_{\text{angle}} , \max(0, |\theta_t| - \theta_{\text{crit}})^2
]

**Example:**

```bash
python train.py --reward anti_fall_progressive \
    --reward-param h_crit=0.9 \
    --reward-param angle_crit=0.5 \
    --reward-param w_h=5.0 \
    --reward-param w_angle=1.0 \
    --timesteps 100000 \
    --video
```

---

### 5.7. Reward 7 – `robust_econ` (robust & economical walking)

**Goal:** Explicit **trade-off between speed and energy**, optionally with survival.

Uses:

* (v_x =) torso horizontal velocity `obs[8]`,
* (|a_t|^2 =) energy of the action,
* `reward_survive`.

Reward:

[
r_t = v_{\text{weight}} , v_x
- \text{energy_weight} , |a_t|^2
+ w_{\text{survive}} , \text{reward_survive}
]

**Example:**

```bash
python train.py --reward robust_econ \
    --reward-param v_weight=1.0 \
    --reward-param energy_weight=0.001 \
    --reward-param w_survive=0.0 \
    --timesteps 100000 \
    --video
```

You can explore the effect of smaller/larger `energy_weight` (more or less energy penalty).

---

## 6. Perturbations (Noise Injection)

The environment wrapper `walker2d_env.py` supports **noise injection** to test robustness:

### 6.1. Observation Noise (Sensor Noise)

Gaussian noise added to observations after `env.step()`:

```bash
python train.py --reward speed_energy --obs-noise-std 0.01 --timesteps 100000
```

### 6.2. Action Noise (Actuator Noise)

Gaussian noise added to actions before `env.step()`:

```bash
python train.py --reward speed_energy --action-noise-std 0.1 --timesteps 100000
```

### 6.3. Combined Noise

Test under realistic conditions with both noises:

```bash
python train.py --reward speed_energy \
    --obs-noise-std 0.01 \
    --action-noise-std 0.1 \
    --timesteps 100000
```

### 6.4. Noise Logging

Noise values are logged to `noise-{run_name}.csv` for analysis:
- Columns: `timestep`, `obs_noise_0`, ..., `obs_noise_N`, `action_noise_0`, ..., `action_noise_M`

### 6.5. Future Extensions

The code contains `# TODO: perturbations` markers for additional perturbation types:
- **Random pushes** on the torso
- **Domain randomization** (mass, friction, etc.)
- **Reset-time perturbations**

---

## 7. Batch Training Pipeline

For running systematic experiments across multiple reward configurations and noise levels, use the bash script `train_pipeline.sh`.

### 8.1. Configuration

Edit the configuration section at the top of the script:

```bash
MAX_JOBS=6          # Number of parallel jobs (leave 2-4 cores free)
TIMESTEPS=800000    # Training duration per model
SEED=42             # Random seed for reproducibility
RESUME=true         # Resume from checkpoints if available
ALGORITHM="PPO"     # RL algorithm (PPO, SAC, etc.)
LOG_DIR="./logs"    # Output directory for logs
```

### 8.2. Noise Scenarios

The pipeline tests each reward configuration under four noise scenarios:

| Scenario | Obs Noise | Act Noise | Description |
|----------|-----------|-----------|-------------|
| Clean    | 0.0       | 0.0       | Baseline (perfect environment) |
| Obs Only | 0.01      | 0.0       | Noisy sensors |
| Act Only | 0.0       | 0.1       | Noisy motors |
| Combined | 0.01      | 0.1       | Both noises (stress test) |

### 8.3. Running the Campaign

```bash
chmod +x train_pipeline.sh
./train_pipeline.sh
```

The script will:
1. Launch parallel training jobs (respecting `MAX_JOBS`),
2. Skip already completed runs (checks for final model files),
3. Save logs, checkpoints, and monitor CSVs to `./logs/`.

### 8.4. Monitoring Progress

Use TensorBoard to monitor training:

```bash
tensorboard --logdir ./logs
# Open http://localhost:6006 in your browser
```

---

## 8. Analysis & Visualization

### 8.1. Per-Experiment Plots (`plots.py`)

Generate detailed plots for a specific reward configuration:

```bash
python plots.py \
    --algorithm PPO \
    --reward speed_energy \
    --reward-param w_forward=1.0 \
    --reward-param w_ctrl=1.0 \
    --reward-param w_survive=1.0 \
    --timesteps 800000 \
    --log-dir ./logs \
    --output-dir ./plots
```

This generates for each configuration:
- **Learning Curve**: Smoothed reward over timesteps (comparing noise levels)
- **Stability Boxplot**: Episode length distribution (last 100 episodes)
- **Performance Barplot**: Final reward with standard deviation
- **Relative Drop**: Performance degradation vs clean baseline
- **AUC (Sample Efficiency)**: Area under the learning curve

Outputs are saved to `./plots/<algorithm>_<reward>_<params>/`.

### 8.2. Batch Plot Generation (`plots_pipeline.sh`)

Generate plots for all configured experiments:

```bash
chmod +x plots_pipeline.sh
./plots_pipeline.sh
```

### 8.3. Global Synthesis Plots (`plots_synthesis.py`)

Generate high-level summary figures comparing all experiments:

```bash
python plots_synthesis.py --log-dir ./final_logs/logs
```

This creates:
1. **Global Stability**: Boxplot of episode lengths under combined noise
2. **Relative Robustness**: Bar chart of % performance drop (Clean → Combined)
3. **Learning Dynamics Grid**: 2×2 grid comparing learning curves per reward type

Outputs are saved to `./final_logs/plots/synthesis/`.

### 8.4. LaTeX Results Table (`latex_summary.py`)

Generate a comprehensive LaTeX table for academic reports:

```bash
python latex_summary.py --log-dir ./logs
```

This produces `results_table_v2.tex` with:
- Rows: Algorithm × Reward × Variant configurations
- Columns: Noise scenarios (Clean, Obs Only, Act Only, Combined)
- Cells: Mean reward ± std (episode length)

---

## 9. Train.py Advanced Options

The training script supports additional features:

### 9.1. VecNormalize (Enabled by Default)

Observation normalization helps with training stability:

```bash
# Disable VecNormalize
python train.py --reward speed_energy --no-vecnormalize

# Normalize rewards as well (usually keep False)
python train.py --reward speed_energy --norm-reward

# Adjust observation clipping
python train.py --reward speed_energy --clip-obs 10.0
```

### 9.2. Resume Training

Resume from the latest checkpoint:

```bash
python train.py --reward speed_energy --timesteps 1000000 --resume
```

The script will:
1. Find the latest model checkpoint in `./logs/checkpoints/`
2. Load VecNormalize statistics
3. Merge monitor logs for continuous tracking

### 9.3. Full CLI Reference

```bash
python train.py --help
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--algorithm` | SAC | RL algorithm (SAC, PPO) |
| `--reward` | required | Reward function name |
| `--timesteps` | 100000 | Training steps |
| `--reward-param` | - | Hyperparameter (repeatable) |
| `--obs-noise-std` | 0.0 | Observation noise σ |
| `--action-noise-std` | 0.0 | Action noise σ |
| `--seed` | 42 | Random seed |
| `--video` | False | Record evaluation video |
| `--resume` | False | Resume from checkpoint |
| `--log-dir` | ./logs | Log directory |

---

## 10. Project File Summary

| File | Description |
|------|-------------|
| `reward_function.py` | 7 custom reward functions with registry |
| `walker2d_env.py` | Generic reward wrapper + environment factory |
| `train.py` | CLI training script with checkpointing |
| `train_pipeline.sh` | Batch training across configs/noise levels |
| `plots.py` | Per-experiment visualization |
| `plots_pipeline.sh` | Batch plot generation |
| `plots_synthesis.py` | Global comparison figures |
| `latex_summary.py` | LaTeX table generator for reports |
| `manual_merge.py` | Utility for merging interrupted logs |

---

## 11. Example Workflow

1. **Run a quick test** to verify setup:
   ```bash
   python train.py --reward speed_energy --timesteps 10000 --video
   ```

2. **Launch full campaign** (takes several hours):
   ```bash
   ./train_pipeline.sh
   ```

3. **Monitor training**:
   ```bash
   tensorboard --logdir ./logs
   ```

4. **Generate all plots**:
   ```bash
   ./plots_pipeline.sh
   python plots_synthesis.py --log-dir ./logs
   ```

5. **Create results table**:
   ```bash
   python latex_summary.py --log-dir ./logs
   ```

---

## 12. Troubleshooting

* **Import error on `np.float_`**
  If you use NumPy 2.x, replace `np.float_` with `np.float64` in all type annotations.

* **Reward name not found**
  Make sure your reward function is correctly decorated:

  ```python
  @register_reward("speed_energy")
  def reward_speed_energy(...):
      ...
  ```

* **Black video**
  Training and video recording are separated. Ensure evaluation uses:
  - `record_video=True`
  - `render_mode="rgb_array"`

* **Out of memory with parallel jobs**
  Reduce `MAX_JOBS` in `train_pipeline.sh`.

* **Missing log files for plots**
  Ensure filenames match the expected pattern:
  ```
  monitor-walker2d-{ALGO}-{REWARD}-ts{TIMESTEPS}-seed{SEED}-obsnoise{OBS}-actnoise{ACT}-rewardparams{PARAMS}.csv.monitor.csv
  ```

---

This setup enables systematic comparison of **reward shaping strategies** under various **perturbation scenarios** for robustness analysis on the Walker2d locomotion task.
