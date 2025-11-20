````markdown
# Walker2d Reward Shaping Project

This project investigates **reward shaping** and **robustness to perturbations** on the `Walker2d-v5` environment using multiple custom reward functions and a generic training script.

We use:
- [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium) with `Walker2d-v5` (MuJoCo),
- [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3) (SAC algorithm),
- A **generic reward interface** so you can switch reward functions from the command line.

Later, perturbations (noise, random pushes, domain randomization, etc.) can be added at clearly marked places in the environment wrapper.

---

## 1. Project Structure

Typical layout:

```text
rl-project/
  reward_function.py      # all 7 custom reward functions + registry REWARD_FNS
  walker2d_env.py         # generic Walker2d environment wrapper using custom rewards
  train.py                # command-line training script (SAC)
  videos_cli/             # (created at runtime) evaluation videos
  notebooks/              # (optional) Jupyter notebooks for analysis
  README.md
````

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

## 6. Perturbations (Future Work)

The file `walker2d_env.py` contains several `# TODO: perturbations` markers where you can later add:

* **Action noise** (sensor/actuator noise),
* **Observation noise**,
* **Random pushes** on the torso,
* **Domain randomization** at reset (mass, friction, etc.).

These are the hooks you will use to run **stress tests** on each reward function and evaluate robustness.

---

## 7. Troubleshooting

* **Import error on `np.float_`**
  If you use NumPy 2.x, replace `np.float_` with `np.float64` in all type annotations.

* **Reward name not found**
  Make sure your reward function is correctly decorated:

  ```python
  @register_reward("speed_energy")
  def reward_speed_energy(...):
      ...
  ```

  And that `reward_function.py` is in the Python path.

* **Black video**
  Training and video recording are separated:

  * Training env: `record_video=False`
  * Evaluation env: `record_video=True, render_mode="rgb_array"`

  If you still get black frames, check your MuJoCo / GPU / ffmpeg setup.

---

This setup should let you train and compare **all seven reward functions** from the command line, and extend the project later with various perturbations to study robustness.

```
::contentReference[oaicite:0]{index=0}
```
