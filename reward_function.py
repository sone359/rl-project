"""
Reward functions for Walker2d-v5.

Each reward has the same generic interface:

    reward_fn(
        obs: np.ndarray,
        action: np.ndarray,
        base_reward: float,
        info: dict,
        state: dict,
        params: dict,
    ) -> (new_reward: float, new_state: dict)

- obs:        observation returned by env.step()
- action:     action applied at this step
- base_reward: original environment reward (Walker2d default reward)
- info:       info dict returned by env.step()
- state:      internal state for the reward function (for history, etc.)
- params:     hyperparameters for this reward (weights, targets, ...)

Most rewards either:
- replace the base_reward by their own computation (e.g. speed/energy),
- or shape the base_reward by subtracting penalties.

The registry REWARD_FNS allows selecting a reward by name from the CLI
or from a wrapper.
"""

from typing import Dict, Any, Tuple, Callable
import numpy as np
import numpy.typing as npt

# Type aliases
Obs = npt.NDArray[np.float64]
Action = npt.NDArray[np.float64]
InfoDict = Dict[str, Any]
RewardState = Dict[str, Any]

RewardFn = Callable[
    [Obs, Action, float, InfoDict, RewardState, Dict[str, Any]],
    Tuple[float, RewardState],
]

REWARD_FNS: Dict[str, RewardFn] = {}


def register_reward(name: str):
    """
    Decorator used to register a reward function into REWARD_FNS.

    Example
    -------
    @register_reward("speed_energy")
    def reward_speed_energy(...):
        ...
    """
    def deco(fn: RewardFn) -> RewardFn:
        if name in REWARD_FNS:
            raise ValueError(f"Reward '{name}' is already registered.")
        REWARD_FNS[name] = fn
        return fn

    return deco



# ============================================================================
# 1) Reward: speed / energy / survival (baseline-style)
# ============================================================================

@register_reward("speed_energy")
def reward_speed_energy(
    obs: Obs,
    action: Action,
    base_reward: float,
    info: InfoDict,
    state: RewardState,
    params: Dict[str, Any],
) -> Tuple[float, RewardState]:
    """
    Reward 1: combine forward speed / control cost / survival.

    This is essentially a re-weighted version of the standard Walker2d reward:
        r_t = w_forward * reward_forward
            + w_ctrl * reward_ctrl     (already negative)
            + w_survive * reward_survive
    """
    w_forward = float(params.get("w_forward", 1.0))
    w_ctrl = float(params.get("w_ctrl", 1.0))
    w_survive = float(params.get("w_survive", 1.0))

    forward = float(info.get("reward_forward", 0.0))
    ctrl = float(info.get("reward_ctrl", 0.0))          # already negative
    survive = float(info.get("reward_survive", 0.0))

    new_reward = w_forward * forward + w_ctrl * ctrl + w_survive * survive
    return new_reward, state



# ============================================================================
# 2) Reward: target speed (tracking a desired walking speed)
# ============================================================================

@register_reward("target_speed")
def reward_target_speed(
    obs: Obs,
    action: Action,
    base_reward: float,
    info: InfoDict,
    state: RewardState,
    params: Dict[str, Any],
) -> Tuple[float, RewardState]:
    """
    Reward 2: target speed + energy + survival.

    We want the agent to walk at a given target speed v_target instead of
    "the faster the better":

        r_t = - alpha * |v_x - v_target|
              - beta  * ||a_t||^2
              + w_survive * reward_survive

    where:
    - v_x = torso horizontal velocity (obs[8] for Walker2d-v5),
    - a_t = action vector at time t.
    """
    v_target = float(params.get("v_target", 1.5))
    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", 1e-3))
    w_survive = float(params.get("w_survive", 1.0))

    vx = float(obs[8])  # torso x-velocity
    energy = float(np.sum(action**2))
    survive = float(info.get("reward_survive", 0.0))

    speed_term = -alpha * abs(vx - v_target)
    energy_term = -beta * energy

    new_reward = speed_term + energy_term + w_survive * survive
    return new_reward, state


# ============================================================================
# 3) Reward: posture stability (keep the torso high and upright)
# ============================================================================

@register_reward("posture_stability")
def reward_posture_stability(
    obs: Obs,
    action: Action,
    base_reward: float,
    info: InfoDict,
    state: RewardState,
    params: Dict[str, Any],
) -> Tuple[float, RewardState]:
    """
    Reward 3: posture stability.

    Idea:
      - keep the usual forward/ctrl/survive terms,
      - add penalties when the torso becomes too low or too tilted.

    We use:
        base_terms = w_forward * reward_forward
                   + w_ctrl    * reward_ctrl
                   + w_survive * reward_survive

        height_penalty = - w_h * max(0, h_target - h)^2
        angle_penalty = - w_angle * theta^2

    where:
      - h = torso height (obs[0]),
      - theta = torso angle (obs[1]) (0 = upright).
    """
    h_target = float(params.get("h_target", 1.25))
    w_forward = float(params.get("w_forward", 1.0))
    w_ctrl = float(params.get("w_ctrl", 1.0))
    w_survive = float(params.get("w_survive", 1.0))
    w_h = float(params.get("w_h", 1.0))
    w_angle = float(params.get("w_angle", 1.0))

    forward = float(info.get("reward_forward", 0.0))
    ctrl = float(info.get("reward_ctrl", 0.0))
    survive = float(info.get("reward_survive", 0.0))

    base_terms = w_forward * forward + w_ctrl * ctrl + w_survive * survive

    h = float(obs[0])
    angle = float(obs[1])

    # Penalize if the torso falls below the target height
    height_penalty = -w_h * max(0.0, h_target - h)**2
    # Penalize large torso angles (we want angle ≈ 0)
    angle_penalty = -w_angle * angle**2

    new_reward = base_terms + height_penalty + angle_penalty
    return new_reward, state


# ============================================================================
# 4) Reward: smooth actions (anti-nervous behavior)
# ============================================================================

@register_reward("smooth_actions")
def reward_smooth_actions(
    obs: Obs,
    action: Action,
    base_reward: float,
    info: InfoDict,
    state: RewardState,
    params: Dict[str, Any],
) -> Tuple[float, RewardState]:
    """
    Reward 4: smooth actions.

    We penalize large changes in actions between two consecutive steps:

        r_t = base_reward_t - lambda_smooth * ||a_t - a_{t-1}||^2

    This encourages smoother, less jittery control policies.
    """
    lambda_smooth = float(params.get("lambda_smooth", 1e-2))
    prev_action = state.get("prev_action", None)

    if prev_action is None:
        # No previous action at the beginning of the episode -> no penalty.
        new_reward = float(base_reward)
    else:
        delta = action - prev_action
        smooth_penalty = lambda_smooth * float(np.sum(delta**2))
        new_reward = float(base_reward) - smooth_penalty

    # Update state for next step
    state["prev_action"] = np.array(action, copy=True)
    return new_reward, state


# ============================================================================
# 5) Reward: dynamic stability (penalize state "jerk")
# ============================================================================

@register_reward("dynamic_stability")
def reward_dynamic_stability(
    obs: Obs,
    action: Action,
    base_reward: float,
    info: InfoDict,
    state: RewardState,
    params: Dict[str, Any],
) -> Tuple[float, RewardState]:
    """
    Reward 5: dynamic stability.

    We penalize "jerky" changes in the state by approximating a second-order
    finite difference:

        accel_t ≈ obs_t - 2 * obs_{t-1} + obs_{t-2}

    and then:

        r_t = base_reward_t - lambda_state * ||accel_t||^2

    This encourages smoother state trajectories over time.
    """
    lambda_state = float(params.get("lambda_state", 1e-2))

    prev_obs = state.get("prev_obs", None)
    prev_prev_obs = state.get("prev_prev_obs", None)

    if prev_obs is None or prev_prev_obs is None:
        new_reward = float(base_reward)
    else:
        accel = obs - 2.0 * prev_obs + prev_prev_obs
        accel_penalty = lambda_state * float(np.sum(accel**2))
        new_reward = float(base_reward) - accel_penalty

    # Update state history
    state["prev_prev_obs"] = state.get("prev_obs", None)
    state["prev_obs"] = np.array(obs, copy=True)
    return new_reward, state


# ============================================================================
# 6) Reward: progressive anti-fall shaping
# ============================================================================

@register_reward("anti_fall_progressive")
def reward_anti_fall_progressive(
    obs: Obs,
    action: Action,
    base_reward: float,
    info: InfoDict,
    state: RewardState,
    params: Dict[str, Any],
) -> Tuple[float, RewardState]:
    """
    Reward 6: progressive anti-fall shaping.

    Instead of only punishing at the moment of the fall, we add progressive
    penalties when the torso becomes too low or too tilted:

        r_t = base_reward_t
              - w_h     * max(0, h_crit - h_t)^2
              - w_angle * max(0, |theta_t| - angle_crit)^2

    where:
      - h_t = torso height (obs[0]),
      - theta_t = torso angle (obs[1]).
    """
    h_crit = float(params.get("h_crit", 0.9))
    angle_crit = float(params.get("angle_crit", 0.5))
    w_h = float(params.get("w_h", 5.0))
    w_angle = float(params.get("w_angle", 1.0))

    h = float(obs[0])
    angle = float(obs[1])

    height_danger = max(0.0, h_crit - h)
    angle_danger = max(0.0, abs(angle) - angle_crit)

    height_penalty = w_h * height_danger**2
    angle_penalty = w_angle * angle_danger**2

    new_reward = float(base_reward) - height_penalty - angle_penalty
    return new_reward, state


# ============================================================================
# 7) Reward: robust & economical walking (speed vs energy)
# ============================================================================

@register_reward("robust_econ")
def reward_robust_econ(
    obs: Obs,
    action: Action,
    base_reward: float,
    info: InfoDict,
    state: RewardState,
    params: Dict[str, Any],
) -> Tuple[float, RewardState]:
    """
    Reward 7: robust & economical walking (speed vs energy).

    A simple multi-objective reward combining:
      - forward speed (v_x),
      - energy use (||a_t||^2),
      - optionally, survival.

    We use:

        r_t = v_weight      * v_x
            - energy_weight * ||a_t||^2
            + w_survive     * reward_survive
    """
    v_weight = float(params.get("v_weight", 1.0))
    energy_weight = float(params.get("energy_weight", 1e-3))
    w_survive = float(params.get("w_survive", 0.0))

    vx = float(obs[8])                    # torso x-velocity
    energy = float(np.sum(action**2))
    survive = float(info.get("reward_survive", 0.0))

    new_reward = v_weight * vx - energy_weight * energy + w_survive * survive
    return new_reward, state
