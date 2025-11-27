"""
Environment utilities for Walker2d-v5 using custom reward functions.

This module provides:
- A generic wrapper that calls any reward function registered in
  reward_function.REWARD_FNS.
- A helper function `make_walker2d_env` to build a training or evaluation env,
  optionally with video recording.

Places where you might want to add perturbations later are marked with
`# TODO: perturbations`.
"""

from typing import Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import csv

# Import the reward registry from your reward file
from reward_function import REWARD_FNS


# Type aliases (for clarity, optional)
RewardState = Dict[str, Any]

import csv
import numpy as np

def save_noise_csv(filename, obs_noise_list, action_noise_list):
    if len(obs_noise_list) != len(action_noise_list):
        raise ValueError("The lengths of obs_noise_list and action_noise_list must be the same.")

    obs_dim = len(obs_noise_list[0])
    act_dim = len(action_noise_list[0])

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["timestep"]
        header += [f"obs_noise_{i}" for i in range(obs_dim)]
        header += [f"action_noise_{i}" for i in range(act_dim)]
        writer.writerow(header)

        for t, (obs_n, act_n) in enumerate(zip(obs_noise_list, action_noise_list)):
            if isinstance(obs_n, np.ndarray):
                obs_n = list(obs_n)
            if isinstance(act_n, np.ndarray):
                act_n = list(act_n)
            row = [t] + obs_n + act_n
            writer.writerow(row)

class GenericRewardWrapper(gym.Wrapper):
    """
    Generic reward wrapper for Walker2d-v5.

    It:
    - calls the base environment to get (obs, base_reward, terminated, truncated, info)
    - replaces/shapes the reward using the selected custom reward function
      from REWARD_FNS[reward_name]
    - maintains a small internal `reward_state` dict (for previous actions, etc.)
    """

    def __init__(
        self,
        env: gym.Env,
        reward_name: str,
        reward_params: Optional[Dict[str, Any]] = None,
        obs_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
        log_dir: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        super().__init__(env)

        if reward_name not in REWARD_FNS:
            raise ValueError(
                f"Unknown reward '{reward_name}'. "
                f"Available rewards: {list(REWARD_FNS.keys())}"
            )

        self.reward_name = reward_name
        self.reward_fn = REWARD_FNS[reward_name]
        self.reward_params: Dict[str, Any] = reward_params or {}

        # Internal state for the reward (previous obs, actions, etc.)
        self.reward_state: RewardState = {}

        #obs_noise
        self.obs_noise_std = obs_noise_std
        self.action_noise_std = action_noise_std
        self.obs_noise_list = []
        self.action_noise_list = []
        self._step_count = 0
        print(
            f"[Env] GenericRewardWrapper initialized with "
            f"obs_noise_std={self.obs_noise_std}, action_noise_std={self.action_noise_std}"
        )

        if log_dir is None:
            self.log_dir = "./logs"
        else:
            self.log_dir = log_dir

        if run_name is None:
            self.run_name = "default_run"
        else:
            self.run_name = run_name

    def reset(self, **kwargs):
        """
        Reset the environment and the internal reward state.
        """
        # TODO: perturbations
        # Here you could add reset-time perturbations later, e.g.:
        # - randomize the initial state more aggressively
        # - inject domain randomization parameters
        self._step_count = 0
        obs, info = self.env.reset(**kwargs)

        # Reset the reward internal state at the beginning of each episode
        self.reward_state = {}
        return obs, info

    def step(self, action):
        """
        Step the underlying environment and then compute the custom reward.

        Here we implement a simple framework to add observation and action noise,
        e.g. to simulate actuator noise, external pushes, sensor noise, etc.

        """
        action_to_env = np.array(action, copy=True)

        if self.action_noise_std > 0.0:
            action_noise = np.random.randn(*action_to_env.shape) * self.action_noise_std
            action_to_env += action_noise
            self.action_noise_list.append(action_noise)

            if self._step_count % 1000 == 0:
                noise_abs_mean = float(np.abs(action_noise).mean())
                noise_abs_max = float(np.abs(action_noise).max())
                print(
                    f"[ActionNoise] step={self._step_count} "
                    f"action_noise_std={self.action_noise_std:.4f} "
                    f"noise_abs_mean={noise_abs_mean:.4f} "
                    f"noise_abs_max={noise_abs_max:.4f}"
                )
                
        # Call the original environment
        obs, base_reward, terminated, truncated, info = self.env.step(action_to_env)

        if self.obs_noise_std > 0.0:
            noise = np.random.randn(*obs.shape) * self.obs_noise_std
            obs_for_agent = obs + noise
            self.obs_noise_list.append(noise)

            if self._step_count % 1000 == 0:
                noise_abs_mean = float(np.abs(noise).mean())
                noise_abs_max = float(np.abs(noise).max())
                print(
                    f"[Noise] step={self._step_count} "
                    f"obs_noise_std={self.obs_noise_std:.4f} "
                    f"noise_abs_mean={noise_abs_mean:.4f} "
                    f"noise_abs_max={noise_abs_max:.4f}"
                )
        else:
            obs_for_agent = obs

        # Compute the new reward using the selected reward function
        new_reward, new_state = self.reward_fn(
            obs=obs_for_agent,
            action=np.array(action_to_env),
            base_reward=float(base_reward),
            info=info,
            state=self.reward_state,
            params=self.reward_params,
        )
        self.reward_state = new_state

        if self._step_count % 1000 == 0 and (self.obs_noise_std > 0.0 or self.action_noise_std > 0.0):
            if self.obs_noise_std <= 0.0:
                self.obs_noise_list = [[0]]*len(self.action_noise_list)
            if self.action_noise_std <= 0.0:
                self.action_noise_list = [[0]]*len(self.obs_noise_list)
            filename = f"{self.log_dir}/noise-{self.run_name}.csv"
            save_noise_csv(filename, self.obs_noise_list, self.action_noise_list)
            print(f"[Env] Saved noise data to {filename}")

        self._step_count += 1

        return obs_for_agent, new_reward, terminated, truncated, info


def make_walker2d_env(
    reward_name: str,
    reward_params: Optional[Dict[str, Any]] = None,
    record_video: bool = False,
    video_folder: str = "./videos",
    video_prefix: Optional[str] = None,
    obs_noise_std: float = 0.0,
    action_noise_std: float = 0.0,
    log_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> gym.Env:
    """
    Create a Walker2d-v5 environment with a custom reward function.

    Parameters
    ----------
    reward_name : str
        Name of the reward function as registered in reward_function.REWARD_FNS
        (e.g. "speed_energy", "target_speed", "posture_stability", etc.).
    reward_params : dict, optional
        Dictionary of hyperparameters for the reward (weights, targets, ...).
    record_video : bool
        If True, wrap the environment with RecordVideo (for evaluation runs).
    video_folder : str
        Directory where videos will be saved if record_video=True.
    video_prefix : str, optional
        Prefix for video filenames. If None, defaults to "walker2d-{reward_name}".
    obs_noise_std : float
        Standard deviation of Gaussian noise to add to observations.
    action_noise_std : float
        Standard deviation of Gaussian noise to add to actions.
    log_dir : str, optional
        Directory for logging noise data.
    run_name : str, optional
        Name of the current run (for logging purposes).

    Returns
    -------
    env : gym.Env
        The wrapped Walker2d environment.
    """
    # Create the base environment.
    # We only request `render_mode="rgb_array"` when recording videos,
    # to avoid overhead during training.
    if record_video:
        base_env = gym.make("Walker2d-v5", render_mode="rgb_array")
    else:
        base_env = gym.make("Walker2d-v5")

    # Wrap it with our generic reward wrapper
    env = GenericRewardWrapper(
        base_env,
        reward_name=reward_name,
        reward_params=reward_params,
        obs_noise_std=obs_noise_std,
        action_noise_std=action_noise_std,
        log_dir=log_dir,
        run_name=run_name,
    )

    # TODO: perturbations
    # Here you could add additional wrappers later, for example:
    # - a custom ObservationWrapper that injects sensor noise
    # - a custom ActionWrapper that clips or rescales actions
    # - a custom wrapper that applies random pushes to the torso
    # Example (pseudo-code):
    #   env = RandomPushWrapper(env, push_prob=0.1, push_force=50.0)

    # Optionally wrap with RecordVideo for evaluation / visualization
    if record_video:
        prefix = video_prefix or f"walker2d-{reward_name}"
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=prefix,
            episode_trigger=lambda ep_id: True,  # record every episode
            video_length=0,                       # full episode
        )

    return env
