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

# Import the reward registry from your reward file
from reward_function import REWARD_FNS


# Type aliases (for clarity, optional)
RewardState = Dict[str, Any]


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
        self._step_count = 0
        print(
            f"[Env] GenericRewardWrapper initialized with "
            f"obs_noise_std={self.obs_noise_std}, action_noise_std={self.action_noise_std}"
        )

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
        """
        # TODO: perturbations
        # Here you could perturb the action before sending it to the env,
        # e.g. actuator noise, external pushes, action scaling, etc.
        # action_perturbed = action
        # action_perturbed = add_noise(action_perturbed, ...)
        # For now we keep it unchanged:
        #Observaion Noise
        self._step_count += 1
        #Action Noise


        action_to_env = np.array(action, copy=True)

        if self.action_noise_std > 0.0:
            action_noise = np.random.randn(*action_to_env.shape) * self.action_noise_std
            action_to_env = action_to_env + action_noise

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

        # TODO: perturbations
        obs_for_agent = obs

        if self.obs_noise_std > 0.0:
            noise = np.random.randn(*obs.shape) * self.obs_noise_std
            obs_for_agent = obs + noise

            if self._step_count % 1000 == 0:
                noise_abs_mean = float(np.abs(noise).mean())
                noise_abs_max = float(np.abs(noise).max())
                print(
                    f"[Noise] step={self._step_count} "
                    f"obs_noise_std={self.obs_noise_std:.4f} "
                    f"noise_abs_mean={noise_abs_mean:.4f} "
                    f"noise_abs_max={noise_abs_max:.4f}"
                )
        # Here you could perturb the observation or info AFTER the env step,
        # e.g. sensor noise, missing joints, etc.
        # obs_perturbed = obs
        # obs_perturbed = add_observation_noise(obs_perturbed, ...)
        # For now, we keep it unchanged:
        obs_for_reward = obs_for_agent

        # Compute the new reward using the selected reward function
        new_reward, new_state = self.reward_fn(
            obs=obs_for_reward,
            action=np.array(action_to_env),
            base_reward=float(base_reward),
            info=info,
            state=self.reward_state,
            params=self.reward_params,
        )
        self.reward_state = new_state

        return obs_for_agent, new_reward, terminated, truncated, info


def make_walker2d_env(
    reward_name: str,
    reward_params: Optional[Dict[str, Any]] = None,
    record_video: bool = False,
    video_folder: str = "./videos",
    video_prefix: Optional[str] = None,
    obs_noise_std: float = 0.0,
    action_noise_std: float = 0.0,

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
