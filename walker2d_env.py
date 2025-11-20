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

    def reset(self, **kwargs):
        """
        Reset the environment and the internal reward state.
        """
        # TODO: perturbations
        # Here you could add reset-time perturbations later, e.g.:
        # - randomize the initial state more aggressively
        # - inject domain randomization parameters
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
        action_to_env = action

        # Call the original environment
        obs, base_reward, terminated, truncated, info = self.env.step(action_to_env)

        # TODO: perturbations
        # Here you could perturb the observation or info AFTER the env step,
        # e.g. sensor noise, missing joints, etc.
        # obs_perturbed = obs
        # obs_perturbed = add_observation_noise(obs_perturbed, ...)
        # For now, we keep it unchanged:
        obs_for_reward = obs

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

        return obs, new_reward, terminated, truncated, info


def make_walker2d_env(
    reward_name: str,
    reward_params: Optional[Dict[str, Any]] = None,
    record_video: bool = False,
    video_folder: str = "./videos",
    video_prefix: Optional[str] = None,
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
