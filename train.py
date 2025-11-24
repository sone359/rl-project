"""
Command-line training script for Walker2d-v5 with custom reward functions.

Usage examples
--------------

# 1) Train with reward "speed_energy" and default parameters:
python train_cli.py --reward speed_energy --timesteps 100000

# 2) Train with "target_speed" and custom hyperparameters:
python train_cli.py --reward target_speed \
    --reward-param v_target=1.5 \
    --reward-param alpha=1.0 \
    --reward-param beta=0.001 \
    --timesteps 150000

# 3) Train with "posture_stability" and record a video after training:
python train_cli.py --reward posture_stability \
    --reward-param h_target=1.25 \
    --reward-param w_h=5.0 \
    --reward-param w_angle=1.0 \
    --timesteps 150000 \
    --video
"""

import argparse
from typing import Dict, Any

from stable_baselines3 import SAC

from walker2d_env import make_walker2d_env
from reward_function import REWARD_FNS


def parse_reward_params(param_list) -> Dict[str, Any]:
    """
    Parse a list of 'key=value' strings into a dict.

    Example:
        ["w_forward=1.0", "w_ctrl=0.5"] -> {"w_forward": 1.0, "w_ctrl": 0.5}

    Values are cast to float when possible, otherwise left as strings.
    """
    params: Dict[str, Any] = {}
    if param_list is None:
        return params

    for item in param_list:
        if "=" not in item:
            raise ValueError(f"Invalid reward-param '{item}'. Expected 'key=value'.")
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Try casting to float
        try:
            val_cast = float(val)
        except ValueError:
            val_cast = val
        params[key] = val_cast

    return params


def main():
    parser = argparse.ArgumentParser(description="Train SAC on Walker2d-v5 with custom rewards.")
    parser.add_argument(
        "--reward",
        type=str,
        required=True,
        help=f"Reward name. Available: {list(REWARD_FNS.keys())}",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Number of training timesteps.",
    )
    parser.add_argument(
        "--reward-param",
        action="append",
        help="Reward hyperparameter in the form key=value. "
             "Can be used multiple times. Example: --reward-param w_forward=1.0",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="If set, record a video of one evaluation episode after training.",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="./videos_cli",
        help="Folder where evaluation videos are saved (if --video is used).",
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to observations in the wrapper.",
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to actions in the wrapper.",
    )

    args = parser.parse_args()

    reward_params = parse_reward_params(args.reward_param)

    # -----------------------
    # 1) Training environment
    # -----------------------
    # NOTE:
    # We do NOT enable video during training, to avoid overhead and
    # known issues with black videos in some MuJoCo + RecordVideo setups.
    train_env = make_walker2d_env(
        reward_name=args.reward,
        reward_params=reward_params,
        record_video=False,
        obs_noise_std=args.obs_noise_std,
        action_noise_std=args.action_noise_std,
    )

    # Train SAC on the selected reward
    model = SAC("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    # -----------------------
    # 2) Optional evaluation + video
    # -----------------------
    if args.video:
        eval_env = make_walker2d_env(
            reward_name=args.reward,
            reward_params=reward_params,
            record_video=True,
            video_folder=args.video_folder,
            video_prefix=f"walker2d-{args.reward}",
            obs_noise_std=args.obs_noise_std,
            action_noise_std=args.action_noise_std,
        )

        obs, info = eval_env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

        eval_env.close()
        print(f"[INFO] Evaluation video saved to: {args.video_folder}")


if __name__ == "__main__":
    main()
