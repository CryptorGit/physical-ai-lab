"""Go2FlatEnvがGymnasium/SB3互換か確認する。"""

from stable_baselines3.common.env_checker import check_env

from env import Go2FlatEnv


def main() -> None:
    """環境検査と短いランダム実行を行う。"""
    env = Go2FlatEnv()

    check_env(env, warn=True)

    observation, info = env.reset(seed=42)

    print(f"Observation shape: {observation.shape}")
    print(f"Action shape: {env.action_space.shape}")

    total_reward = 0.0

    for step in range(200):
        action = env.action_space.sample()

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        if terminated or truncated:
            print(
                f"Episode ended at step={step}, "
                f"height={info['base_height']:.3f}, "
                f"reward={total_reward:.3f}"
            )
            observation, info = env.reset()

    env.close()
    print("Environment check completed.")


if __name__ == "__main__":
    main()