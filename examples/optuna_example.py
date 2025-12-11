import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld
from src.agents.actor_critic import ActorCriticAgent
from src.utils.visualization import Visualizer

import torch
import numpy as np
import optuna


# ───────────────────────── Helper: build env ─────────────────────────

def make_env(size: int = 9) -> GridWorld:
    """Create GridWorld with the custom goal + trap configuration."""
    env = GridWorld(size=size)

    # Set terminal states that don't conflict with starting position (0,0)
    goal_state = env._pos_to_state((4, 6))  # Bottom-right-ish
    trap_states = [
        env._pos_to_state((2, 5)),
        env._pos_to_state((2, 4)),
        env._pos_to_state((3, 4)),
        env._pos_to_state((4, 4)),
        env._pos_to_state((5, 4))
    ]

    # Clear default terminal states and set new ones
    env.terminal_states.clear()
    env.terminal_states[goal_state] = 10.0
    for trap_state in trap_states:
        env.terminal_states[trap_state] = -2.0

    return env


# ─────────────────── Helper: train & return metric ────────────────────

def train_and_evaluate(
    learning_rate_actor: float,
    learning_rate_critic: float,
    entropy_coef: float,
    gamma: float = 0.99,
    optimizer_name: str = "Muon",
    n_episodes: int = 500,
    max_steps: int = 100,
    seed: int = 42,
):
    """
    Train an Actor-Critic agent with given hyperparameters and
    return (mean_reward_last_100, rewards, lengths, agent, env).

    During Optuna search we mostly care about the scalar metric
    (mean reward over the last 100 episodes).
    """
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(size=9)

    agent = ActorCriticAgent(
        env,
        learning_rate_actor=learning_rate_actor,
        learning_rate_critic=learning_rate_critic,
        gamma=gamma,
        entropy_coef=entropy_coef,
        optimizer=optimizer_name,
    )

    rewards, lengths = agent.train(
        n_episodes=n_episodes,
        max_steps=max_steps
    )

    # Metric: average reward over the last 100 episodes (or fewer if n_episodes < 100)
    window = min(100, len(rewards))
    mean_last_rewards = float(np.mean(rewards[-window:]))

    return mean_last_rewards, rewards, lengths, agent, env


# ───────────────────── Optuna objective function ──────────────────────

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective: given a trial, sample hyperparameters, train, and
    return the scalar performance metric to MAXIMIZE.
    """

    # Search space: two learning rates + entropy coefficient
    lr_actor = trial.suggest_float("learning_rate_actor", 1e-5, 1e-2, log=True)
    lr_critic = trial.suggest_float("learning_rate_critic", 1e-5, 1e-2, log=True)
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.05)

    # You can optionally search over gamma too, but keeping it fixed for now.
    gamma = 0.99

    # Different seed per trial to avoid overfitting noise to one RNG draw
    seed = 42 + trial.number

    # Shorter training for tuning to keep things fast
    mean_last_rewards, _, _, _, _ = train_and_evaluate(
        learning_rate_actor=lr_actor,
        learning_rate_critic=lr_critic,
        entropy_coef=entropy_coef,
        gamma=gamma,
        optimizer_name="Muon",
        n_episodes=400,   # shorter than full 2000 for speed
        max_steps=100,
        seed=seed,
    )

    return mean_last_rewards


# ──────────────── Main: run Optuna, then final training ───────────────

def run_optuna_and_train_final(
    n_trials: int = 30,
    final_episodes: int = 2000,
    max_steps: int = 100,
):
    # Global seed for reproducibility of the search process itself
    torch.manual_seed(42)
    np.random.seed(42)

    print("Starting Optuna hyperparameter search...")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n=== Optuna finished ===")
    print("Best value (mean reward on last episodes):", study.best_value)
    print("Best hyperparameters:", study.best_params)

    best_lr_actor = study.best_params["learning_rate_actor"]
    best_lr_critic = study.best_params["learning_rate_critic"]
    best_entropy_coef = study.best_params["entropy_coef"]

    # Retrain with best hyperparameters on a longer run + full visualization
    print("\nRetraining final Actor-Critic agent with best hyperparameters...")
    mean_last_rewards, rewards, lengths, agent, env = train_and_evaluate(
        learning_rate_actor=best_lr_actor,
        learning_rate_critic=best_lr_critic,
        entropy_coef=best_entropy_coef,
        gamma=0.99,
        optimizer_name="Muon",
        n_episodes=final_episodes,
        max_steps=max_steps,
        seed=123,  # fixed seed for the final run
    )

    print(f"Final run: mean reward over last 100 episodes = {mean_last_rewards:.3f}")

    # Plot training curves
    training_fig = Visualizer.plot_training_results(
        rewards=rewards,
        lengths=lengths,
        actor_losses=agent.actor_losses,
        critic_losses=agent.critic_losses,
        title='Actor-Critic Training Progress (Best Hyperparameters)'
    )
    training_fig.savefig('actor_critic_training_progress_best.png')

    # Get and display the learned policy
    policy = agent.get_optimal_policy()
    policy_fig = Visualizer.plot_policy(
        policy,
        env.size,
        "Actor-Critic Learned Policy (Best Hyperparameters)"
    )
    policy_fig.savefig("actor_critic_learned_policy_best.png")

    # Run and visualize a test episode
    print("\nRunning test episode with learned policy...")
    state = env.reset()
    episode_data = []
    done = False
    total_reward = 0

    while not done:
        action, _, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_data.append((state, action, reward))
        total_reward += reward
        state = next_state

    episode_fig = Visualizer.visualize_episode_trajectory(
        env,
        episode_data,
        "Actor-Critic Test Episode Trajectory (Best Hyperparameters)"
    )
    episode_fig.savefig("actor_critic_test_episode_trajectory_best.png")

    print(f"Test episode finished with total reward: {total_reward}")

    return agent, policy, total_reward, study


if __name__ == "__main__":
    # Adjust n_trials down if this is too slow / too many runs
    agent, policy, final_reward, study = run_optuna_and_train_final(
        n_trials=30,
        final_episodes=2000,
        max_steps=100,
    )
