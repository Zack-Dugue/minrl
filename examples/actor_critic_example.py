import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld
from src.agents.actor_critic import ActorCriticAgent
from src.utils.visualization import Visualizer

import torch
import numpy as np
import matplotlib.pyplot as plt  # NEW


def train_with_l1_logging(
    agent: ActorCriticAgent,
    env: GridWorld,
    n_episodes: int = 2000,
    max_steps: int = 100,
):
    """
    Custom training loop that mirrors ActorCriticAgent.train but also
    logs L1 gradient norms, step norms, relative steps, grad–step cosine,
    policy KL, and TD-error stats for both actor and critic.
    """

    # Make sure the tracking lists exist
    if not hasattr(agent, "episode_rewards"):
        agent.episode_rewards = []
    if not hasattr(agent, "episode_lengths"):
        agent.episode_lengths = []
    if not hasattr(agent, "actor_losses"):
        agent.actor_losses = []
    if not hasattr(agent, "critic_losses"):
        agent.critic_losses = []
    if not hasattr(agent, "entropy_losses"):
        agent.entropy_losses = []

    # Per-episode L1 stats lists
    agent.actor_L1_grad_norm_avgs = []
    agent.actor_L1_step_norm_avgs = []
    agent.actor_L1_rel_step_avgs = []
    agent.critic_L1_grad_norm_avgs = []
    agent.critic_L1_step_norm_avgs = []
    agent.critic_L1_rel_step_avgs = []

    # NEW: per-episode grad–step cosine similarity
    agent.actor_L1_cos_g_step_avgs = []
    agent.critic_L1_cos_g_step_avgs = []

    # NEW: per-episode policy KL and TD-error stats
    agent.policy_kl_avgs = []
    agent.td_error_mean = []
    agent.td_error_std = []
    agent.td_error_abs_mean = []

    eps = 1e-12

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        steps = 0

        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        entropy_loss_sum = 0.0

        # Per-episode accumulators for L1 stats
        actor_L1_grad_norm_sum = 0.0
        actor_L1_step_norm_sum = 0.0
        actor_L1_rel_step_sum = 0.0

        critic_L1_grad_norm_sum = 0.0
        critic_L1_step_norm_sum = 0.0
        critic_L1_rel_step_sum = 0.0

        # NEW: accumulators for grad–step cosine
        actor_L1_cos_sum = 0.0
        critic_L1_cos_sum = 0.0

        # NEW: accumulators for policy KL and TD error
        kl_sum = 0.0
        td_error_sum = 0.0
        td_error_sq_sum = 0.0
        td_error_abs_sum = 0.0
        td_count = 0

        l1_step_count = 0

        for step in range(max_steps):
            # Select action (uses its own state_to_tensor & no_grad internally)
            action, log_prob, value_scalar = agent.select_action(state)

            # --- Build state tensor once for diagnostics ---
            state_tensor = agent.state_to_tensor(state)

            # Policy distribution BEFORE update (for KL)
            with torch.no_grad():
                dist_old = agent.actor(state_tensor)

            # Take action in environment
            next_state, reward, done, _ = env.step(action)

            # TD-error diagnostics BEFORE update (same formula as in agent.update)
            with torch.no_grad():
                next_state_tensor = agent.state_to_tensor(next_state)
                value = agent.critic(state_tensor)
                next_value = agent.critic(next_state_tensor)
                td_target = reward + (1 - int(done)) * agent.gamma * next_value
                td_error = (td_target - value).item()

            td_error_sum += td_error
            td_error_sq_sum += td_error ** 2
            td_error_abs_sum += abs(td_error)
            td_count += 1

            # Snapshot L1 weights BEFORE the update
            actor_L1_before = agent.actor.L1.weight.detach().clone()
            critic_L1_before = agent.critic.L1.weight.detach().clone()

            # Update networks (backward + optimizer.step() for both)
            actor_loss, critic_loss, entropy_loss = agent.update(
                state, action, reward, next_state, done
            )

            # After update(), .grad fields correspond to the last backward pass.

            # Actor L1 stats
            a_grad = agent.actor.L1.weight.grad
            if a_grad is not None:
                a_grad_flat = a_grad.detach().flatten()
                actor_grad_norm = a_grad_flat.norm().item()
            else:
                a_grad_flat = None
                actor_grad_norm = 0.0

            actor_L1_after = agent.actor.L1.weight.detach()
            actor_step = actor_L1_after - actor_L1_before
            actor_step_flat = actor_step.flatten()
            actor_step_norm = actor_step_flat.norm().item()
            actor_param_norm = actor_L1_after.norm().item()
            actor_rel_step = actor_step_norm / (actor_param_norm + eps)

            # Critic L1 stats
            c_grad = agent.critic.L1.weight.grad
            if c_grad is not None:
                c_grad_flat = c_grad.detach().flatten()
                critic_grad_norm = c_grad_flat.norm().item()
            else:
                c_grad_flat = None
                critic_grad_norm = 0.0

            critic_L1_after = agent.critic.L1.weight.detach()
            critic_step = critic_L1_after - critic_L1_before
            critic_step_flat = critic_step.flatten()
            critic_step_norm = critic_step_flat.norm().item()
            critic_param_norm = critic_L1_after.norm().item()
            critic_rel_step = critic_step_norm / (critic_param_norm + eps)

            # NEW: cosine similarity between grad and step for L1
            if a_grad_flat is not None and actor_grad_norm > 0.0 and actor_step_norm > 0.0:
                actor_cos = torch.dot(a_grad_flat, actor_step_flat).item() / (
                    actor_grad_norm * actor_step_norm + eps
                )
            else:
                actor_cos = 0.0

            if c_grad_flat is not None and critic_grad_norm > 0.0 and critic_step_norm > 0.0:
                critic_cos = torch.dot(c_grad_flat, critic_step_flat).item() / (
                    critic_grad_norm * critic_step_norm + eps
                )
            else:
                critic_cos = 0.0

            actor_L1_cos_sum += actor_cos
            critic_L1_cos_sum += critic_cos

            # Accumulate per-episode L1 stats
            actor_L1_grad_norm_sum += actor_grad_norm
            actor_L1_step_norm_sum += actor_step_norm
            actor_L1_rel_step_sum += actor_rel_step

            critic_L1_grad_norm_sum += critic_grad_norm
            critic_L1_step_norm_sum += critic_step_norm
            critic_L1_rel_step_sum += critic_rel_step

            l1_step_count += 1

            actor_loss_sum += actor_loss
            critic_loss_sum += critic_loss
            entropy_loss_sum += entropy_loss

            episode_reward += reward
            steps += 1

            # Policy distribution AFTER update (for KL)
            with torch.no_grad():
                dist_new = agent.actor(state_tensor)
                kl = torch.distributions.kl.kl_divergence(dist_old, dist_new).mean().item()
            kl_sum += kl

            if done:
                break

            state = next_state

        # Episode-level stats
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(steps)
        agent.actor_losses.append(actor_loss_sum / max(steps, 1))
        agent.critic_losses.append(critic_loss_sum / max(steps, 1))
        agent.entropy_losses.append(entropy_loss_sum / max(steps, 1))

        # Episode-average L1 stats
        if l1_step_count > 0:
            actor_L1_grad_avg = actor_L1_grad_norm_sum / l1_step_count
            actor_L1_step_avg = actor_L1_step_norm_sum / l1_step_count
            actor_L1_rel_step_avg = actor_L1_rel_step_sum / l1_step_count

            critic_L1_grad_avg = critic_L1_grad_norm_sum / l1_step_count
            critic_L1_step_avg = critic_L1_step_norm_sum / l1_step_count
            critic_L1_rel_step_avg = critic_L1_rel_step_sum / l1_step_count

            actor_L1_cos_avg = actor_L1_cos_sum / l1_step_count
            critic_L1_cos_avg = critic_L1_cos_sum / l1_step_count
        else:
            actor_L1_grad_avg = actor_L1_step_avg = actor_L1_rel_step_avg = 0.0
            critic_L1_grad_avg = critic_L1_step_avg = critic_L1_rel_step_avg = 0.0
            actor_L1_cos_avg = critic_L1_cos_avg = 0.0

        agent.actor_L1_grad_norm_avgs.append(actor_L1_grad_avg)
        agent.actor_L1_step_norm_avgs.append(actor_L1_step_avg)
        agent.actor_L1_rel_step_avgs.append(actor_L1_rel_step_avg)

        agent.critic_L1_grad_norm_avgs.append(critic_L1_grad_avg)
        agent.critic_L1_step_norm_avgs.append(critic_L1_step_avg)
        agent.critic_L1_rel_step_avgs.append(critic_L1_rel_step_avg)

        agent.actor_L1_cos_g_step_avgs.append(actor_L1_cos_avg)
        agent.critic_L1_cos_g_step_avgs.append(critic_L1_cos_avg)

        # Episode-average KL
        if l1_step_count > 0:
            kl_avg = kl_sum / l1_step_count
        else:
            kl_avg = 0.0
        agent.policy_kl_avgs.append(kl_avg)

        # Episode TD-error stats
        if td_count > 0:
            td_mean = td_error_sum / td_count
            td_abs_mean = td_error_abs_sum / td_count
            td_var = max(td_error_sq_sum / td_count - td_mean ** 2, 0.0)
            td_std = td_var ** 0.5
        else:
            td_mean = td_abs_mean = td_std = 0.0

        agent.td_error_mean.append(td_mean)
        agent.td_error_std.append(td_std)
        agent.td_error_abs_mean.append(td_abs_mean)

        # Print progress + metrics at same cadence as original reward logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_length = np.mean(agent.episode_lengths[-100:])

            print(
                f"Episode {episode + 1}/{n_episodes} - "
                f"Avg Reward: {avg_reward:.2f} - "
                f"Avg Length: {avg_length:.2f}"
            )
            print(
                "  Actor L1:  "
                f"grad_norm={actor_L1_grad_avg:.4f}, "
                f"step_norm={actor_L1_step_avg:.4f}, "
                f"rel_step={actor_L1_rel_step_avg:.4e}, "
                f"cos(g, step)={actor_L1_cos_avg:.4f}"
            )
            print(
                "  Critic L1: "
                f"grad_norm={critic_L1_grad_avg:.4f}, "
                f"step_norm={critic_L1_step_avg:.4f}, "
                f"rel_step={critic_L1_rel_step_avg:.4e}, "
                f"cos(g, step)={critic_L1_cos_avg:.4f}"
            )
            print(
                f"  Policy KL (avg per step): {kl_avg:.4e}"
            )
            print(
                f"  TD error: mean={td_mean:.4f}, std={td_std:.4f}, mean|TD|={td_abs_mean:.4f}"
            )

    return agent.episode_rewards, agent.episode_lengths


def plot_l1_stats(agent: ActorCriticAgent, prefix: str = "L1_"):
    """
    Plot and save the tracked L1 stats for actor and critic.
    Creates several figures:
      - <prefix>grad_norms.png
      - <prefix>step_norms.png
      - <prefix>relative_steps.png
      - <prefix>cos_grad_step.png
      - policy_kl.png
      - td_error_stats.png
    """
    episodes = np.arange(1, len(agent.actor_L1_grad_norm_avgs) + 1)

    # Grad norms
    plt.figure()
    plt.plot(episodes, agent.actor_L1_grad_norm_avgs, label="Actor L1 grad norm")
    plt.plot(episodes, agent.critic_L1_grad_norm_avgs, label="Critic L1 grad norm")
    plt.xlabel("Episode")
    plt.ylabel("Grad norm (L2)")
    plt.title("L1 Gradient Norms per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}grad_norms.png")
    plt.close()

    # Step norms
    plt.figure()
    plt.plot(episodes, agent.actor_L1_step_norm_avgs, label="Actor L1 step norm")
    plt.plot(episodes, agent.critic_L1_step_norm_avgs, label="Critic L1 step norm")
    plt.xlabel("Episode")
    plt.ylabel("Step norm (L2 of Δθ)")
    plt.title("L1 Step Norms per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}step_norms.png")
    plt.close()

    # Relative steps (log y-scale)
    plt.figure()
    plt.plot(episodes, agent.actor_L1_rel_step_avgs, label="Actor L1 relative step")
    plt.plot(episodes, agent.critic_L1_rel_step_avgs, label="Critic L1 relative step")
    plt.xlabel("Episode")
    plt.ylabel("Relative step ||Δθ|| / ||θ||")
    plt.title("L1 Relative Step Sizes per Episode")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}relative_steps.png")
    plt.close()

    # NEW: cosine between grad and step
    plt.figure()
    plt.plot(episodes, agent.actor_L1_cos_g_step_avgs, label="Actor L1 cos(g, step)")
    plt.plot(episodes, agent.critic_L1_cos_g_step_avgs, label="Critic L1 cos(g, step)")
    plt.xlabel("Episode")
    plt.ylabel("Cosine similarity")
    plt.title("L1 Grad–Step Alignment per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}cos_grad_step.png")
    plt.close()

    # NEW: policy KL
    if len(agent.policy_kl_avgs) == len(episodes):
        plt.figure()
        plt.plot(episodes, agent.policy_kl_avgs, label="Avg policy KL per step")
        plt.xlabel("Episode")
        plt.ylabel("KL(π_old || π_new)")
        plt.title("Policy KL per Episode")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("policy_kl.png")
        plt.close()

    # NEW: TD error stats
    if (
        len(agent.td_error_mean) == len(episodes)
        and len(agent.td_error_std) == len(episodes)
    ):
        plt.figure()
        plt.plot(episodes, agent.td_error_mean, label="TD error mean")
        plt.plot(episodes, agent.td_error_abs_mean, label="TD error mean |TD|")
        plt.plot(episodes, agent.td_error_std, label="TD error std")
        plt.xlabel("Episode")
        plt.ylabel("TD error")
        plt.title("TD Error Stats per Episode")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("td_error_stats.png")
        plt.close()


def run_actor_critic_demo():
    """Run a demonstration of the Actor-Critic algorithm"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create environment with interesting terminal states
    env = GridWorld(size=9)

    # Set terminal states that don't conflict with starting position (0,0)
    goal_state = env._pos_to_state((4, 6))  # Bottom-right corner
    trap_states = [
        env._pos_to_state((2, 5)),
        env._pos_to_state((2, 4)),
        env._pos_to_state((3, 4)),
        env._pos_to_state((4, 4)),
        env._pos_to_state((5, 4))
    ]

    # Clear default terminal states and set new ones
    env.terminal_states.clear()  # Clear default terminal states
    env.terminal_states[goal_state] = 10.0
    for trap_state in trap_states:
        env.terminal_states[trap_state] = -10.0  # Trap states with negative reward

    # Create Actor-Critic agent with custom hyperparameters
    agent = ActorCriticAgent(
        env,
        learning_rate_actor=0.0003,
        learning_rate_critic=0.001,
        gamma=0.99,
        entropy_coef=0.01,
        optimizer="Muon"  # or "Adam" for comparison
    )

    # Train the agent with extended logging
    print("Training Actor-Critic agent...")
    n_episodes = 8000
    max_steps = 100
    rewards, lengths = train_with_l1_logging(
        agent,
        env,
        n_episodes=n_episodes,
        max_steps=max_steps
    )

    # Plot training results using the existing Visualizer
    training_fig = Visualizer.plot_training_results(
        rewards=rewards,
        lengths=lengths,
        actor_losses=agent.actor_losses,
        critic_losses=agent.critic_losses,
        title='Actor-Critic Training Progress'
    )
    training_fig.savefig('Actor-Critic Training Progress')

    # Plot L1 + KL + TD metrics
    plot_l1_stats(agent, prefix="L1_")

    # Get and display the learned policy
    policy = agent.get_optimal_policy()
    policy_fig = Visualizer.plot_policy(
        policy,
        env.size,
        "Actor-Critic Learned Policy"
    )
    policy_fig.savefig("Actor-Critic Learned Policy")

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

    # Visualize the test episode
    episode_fig = Visualizer.visualize_episode_trajectory(
        env,
        episode_data,
        "Actor-Critic Test Episode Trajectory"
    )
    episode_fig.savefig("Actor-Critic Test Episode Trajectory")

    print(f"Test episode finished with total reward: {total_reward}")

    return agent, policy, total_reward


if __name__ == "__main__":
    agent, policy, final_reward = run_actor_critic_demo()
