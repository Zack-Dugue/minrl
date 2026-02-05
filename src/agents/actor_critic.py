import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_   # NEW
from torch.optim.lr_scheduler import CosineAnnealingLR  # NEW

import numpy as np
from typing import List, Tuple, Dict, Optional

from ..utils.optimizers import *
from ..environment.grid_world import GridWorld, Action


class ActorNetwork(nn.Module):
    """
    Actor network that outputs action probabilities.
    Uses the same architecture as PPO for consistency.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorNetwork, self).__init__()
        self.L0 = nn.Linear(state_dim, hidden_dim)
        self.act = nn.GELU()
        self.L1 = nn.Linear(hidden_dim, hidden_dim)
        self.L2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        x = self.L0(x)
        x = self.act(x)
        x = self.L1(x)
        x = self.act(x)
        logits = self.L2(x)
        return torch.distributions.Categorical(logits=logits)

    def get_split_params(self):
        adam_parameters = [p for p in self.L0.parameters()] + [p for p in self.L2.parameters()]
        adam_parameters.append(self.L1.bias)
        muon_paramters = [self.L1.weight]
        return muon_paramters, adam_parameters


class CriticNetwork(nn.Module):
    """
    Critic network that estimates state values.
    Uses the same architecture as PPO for consistency.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(CriticNetwork, self).__init__()
        self.L0 = nn.Linear(state_dim, hidden_dim)
        self.act = nn.GELU()
        self.L1 = nn.Linear(hidden_dim, hidden_dim)
        self.L2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.L0(x)
        x = self.act(x)
        x = self.L1(x)
        x = self.act(x)
        value = self.L2(x)
        return value

    def get_split_params(self):
        adam_parameters = [p for p in self.L0.parameters()] + [p for p in self.L2.parameters()]
        adam_parameters.append(self.L1.bias)
        muon_paramters = [self.L1.weight]
        return muon_paramters, adam_parameters


class ActorCriticAgent:
    """
    Actor-Critic agent implementation using separate actor and critic networks.
    Features TD(0) learning and policy gradient updates.
    Now includes:
      • Gradient clipping
      • Cosine LR schedulers
    """

    def __init__(self,
                 env: GridWorld,
                 learning_rate_actor: float = 0.001,
                 learning_rate_critic: float = 0.003,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 momentum: float = 0.95,
                 optimizer: str = "Adam",
                 gradient_clip_norm: Optional[float] = 0.5,  # NEW: max grad norm (None to disable)
                 use_cosine_scheduler: bool = True           # NEW
                 ):
        """
        Initialize the Actor-Critic agent.

        Args:
            env: The GridWorld environment
            learning_rate_actor: Learning rate for the actor
            learning_rate_critic: Learning rate for the critic
            gamma: Discount factor
            entropy_coef: Entropy bonus coefficient
            momentum: Momentum / beta1 for optimizers
            optimizer: "SGD" | "Adam" | "Muon" | "AdaMuon" | "NorMuon"
            gradient_clip_norm: If not None, clip grad norm to this value
            use_cosine_scheduler: If True, use CosineAnnealingLR per episode
        """
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.gradient_clip_norm = gradient_clip_norm      # NEW
        self.use_cosine_scheduler = use_cosine_scheduler  # NEW

        # Initialize networks
        self.state_dim = env.size * env.size
        self.action_dim = env.get_action_space_size()

        self.actor = ActorNetwork(self.state_dim, self.action_dim)
        self.critic = CriticNetwork(self.state_dim)

        # Initialize optimizers
        print(f"Initializing AC aglorithm with optimizer: {optimizer}")
        if optimizer == "SGD":
            self.actor_optimizer = optim.SGD(
                self.actor.parameters(), momentum=momentum, lr=learning_rate_actor
            )
            self.critic_optimizer = optim.SGD(
                self.critic.parameters(), momentum=momentum, lr=learning_rate_critic
            )

        elif optimizer == "Adam":
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), betas=(momentum, 0.999), lr=learning_rate_actor
            )
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), betas=(momentum, 0.999), lr=learning_rate_critic
            )

        elif optimizer == "Muon":
            muon_params, aux_params = self.actor.get_split_params()
            ns_steps = 5
            param_groups = [
                dict(params=muon_params, lr=learning_rate_actor, momentum=momentum,
                     weight_decay=1e-4, use_muon=True, ns_steps=ns_steps),
                dict(params=aux_params, lr=learning_rate_actor / 66.0, momentum=momentum,
                     weight_decay=1e-4, use_muon=False),
            ]
            self.actor_optimizer = MuonWithAuxAdam(param_groups)

            muon_params, aux_params = self.critic.get_split_params()
            param_groups = [
                dict(params=muon_params, lr=learning_rate_critic, momentum=momentum,
                     weight_decay=1e-4, use_muon=True, ns_steps=ns_steps),
                dict(params=aux_params, lr=learning_rate_critic / 66.0, momentum=momentum,
                     weight_decay=1e-4, use_muon=False),
            ]
            self.critic_optimizer = MuonWithAuxAdam(param_groups)

        elif optimizer == "AdaMuon":
            muon_params, aux_params = self.actor.get_split_params()
            ns_steps = 5
            param_groups = [
                dict(params=muon_params, lr=learning_rate_actor, momentum=momentum,
                     weight_decay=1e-4, use_muon=True, ns_steps=ns_steps),
                dict(params=aux_params, lr=learning_rate_actor, momentum=momentum,
                     weight_decay=1e-4, use_muon=False),
            ]
            self.actor_optimizer = AdaMuonWithAuxAdam(param_groups)

            muon_params, aux_params = self.critic.get_split_params()
            param_groups = [
                dict(params=muon_params, lr=learning_rate_critic, momentum=momentum,
                     weight_decay=1e-4, use_muon=True, ns_steps=ns_steps),
                dict(params=aux_params, lr=learning_rate_critic, momentum=momentum,
                     weight_decay=1e-4, use_muon=False),
            ]
            self.critic_optimizer = AdaMuonWithAuxAdam(param_groups)

        elif optimizer == "NorMuon":
            muon_params, aux_params = self.actor.get_split_params()
            ns_steps = 5
            param_groups = [
                dict(params=muon_params, lr=learning_rate_actor, momentum=momentum,
                     weight_decay=1e-4, use_muon=True, ns_steps=ns_steps),
                dict(params=aux_params, lr=learning_rate_actor, momentum=momentum,
                     weight_decay=1e-4, use_muon=False),
            ]
            self.actor_optimizer = SingleDeviceNorMuonWithAuxAdam(param_groups)

            muon_params, aux_params = self.critic.get_split_params()
            param_groups = [
                dict(params=muon_params, lr=learning_rate_critic, momentum=momentum,
                     weight_decay=1e-4, use_muon=True, ns_steps=ns_steps),
                dict(params=aux_params, lr=learning_rate_critic, momentum=momentum,
                     weight_decay=1e-4, use_muon=False),
            ]
            self.critic_optimizer = AdaMuonWithAuxAdam(param_groups)

        elif optimizer == "BGD":
            print("BGD NOT YET IMPLEMENTED")
            raise ValueError("BGD NOT YET IMPLEMENTED")
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Schedulers (created later in train() once n_episodes is known)
        self.actor_scheduler: Optional[CosineAnnealingLR] = None   # NEW
        self.critic_scheduler: Optional[CosineAnnealingLR] = None  # NEW

        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.entropy_losses: List[float] = []

    def state_to_tensor(self, state: int) -> torch.Tensor:
        """Convert state number to one-hot tensor"""
        state_tensor = torch.zeros(self.state_dim)
        state_tensor[state] = 1.0
        return state_tensor

    def select_action(self, state: int) -> Tuple[Action, torch.Tensor, float]:
        """
        Select action using the current policy.

        Returns:
            (action, log_prob, value_estimate)
        """
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)

            # Get action distribution from actor
            dist = self.actor(state_tensor)

            # Get value estimate from critic
            value = self.critic(state_tensor).item()

            # Mask invalid actions
            valid_actions = self.env.get_valid_actions(state)
            if not valid_actions:
                return Action(0), torch.tensor(0.0), value

            mask = torch.zeros(self.action_dim)
            mask[list(valid_actions)] = 1

            masked_probs = dist.probs * mask
            masked_sum = masked_probs.sum()

            if masked_sum > 0:
                masked_probs = masked_probs / masked_sum
            else:
                masked_probs = mask / len(valid_actions)

            masked_dist = torch.distributions.Categorical(probs=masked_probs)

            action = masked_dist.sample()
            log_prob = masked_dist.log_prob(action)

            return Action(action.item()), log_prob, value

    def update(self, state: int, action: Action, reward: float,
               next_state: int, done: bool) -> Tuple[float, float, float]:
        """
        Update policy and value functions using TD(0) learning.

        Returns:
            (actor_loss, critic_loss, entropy_loss)
        """
        # Convert states to tensors
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)

        # ---- Critic update ----
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)

        td_target = reward + (1 - int(done)) * self.gamma * next_value.detach()
        td_error = td_target - value

        critic_loss = td_error.pow(2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # NEW: gradient clipping for critic
        if self.gradient_clip_norm is not None:
            clip_grad_norm_(self.critic.parameters(), self.gradient_clip_norm)

        self.critic_optimizer.step()

        # ---- Actor update ----
        dist = self.actor(state_tensor)
        log_prob = dist.log_prob(torch.tensor(action))

        actor_loss = -log_prob * td_error.detach()
        entropy_loss = -dist.entropy()
        total_actor_loss = actor_loss + self.entropy_coef * entropy_loss

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()

        # NEW: gradient clipping for actor
        if self.gradient_clip_norm is not None:
            clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)

        self.actor_optimizer.step()

        return (actor_loss.item(),
                critic_loss.item(),
                entropy_loss.item())

    def train(self, n_episodes: int = 1000, max_steps: int = 100) -> Tuple[List[float], List[int]]:
        """
        Train the agent using Actor-Critic.

        Cosine LR schedulers are stepped once per episode (if enabled).
        """
        # Create / reset cosine schedulers with T_max = n_episodes
        if self.use_cosine_scheduler:
            self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=n_episodes)
            self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=n_episodes)

        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            steps = 0

            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            entropy_loss_sum = 0.0

            for step in range(max_steps):
                action, _, _ = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                actor_loss, critic_loss, entropy_loss = self.update(
                    state, action, reward, next_state, done
                )

                actor_loss_sum += actor_loss
                critic_loss_sum += critic_loss
                entropy_loss_sum += entropy_loss

                episode_reward += reward
                steps += 1

                if done:
                    break

                state = next_state

            # Record episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.actor_losses.append(actor_loss_sum / max(steps, 1))
            self.critic_losses.append(critic_loss_sum / max(steps, 1))
            self.entropy_losses.append(entropy_loss_sum / max(steps, 1))

            # Step schedulers once per episode (NEW)
            if self.use_cosine_scheduler:
                self.actor_scheduler.step()
                self.critic_scheduler.step()

            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(
                    f"Episode {episode + 1}/{n_episodes} - "
                    f"Avg Reward: {avg_reward:.2f} - "
                    f"Avg Length: {avg_length:.2f}"
                )

        return self.episode_rewards, self.episode_lengths

    def get_optimal_policy(self) -> Dict[int, List[float]]:
        """
        Extract the optimal policy from the actor network.
        """
        policy = {}

        with torch.no_grad():
            for state in range(self.env.get_state_space_size()):
                state_tensor = self.state_to_tensor(state)

                if state in self.env.terminal_states:
                    policy[state] = [1.0 / self.action_dim] * self.action_dim
                else:
                    dist = self.actor(state_tensor)

                    valid_actions = self.env.get_valid_actions(state)
                    mask = torch.zeros_like(dist.probs)
                    mask[list(valid_actions)] = 1
                    masked_probs = dist.probs * mask

                    masked_sum = masked_probs.sum()
                    if masked_sum > 0:
                        masked_probs = masked_probs / masked_sum
                    else:
                        mask = torch.zeros_like(dist.probs)
                        mask[list(valid_actions)] = 1.0 / len(valid_actions)
                        masked_probs = mask

                    policy[state] = masked_probs.tolist()

        return policy

    def save(self, path: str) -> None:
        """
        Save the agent's networks to disk.
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropy_losses': self.entropy_losses,
        }, path)

    def load(self, path: str) -> None:
        """
        Load the agent's networks from disk.
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        self.entropy_losses = checkpoint['entropy_losses']

    def evaluate(self, n_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate the agent's performance without training.
        """
        eval_rewards = []
        eval_lengths = []

        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            steps = 0
            done = False

            while not done and steps < 1000:
                with torch.no_grad():
                    action, _, _ = self.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    steps += 1
                    state = next_state

            eval_rewards.append(episode_reward)
            eval_lengths.append(steps)

        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)

        return avg_reward, avg_length
