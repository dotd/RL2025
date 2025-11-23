import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

import cv2
import ale_py
import gymnasium as gym

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


gym.register_envs(ale_py)


# Define the convolutional DQN network for Breakout
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # Input will be stack of 4 frames, shape (4, 84, 84)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, x):
        x = x / 255.0  # normalize input
        return self.net(x)


def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 110))
    obs = obs[18:102, :]  # crop to 84x84
    return obs


# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# Epsilon-greedy action selection
def select_action(state, policy_net, epsilon, action_size, device="cpu"):
    if random.random() < epsilon:
        return random.randrange(action_size)
    else:
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.argmax(1).item()


loss_value = float("nan")


# Main training loop
def train():
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    action_size = env.action_space.n
    logger.info(f"action_size: {action_size}")
    logger.info(f"env: {env}")
    logger.info(f"env.action_space: {env.action_space}")
    logger.info(f"env.observation_space: {env.observation_space}")
    logger.info(f"env.reward_range: {env.reward_range if 'reward_range' in env.__dict__ else 'None'}")
    logger.info(f"env.metadata: {env.metadata if 'metadata' in env.__dict__ else 'None'}")
    logger.info(f"env.spec: {env.spec if 'spec' in env.__dict__ else 'None'}")
    logger.info(f"env.unwrapped: {env.unwrapped if 'unwrapped' in env.__dict__ else 'None'}")
    logger.info(f"env.game_path: {env.game_path if 'game_path' in env.__dict__ else 'None'}")
    # Hyperparameters
    num_episodes = 1000
    gamma = 0.99
    buffer_capacity = 100_000
    batch_size = 32
    lr = 1e-4
    update_target_every = 1000
    start_learning = 10_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 1_000_000

    # Networks and optimizer
    policy_net = DQN(action_size).to(device)
    target_net = DQN(action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start
    total_steps = 0
    time_training = float("nan")

    for episode in range(num_episodes):
        steps_done = 0
        obs, _ = env.reset()
        state = preprocess(obs)
        state_stack = np.stack([state] * 4, axis=0)  # shape (4, 84, 84)
        episode_reward = 0
        done = False
        while not done:
            epsilon = max(epsilon_final, epsilon_start - steps_done / epsilon_decay)
            action = select_action(state_stack.astype(np.float32), policy_net, epsilon, action_size, device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess(next_obs)
            next_state_stack = np.roll(state_stack, -1, axis=0)
            next_state_stack[-1] = next_state
            replay_buffer.push(state_stack, action, reward, next_state_stack, done)
            state_stack = next_state_stack
            episode_reward += reward
            steps_done += 1
            total_steps += 1

            # Training step
            if len(replay_buffer) > start_learning:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                    target = rewards + gamma * (1 - dones) * max_next_q
                loss = nn.functional.mse_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                loss_value = loss.item()
                optimizer.step()

                # Update target network
                if steps_done % update_target_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            else:
                if total_steps % 100 == 0:
                    logger.info(f"Not training yet, len replay buffer: {len(replay_buffer)}")

        if total_steps % 100 == 0:
            logger.info(f"Total steps: {total_steps}")
            logger.info(f"Episode {episode} Reward:{episode_reward} steps_done:{steps_done} epsilon:{epsilon}  lenRB: {len(replay_buffer)}")
    env.close()


if __name__ == "__main__":
    train()
