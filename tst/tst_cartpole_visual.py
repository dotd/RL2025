import os
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import cv2
import logging
from RL2025.definitions import PROJECT_ROOT_DIR


def preprocess_frame(frame, debug=False, debug_folder=f"{PROJECT_ROOT_DIR}/data/cartpole/"):
    """Preprocess frame: convert to grayscale and resize to 84x84"""
    if debug:
        # save frame as image
        cv2.imwrite(f"{debug_folder}/frame_original.png", frame)
    # frame is RGB image (height, width, 3)
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Ensure uint8 dtype for consistent normalization
    if debug:
        cv2.imwrite(f"{debug_folder}/frame_resized.png", resized)
    return resized.astype(np.uint8)


class DQN(nn.Module):
    """Convolutional Deep Q-Network for CartPole using vision input"""

    def __init__(self, action_size, num_frames=1, debug=False):
        super(DQN, self).__init__()
        # Input shape: (num_frames, 84, 84) - grayscale images
        self.conv1 = nn.Conv2d(num_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate flattened size: after conv layers, we get (64, 7, 7) for 84x84 input
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        # Normalize input to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x / 255.0

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    """Experience replay buffer for storing transitions"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for CartPole using vision input"""

    def __init__(
        self,
        action_size,
        ender_frequency,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        learning_rate,
        batch_size,
        buffer_size,
        num_frames,
    ):

        self.action_size = action_size
        self.num_frames = num_frames

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = ender_frequency

        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(action_size, num_frames).to(self.device)
        self.target_net = DQN(action_size, num_frames).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(self.buffer_size)

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            # state is preprocessed frame (84, 84) or stack of frames
            if len(state.shape) == 2:
                # Single frame: add channel and batch dimensions (1, 1, 84, 84)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            elif len(state.shape) == 3:
                # Stack of frames: add batch dimension (1, num_frames, 84, 84)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                # Already batched
                state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors - states are preprocessed frames (84, 84)
        # np.array(states) gives shape (batch_size, 84, 84)
        # Add channel dimension: (batch_size, 1, 84, 84) for CNN
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        policy_net = self.policy_net(states)
        current_q_values = policy_net.gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            target_net = self.target_net(next_states)
            next_q_values = target_net.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # logging.info(f"states: {states.shape}")
        # logging.info(f"next_states: {next_states.shape}")
        # logging.info(f"actions: {actions.shape}")
        # logging.info(f"rewards: {rewards.shape}")
        # logging.info(f"dones: {dones.shape}")
        # logging.info(f"current_q_values: {current_q_values.shape}")
        # logging.info(f"target_q_values: {target_q_values.shape}")

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(
    episodes,
    ender_frequency,
    gamma,
    epsilon,
    epsilon_min,
    epsilon_decay,
    learning_rate,
    batch_size,
    buffer_size,
    threshold_early_stopping,
    episodes_early_stopping,
    num_frames,
):
    logging.info("Train the DQN agent on CartPole using vision input")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    action_size = env.action_space.n

    agent = DQNAgent(action_size, ender_frequency, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, batch_size, buffer_size, num_frames)

    scores = []
    avg_scores = []
    losses = []
    frames = deque(maxlen=num_frames)

    logging.info("Training DQN Agent on CartPole-v1 (Vision-based)")
    logging.info(f"Device: {agent.device}")
    logging.info("-" * 50)

    for episode in range(episodes):
        _, _ = env.reset()
        # Get initial frame and preprocess
        frame = env.render()
        frame_processed = preprocess_frame(frame)
        for _ in range(num_frames):
            frames.append(frame_processed)
        state = np.array(frames)

        total_reward = 0
        done = False
        episode_losses = []

        while not done:
            # Select and perform action
            action = agent.select_action(state)
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Get next frame and preprocess
            next_frame = env.render()
            next_frame_processed = preprocess_frame(next_frame)
            frames.append(next_frame_processed)
            next_state = np.array(frames)

            # Store transition
            agent.memory.push(state, action, reward, next_state, float(done))

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        # Update target network
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        # Record statistics
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        if episode_losses:
            losses.append(np.mean(episode_losses))

        # Print progress
        if (episode + 1) % 20 == 0:
            logging.info(
                f"Episode {episode + 1}/{episodes} | "
                f"Score: {total_reward:.0f} | "
                f"Avg Score: {avg_score:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        # Check if solved (average score of 195+ over 100 episodes)
        if avg_score >= threshold_early_stopping and episode >= episodes_early_stopping:
            logging.info(f"Solved in {episode + 1} episodes!")
            logging.info(f"Average Score: {avg_score:.2f}")
            break

    env.close()

    # Plot results
    plot_results(scores, avg_scores, losses)

    return agent, scores


def plot_results(scores, avg_scores, losses):
    logging.info("Plot training results")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot scores
    ax1.plot(scores, alpha=0.3, label="Episode Score")
    ax1.plot(avg_scores, linewidth=2, label="Average Score (100 episodes)")
    ax1.axhline(y=195, color="r", linestyle="--", label="Solved Threshold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.set_title("Training Scores")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot losses
    if losses:
        ax2.plot(losses, linewidth=1)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Loss")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PROJECT_ROOT_DIR}/data/cartpole/cartpole_training.png", dpi=150, bbox_inches="tight")
    logging.info(f"Training plot saved to {PROJECT_ROOT_DIR}/data/cartpole/cartpole_training.png")


def test_agent(agent, num_episodes=10, render=False):
    """Test the trained agent"""
    render_mode = "human" if render else "rgb_array"
    env = gym.make("CartPole-v1", render_mode=render_mode)

    logging.info("Testing trained agent...")
    logging.info("-" * 50)

    test_scores = []

    for episode in range(num_episodes):
        _, _ = env.reset()
        # Get initial frame and preprocess
        frame = env.render()
        state = preprocess_frame(frame) if frame is not None else None

        total_reward = 0
        done = False

        while not done:
            if state is None:
                # If render mode is None, we can't get frames
                logging.warning("Cannot test vision-based agent without render mode")
                break

            action = agent.select_action(state, training=False)
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Get next frame
            frame = env.render()
            if frame is not None:
                state = preprocess_frame(frame)

            total_reward += reward

        test_scores.append(total_reward)
        logging.info(f"Test Episode {episode + 1}: Score = {total_reward}")

    env.close()

    logging.info(f"Average Test Score: {np.mean(test_scores):.2f}")
    logging.info(f"Max Test Score: {np.max(test_scores):.0f}")
    logging.info(f"Min Test Score: {np.min(test_scores):.0f}")

    return test_scores


def record_video(agent, model_path=None, output_path="cartpole_agent_video.mp4", fps=30):
    """Record a video of the trained agent balancing the pole"""

    # Load model if path is provided
    if model_path:
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
        logging.info(f"Loaded model from {model_path}")

    # Ensure network is in eval mode for inference
    agent.policy_net.eval()

    # Create environment with rgb_array render mode for video recording
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    logging.info("Recording video of agent playing...")
    logging.info("-" * 50)

    # Collect frames from a successful episode
    frames = []
    total_reward = 0
    done = False
    max_steps = 500  # Maximum steps for CartPole-v1
    step_count = 0

    # Run until we get a successful episode (500 steps) or maximum attempts
    for attempt in range(10):
        _, _ = env.reset()
        frames = []
        total_reward = 0
        done = False
        step_count = 0

        # Get initial frame and preprocess
        frame = env.render()
        state = preprocess_frame(frame)

        while not done and step_count < max_steps:
            # Render and save frame (before action to show state)
            frame = env.render()
            frames.append(frame.copy())

            # Select action using preprocessed state
            action = agent.select_action(state, training=False)
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Get next frame and preprocess for next iteration
            next_frame = env.render()
            if next_frame is not None:
                state = preprocess_frame(next_frame)

            total_reward += reward
            step_count += 1

        # Add final frame if we reached max steps (successful episode)
        if step_count >= max_steps:
            frame = env.render()
            if frame is not None:
                frames.append(frame.copy())

        # If we reached max steps, this is a successful episode
        if step_count >= max_steps or total_reward >= 500:
            logging.info(f"Recorded successful episode: {step_count} steps, {total_reward:.0f} reward")
            break
        else:
            logging.info(f"Attempt {attempt + 1}: Episode ended early ({step_count} steps), trying again...")

    env.close()

    if not frames:
        logging.error("Error: No frames collected!")
        return

    # Save frames as video
    logging.info(f"Saving {len(frames)} frames to {output_path}...")

    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
    video.release()

    logging.info(f"Video saved successfully to {output_path}!")
    logging.info(f"Episode length: {len(frames)} frames ({len(frames)/fps:.2f} seconds)")

    return output_path


def main(args):
    model_folder = f"{PROJECT_ROOT_DIR}/data/cartpole/"
    os.makedirs(model_folder, exist_ok=True)
    video_folder = f"{PROJECT_ROOT_DIR}/data/cartpole/"
    os.makedirs(video_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s> %(message)s")
    logging.info("Starting the program")

    logging.info("Setting random seeds for reproducibility")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    logging.info("Training the agent")
    agent, scores = train_agent(
        episodes=args.train_episodes,
        ender_frequency=args.ender_frequency,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        threshold_early_stopping=args.threshold_early_stopping,
        episodes_early_stopping=args.episodes_early_stopping,
        num_frames=args.num_frames,
    )

    logging.info("Testing the agent")
    test_scores = test_agent(agent, num_episodes=args.test_episodes)
    logging.info(f"Test scores: {test_scores}")

    logging.info("Saving the model")
    torch.save(agent.policy_net.state_dict(), f"{model_folder}/cartpole_dqn_model.pth")
    logging.info(f"Model saved to {model_folder}/cartpole_dqn_model.pth")

    # Record a video of the agent playing
    logging.info("Recording a video of the agent playing")
    record_video(agent, model_path=None, output_path=f"{video_folder}/cartpole_agent_video.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_episodes", type=int, default=5000)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--ender_frequency", type=int, default=100)
    parser.add_argument("--test_episodes", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--epsilon_min", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.999)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--threshold_early_stopping", type=int, default=450)
    parser.add_argument("--episodes_early_stopping", type=int, default=200)
    parser.add_argument("--num_frames", type=int, default=2)
    args = parser.parse_args()
    main(args)
