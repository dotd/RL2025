import os
import argparse
import numpy as np
from collections import deque
import random
import logging
from datetime import datetime
import time
import math
from PIL import Image
import json
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import gymnasium as gym
import wandb

from RL2025.definitions import PROJECT_ROOT_DIR

try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    # get device from cpu, mps, cuda
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class DQN(nn.Module):
    """Convolutional Deep Q-Network for CartPole using vision input"""

    def __init__(self, action_size, num_frames, debug=False):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(num_frames, 64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Calculate flattened size: after conv layers, we get (64, 7, 7) for 84x84 input
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        # Normalize input to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x / 255.0

        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = torch.relu(self.conv2(x))
        x = self.bn2(x)
        x = torch.relu(self.conv3(x))
        x = self.bn3(x)
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

    def __init__(self, action_size, args):

        self.action_size = action_size

        # Hyperparameters
        self.num_frames = args.num_frames
        self.gamma = args.gamma
        self.epsilon_max = args.epsilon_max
        self.epsilon_min = args.epsilon_min
        self.tau_decay = args.tau_decay
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.target_update_freq = args.target_update_freq
        self.optimization_type = args.optimization_type
        self.frequency_save_check_points = args.frequency_save_check_points  # frequency of saving the model checkpoint

        # parameters
        self.steps_done = 0
        self.epsilon = self.epsilon_max

        # Device and Networks
        self.device = get_device()
        logging.info(f"Device: {self.device}")
        self.policy_net = DQN(action_size, args.num_frames).to(self.device)
        self.target_net = DQN(action_size, args.num_frames).to(self.device)

        # Load network if provided from checkpoint
        if args.network_start_path is not None:
            self.policy_net.load_state_dict(torch.load(args.network_start_path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon_max = self.epsilon_min
            logging.info(f"Network loaded from {args.network_start_path}")
        else:
            logging.info("Starting from scratch")

        # Optimizer
        if self.optimization_type.lower() == "adam".lower():
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        elif self.optimization_type.lower() == "rmsprop".lower():
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Invalid optimization type: {self.optimization_type}")

        # Replay buffer
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

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps_done += 1
        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.steps_done / self.tau_decay)


class PreprocessFrame:
    def __init__(self, resize_pixels):
        self.resize_pixels = resize_pixels

    def cv2(self, frame):
        # Convert to grayscale using weighted average
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        # Resize using cv2 (much faster than PIL)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        resized = cv2.resize(gray, (int(self.resize_pixels * aspect_ratio), self.resize_pixels), interpolation=cv2.INTER_AREA)
        frame_processed = torch.from_numpy(resized).float() / 255.0
        frame_processed = frame_processed.unsqueeze(0)
        return frame_processed

    def torch(self, frame):
        frame_processed = T.Compose([T.ToPILImage(), T.Resize(self.resize_pixels, interpolation=BICUBIC), T.Grayscale(), T.ToTensor()])(frame)
        return frame_processed


def train_agent(args):
    resize = PreprocessFrame(args.resize_pixels).cv2 if args.use_cv2_for_preprocessing else PreprocessFrame(args.resize_pixels).torch
    logging.info("Train the DQN agent on CartPole using vision input")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    action_size = env.action_space.n
    logging.info(f"Action size: {action_size}")

    agent = DQNAgent(action_size, args)
    start_time = time.time()
    logging.info(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    scores = []
    avg_scores = []
    losses = []
    frames = deque(maxlen=args.num_frames)

    logging.info("Training DQN Agent on CartPole-v1 (Vision-based)")
    logging.info(f"Device: {agent.device}")
    logging.info("-" * 50)
    lengths = list()
    avg_lengths = list()

    for episode in range(args.train_episodes):
        state, _ = env.reset()
        # Get initial frame and preprocess
        frame = env.render()
        frame_processed = resize(frame)
        for _ in range(args.num_frames):
            frames.append(frame_processed)
        state = torch.concat(list(frames), dim=0)

        done = False
        episode_losses = []
        reward_vec = list()

        for t in range(args.episode_length):
            # Select and perform action
            action = agent.select_action(state)
            state_variables, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            x, x_dot, theta, theta_dot = state_variables
            r1 = (env.unwrapped.x_threshold - abs(x)) / env.unwrapped.x_threshold - 0.8
            r2 = (env.unwrapped.theta_threshold_radians - abs(theta)) / env.unwrapped.theta_threshold_radians - 0.5
            reward2 = r1 + r2

            # Get next frame and preprocess
            next_frame = env.render()
            next_frame_processed = resize(next_frame)
            frames.append(next_frame_processed)
            next_state = torch.concat(list(frames), dim=0)

            # Store transition
            reward_shaped = reward2
            agent.memory.push(state, action, reward_shaped, next_state, float(done))
            reward_vec.append(reward)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            if done:
                break

        lengths.append(t)
        avg_lengths.append(np.mean(lengths[-100:-1]))
        total_reward = sum(reward_vec)
        wandb.log({"episode_length": t, "epsilon": agent.epsilon, "steps_done": agent.steps_done, "total_reward": total_reward})
        # print(f"Reward vector: {reward_vec}")
        average_time_episode = (time.time() - start_time) / (episode + 1)
        average_time_step = (time.time() - start_time) / (agent.steps_done + 1)
        # Update target network
        if episode % agent.target_update_freq == 0:
            logging.info(f"Updating target network at episode {episode}")
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
        if (episode + 1) % args.frequency_print_progress == 0:
            logging.info(
                f"elapsed: {time.time() - start_time:.2f} sec.|"
                f"len: {t}|"
                f"Episode {episode + 1}/{args.train_episodes}|"
                # f"Score: {total_reward:.0f}|"
                f"Avg Length: {avg_lengths[-1]:.2f}|"
                f"eps: {agent.epsilon:.3f}|"
                f"steps: {agent.steps_done}|"
                f"Avg Ep: {average_time_episode:.2f} sec.|"
                f"Avg Step: {average_time_step:.2f} sec.|"
                f"Time remaining: {average_time_episode * (args.train_episodes - episode - 1):.2f} sec."
            )

        # Save checkpoint
        if episode % args.frequency_save_check_points == 0 or episode == args.train_episodes - 1:
            checkpoint_folder = f"{args.run_folder}/checkpoints/"
            os.makedirs(checkpoint_folder, exist_ok=True)
            torch.save(agent.policy_net.state_dict(), f"{checkpoint_folder}/checkpoint_{episode}.pth")
            logging.info(f"Checkpoint saved to {checkpoint_folder}/checkpoint_{episode}.pth")

        # Check if solved (average score of 195+ over 100 episodes)
        if avg_score >= args.threshold_early_stopping and episode >= args.episodes_early_stopping:
            logging.info(f"Solved in {episode + 1} episodes!")
            logging.info(f"Average Score: {avg_score:.2f}")
            break

    env.close()
    return agent, scores


def setup_wandb(args):
    run = wandb.init(project="cartpole_vision", config=args)
    wandb.log({"config": vars(args)})
    return run


def train_agent_vision(args):
    os.makedirs(args.run_folder, exist_ok=True)
    logging.info("Starting train_agent_vision()")

    logging.info(f"Setting random seeds for reproducibility to {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("Training the agent")
    _ = setup_wandb(args)
    agent, _ = train_agent(args=args)

    logging.info("Saving the model")
    torch.save(agent.policy_net.state_dict(), f"{args.run_folder}/test_model.pth")
    logging.info(f"Model saved to {args.run_folder}/test_model.pth")


def create_loggers(timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"), folder=os.path.dirname(os.path.abspath(__file__)), level=logging.DEBUG):

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    # File handler
    os.makedirs(f"{folder}/logs", exist_ok=True)
    file_handler = logging.FileHandler(f"{folder}/logs/log_{timestamp}.txt")
    file_handler.setLevel(level)

    logging.basicConfig(level=level, format="%(levelname)s> %(message)s", handlers=[file_handler, console_handler])


def process_args():
    create_loggers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_episodes", type=int, default=1000, help="Number of episodes to train the agent")
    parser.add_argument("--target_update_freq", type=int, default=5, help="Frequency of updating the target network")
    parser.add_argument("--epsilon_max", type=float, default=0.9, help="Maximum epsilon-greedy exploration rate. Starting value.")
    parser.add_argument("--epsilon_min", type=float, default=0.01, help="Minimum epsilon-greedy exploration rate. Ending value.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate. How much to update the weights of the network each step.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size. How many samples to use for each training step.")
    parser.add_argument("--buffer_size", type=int, default=10_000, help="Buffer size. How many samples to store in the replay buffer.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor. How much to discount the future rewards.")
    parser.add_argument("--frequency_save_check_points", type=int, default=100, help="Frequency of saving the model checkpoint.")
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"{PROJECT_ROOT_DIR}/data/cartpole_vision/run_{current_timestamp}/"
    parser.add_argument("--run_folder", type=str, default=run_folder, help="Folder to save the model.")
    parser.add_argument(
        "--threshold_early_stopping",
        type=int,
        default=400,
        help="Threshold for early stopping.",
    )
    help_early_stopping = "Threshold for early stopping. If the average score is above this threshold, stop training."
    parser.add_argument("--episodes_early_stopping", type=int, default=100, help=help_early_stopping)
    parser.add_argument("--num_frames", type=int, default=3, help="Number of frames to stack. How many frames to stack to get the state.")
    parser.add_argument("--optimization_type", type=str, default="adam", help="Optimization type. Adam, SGD.")
    # ready_model_path = f"/Users/dotan/projects/RL2025/data/cartpole_vision/run_20251203_111322/checkpoints/checkpoint_580.pth"
    parser.add_argument("--network_start_path", type=str, default=None, help="Path to the network to start from.")
    parser.add_argument("--tau_decay", type=float, default=2000, help="Tau decay. How much to decay the epsilon-greedy exploration rate.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training. cpu, mps, cuda.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for random number generator.")
    parser.add_argument("--frequency_print_progress", type=int, default=1, help="Frequency of printing progress.")
    parser.add_argument("--episode_length", type=int, default=200, help="")
    parser.add_argument("--resize_pixels", type=int, default=60, help="Number of pixels for downsampling.")
    parser.add_argument(
        "--use_cv2_for_preprocessing", type=bool, default=False, help="Use cv2 for preprocessing. If True, use cv2 to preprocess the frame. If False, use torch to preprocess the frame."
    )
    args = parser.parse_args()
    # log args to logging
    logging.info(f"Args:\n{json.dumps(vars(args), indent=2)}")
    return args


def main():
    args = process_args()
    train_agent_vision(args)


if __name__ == "__main__":
    main()
