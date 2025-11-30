import os
import argparse
import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import deque
import random
import matplotlib.pyplot as plt
import cv2
import logging
from datetime import datetime
import time
import math
from itertools import count
from PIL import Image

import wandb

from RL2025.definitions import PROJECT_ROOT_DIR

try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC


def get_device():
    # get device from cpu, mps, cuda
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


"""
def preprocess_frame(frame, debug=False, debug_folder=f"{PROJECT_ROOT_DIR}/data/cartpole/"):
    '''Preprocess frame: convert to grayscale and resize to 84x84'''
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
"""


class DQN(nn.Module):
    """Convolutional Deep Q-Network for CartPole using vision input"""

    def __init__(self, action_size, num_frames, debug=False):
        super(DQN, self).__init__()
        # Input shape: (num_frames, 84, 84) - grayscale images
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
        # self.epsilon_decay = args.epsilon_decay
        self.tau_decay = args.tau_decay
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.target_update_freq = args.target_update_freq
        self.optimization_type = args.optimization_type
        self.scheduler_type = args.scheduler_type
        self.frequency_save_check_points = args.frequency_save_check_points  # frequency of saving the model checkpoint

        # parameters
        self.steps_done = 0
        self.epsilon = self.epsilon_max

        # Device and Networks
        self.device = get_device()
        self.policy_net = DQN(action_size, args.num_frames).to(self.device)
        self.target_net = DQN(action_size, args.num_frames).to(self.device)

        # Load network if provided from checkpoint
        if args.network_start_path is not None:
            self.policy_net.load_state_dict(torch.load(args.network_start_path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
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


def preprocess_frame(frame, resize, state):
    # frame = frame.transpose((2, 0, 1))
    frame = resize(frame)
    return frame


def tst_transform_env_frame(args):
    # general
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    resize = T.Compose([T.ToPILImage(), T.Resize(args.RESIZE_PIXELS, interpolation=BICUBIC), T.Grayscale(), T.ToTensor()])

    # get the frame
    state, _ = env.reset()
    logging.info(f"state shape: {state.shape}")
    frame = env.render()
    show_and_save_image(frame)
    logging.info(f"frame shape: {frame.shape}")
    # frame_processed = preprocess_frame(frame, resize, state)
    frame_processed = resize(frame)
    show_and_save_image(frame_processed)
    return frame_processed


# Cart location for centering image crop
def get_cart_location(screen_width, cart_position, env):
    """
    Calculate cart location on screen.

    Args:
        screen_width: Width of the screen
        cart_position: Current cart position from observation[0].
                      If None, tries to access from env (may not work in Gymnasium)
    """
    world_width = env.unwrapped.x_threshold * 2
    scale = screen_width / world_width

    # Try to get cart position from parameter or environment
    if cart_position is None:
        # Try to access from environment (may not work in all Gymnasium versions)
        try:
            cart_pos = env.unwrapped.state[0]
        except (AttributeError, KeyError):
            # Fallback: use center of screen if state is not accessible
            logging.warning("Cannot access cart position from env.unwrapped.state, using screen center")
            return screen_width // 2
    else:
        cart_pos = cart_position

    return int(cart_pos * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(
    frame,  ## frame of H x W x 3
    state,  ## state location, angle, velocity, angular velocity
    resize,  ## resize function
    device,  ## cpu, mps, cuda
):
    screen = frame.transpose((2, 0, 1))  # Transpose it into torch order (CHW).
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)

    # Get cart position from observation if available
    cart_position = state[0] if state is not None else None
    cart_location = get_cart_location(screen_width, cart_position, env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


def show_and_save_image(image, path=None):
    if path is None:
        # get current folder and create it if not exists
        os.makedirs(f"{os.path.dirname(os.path.abspath(__file__))}/frames", exist_ok=True)
        path = f"{os.path.dirname(os.path.abspath(__file__))}/frames/frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # if image is a numpy array, convert to PIL image
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        # no matter is this is grey scale image or rgb image, convert to rgb image
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        # change the shape to (H, W, 3)
        image = image.transpose((1, 2, 0))
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError(f"Invalid image type: {type(image)}")
    logging.info(f"Showing and saving image to {path}")
    logging.info(f"image shape: {image.shape}")
    # do new figure and plot the image
    plt.figure()
    plt.imshow(image)
    plt.savefig(path)
    plt.show(block=False)
    plt.close()
    return


def tst_transform_env_frame2(run_folder=None):
    # general
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    resize = T.Compose([T.ToPILImage(), T.Resize(args.resize_pixels, interpolation=BICUBIC), T.Grayscale(), T.ToTensor()])

    # get the frame
    state, _ = env.reset()

    logging.info(f"state shape: {state.shape}")
    frame = env.render()
    show_and_save_image(frame, f"{run_folder}/frame.png")

    logging.info(f"frame shape: {frame.shape}")
    frame_processed = get_screen(frame, state, resize, device)
    return frame_processed


def train_agent(args):
    resize = T.Compose([T.ToPILImage(), T.Resize(args.resize_pixels, interpolation=BICUBIC), T.Grayscale(), T.ToTensor()])
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

    for episode in range(args.train_episodes):
        state, _ = env.reset()
        # Get initial frame and preprocess
        frame = env.render()
        frame_processed = resize(frame)
        for _ in range(args.num_frames):
            frames.append(frame_processed)
        state = torch.concat(list(frames), dim=0)

        total_reward = 0
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
            agent.memory.push(state, action, reward2, next_state, float(done))
            reward_vec.append(reward)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward
            if done:
                break

        wandb.log({"score": total_reward, "epsilon": agent.epsilon, "steps_done": agent.steps_done})
        # print(f"Reward vector: {reward_vec}")
        average_time = (time.time() - start_time) / (episode + 1)
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
                f"Episode {episode + 1}/{args.train_episodes} | "
                f"Score: {total_reward:.0f} | "
                f"Avg Score: {avg_score:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"steps_done: {agent.steps_done} | "
                f"Average time per episode: {average_time:.2f} seconds | "
                f"Time remaining: {average_time * (args.train_episodes - episode - 1):.2f} seconds"
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

    # Plot results
    plot_results(args.run_folder, scores, avg_scores, losses)

    return agent, scores


def plot_results(run_folder, scores, avg_scores, losses):
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
    plt.savefig(f"{run_folder}/cartpole_training.png", dpi=150, bbox_inches="tight")
    logging.info(f"Training plot saved to {run_folder}/cartpole_training.png")


def setup_wandb(args):
    run = wandb.init(project="cartpole_vision", config=args)
    wandb.log({"config": vars(args)})
    return run


def train_agent_vision(args):
    os.makedirs(args.run_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s> %(message)s")
    logging.info("Starting train_agent_vision()")

    logging.info(f"Setting random seeds for reproducibility to {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("Training the agent")
    run = setup_wandb(args)
    agent, _ = train_agent(args=args)

    # logging.info("Testing the agent")
    # test_scores = test_agent(run_folder, agent, num_episodes=args.test_episodes)
    # logging.info(f"Test scores: {test_scores}")

    logging.info("Saving the model")
    torch.save(agent.policy_net.state_dict(), f"{run_folder}/test_model.pth")
    logging.info(f"Model saved to {run_folder}/test_model.pth")

    # Record a video of the agent playing
    # logging.info("Recording a video of the agent playing")
    # record_video(agent, model_path=None, output_path=f"{run_folder}/cartpole_agent_video.mp4")


def main(args):
    # tst_transform_env_frame(args)
    train_agent_vision(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_episodes", type=int, default=50000, help="Number of episodes to train the agent")
    parser.add_argument("--target_update_freq", type=int, default=5, help="Frequency of updating the target network")
    parser.add_argument("--test_episodes", type=int, default=100, help="Number of episodes to test the agent")
    parser.add_argument("--epsilon", type=float, default=0.9, help="Epsilon-greedy exploration rate. Starting value.")
    parser.add_argument("--epsilon_max", type=float, default=0.9, help="Maximum epsilon-greedy exploration rate. Starting value.")
    parser.add_argument("--epsilon_min", type=float, default=0.01, help="Minimum epsilon-greedy exploration rate. Ending value.")
    # parser.add_argument("--epsilon_decay", type=float, default=0.99, help="Epsilon-greedy exploration rate exponential decay.")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate. How much to update the weights of the network each step.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size. How many samples to use for each training step.")
    parser.add_argument("--buffer_size", type=int, default=10_000, help="Buffer size. How many samples to store in the replay buffer.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor. How much to discount the future rewards.")
    parser.add_argument("--frequency_save_check_points", type=int, default=20, help="Frequency of saving the model checkpoint.")
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
    parser.add_argument("--num_frames", type=int, default=2, help="Number of frames to stack. How many frames to stack to get the state.")
    parser.add_argument("--optimization_type", type=str, default="rmsprop", help="Optimization type. Adam, SGD.")
    help_scheduler_type = "Scheduler type. None, CosineAnnealingWarmRestarts, ReduceLROnPlateau."
    parser.add_argument("--scheduler_type", type=str, default="None", help=help_scheduler_type)
    ready_model_path = f"{PROJECT_ROOT_DIR}/data/cartpole_vision/chech_points/cartpole_dqn_model_checkpoint_episode_4980.pth"
    parser.add_argument("--network_start_path", type=str, default=None, help="Path to the network to start from.")
    parser.add_argument("--tau_decay", type=float, default=2000, help="Tau decay. How much to decay the epsilon-greedy exploration rate.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training. cpu, mps, cuda.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for random number generator.")
    parser.add_argument("--frequency_print_progress", type=int, default=1, help="Frequency of printing progress.")
    parser.add_argument("--episode_length", type=int, default=200, help="")
    parser.add_argument("--resize_pixels", type=int, default=80, help="Number of pixels for downsampling.")
    args = parser.parse_args()
    main(args)
