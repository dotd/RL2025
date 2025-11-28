import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from collections import namedtuple
from itertools import count
from PIL import Image
from collections import deque
import argparse


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import gymnasium as gym

logger = logging.getLogger(__name__)

# For newer Pillow versions, use Image.Resampling.BICUBIC
# For backward compatibility, try Image.BICUBIC first
try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC

############ HYPERPARAMETERS ##############


def get_args():
    parser = argparse.ArgumentParser(description="CartPole visual DQN parameters")
    parser.add_argument("--BATCH_SIZE", type=int, default=128, help="Batch size for optimization")
    parser.add_argument("--GAMMA", type=float, default=0.999, help="Discount factor for rewards")
    parser.add_argument("--EPS_START", type=float, default=0.9, help="Starting value of epsilon for epsilon-greedy")
    parser.add_argument("--EPS_END", type=float, default=0.01, help="Final value of epsilon")
    parser.add_argument("--EPS_DECAY", type=int, default=3000, help="Epsilon decay rate")
    parser.add_argument("--TARGET_UPDATE", type=int, default=50, help="Episode frequency for updating target network")
    parser.add_argument("--MEMORY_SIZE", type=int, default=100000, help="Replay memory capacity")
    parser.add_argument("--END_SCORE", type=int, default=200, help="Score at which to end episode (Cartpole-v0)")
    parser.add_argument("--TRAINING_STOP", type=int, default=142, help="Stop training when mean score falls below this")
    parser.add_argument("--N_EPISODES", type=int, default=5000, help="Total number of episodes")
    parser.add_argument("--RUNS", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--LAST_EPISODES_NUM", type=int, default=20, help="Number of last episodes for early stopping")
    parser.add_argument("--FRAMES", type=int, default=2, help="Number of last frames to compose state")
    parser.add_argument("--RESIZE_PIXELS", type=int, default=60, help="Image resize dimension")

    # ---- CONVOLUTIONAL NEURAL NETWORK ----
    parser.add_argument("--HIDDEN_LAYER_1", type=int, default=64, help="First hidden layer size")
    parser.add_argument("--HIDDEN_LAYER_2", type=int, default=64, help="Second hidden layer size")
    parser.add_argument("--HIDDEN_LAYER_3", type=int, default=32, help="Third hidden layer size")
    parser.add_argument("--KERNEL_SIZE", type=int, default=5, help="CNN kernel size")
    parser.add_argument("--STRIDE", type=int, default=2, help="CNN stride")
    # --------------------------------------

    parser.add_argument("--GRAYSCALE", action="store_true", default=True, help="Use grayscale images (default True)")
    parser.add_argument("--RGB", dest="GRAYSCALE", action="store_false", help="Use RGB images instead of grayscale")
    parser.add_argument("--LOAD_MODEL", action="store_true", default=False, help="Load model from file")
    parser.add_argument("--USE_CUDA", action="store_true", default=False, help="Use CUDA (GPU) for computation")

    args = parser.parse_args()
    return args


# Memory for Experience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # if we haven't reached full capacity, we append a new transition
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # e.g if the capacity is 100, and our position is now 101, we don't append to
        # position 101 (impossible), but to position 1 (its remainder), overwriting old data

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, outputs, nn_inputs, HIDDEN_LAYER_1, HIDDEN_LAYER_2, HIDDEN_LAYER_3, KERNEL_SIZE, STRIDE):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(nn_inputs, HIDDEN_LAYER_1, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn1 = nn.BatchNorm2d(HIDDEN_LAYER_1)
        self.conv2 = nn.Conv2d(HIDDEN_LAYER_1, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn2 = nn.BatchNorm2d(HIDDEN_LAYER_2)
        self.conv3 = nn.Conv2d(HIDDEN_LAYER_2, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn3 = nn.BatchNorm2d(HIDDEN_LAYER_3)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=KERNEL_SIZE, stride=STRIDE):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        nn.Dropout()
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def get_cart_location(screen_width, cart_position=None):
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
            logger.warning("Cannot access cart position from env.unwrapped.state, using screen center")
            return screen_width // 2
    else:
        cart_pos = cart_position

    return int(cart_pos * scale + screen_width / 2.0)  # MIDDLE OF CART


# Cropping, downsampling (and Grayscaling) image
def get_screen(observation=None):
    """
    Get and preprocess screen image.

    Args:
        observation: Current observation from env. If provided, used for cart location.
    """
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render().transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)

    # Get cart position from observation if available
    cart_position = observation[0] if observation is not None else None
    cart_location = get_cart_location(screen_width, cart_position)
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


# Action selection , if stop training == True, only exploitation
def select_action(state, stop_training):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    # print('Epsilon = ', eps_threshold, end='\n')
    if sample > eps_threshold or stop_training:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# Plotting
def plot_durations(score):
    fig, ax = plt.subplots(figsize=(16, 8))
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    episode_number = len(durations_t)
    ax.set_title("Training...")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Duration")
    dur = durations_t.numpy()
    plt.style.use("seaborn")  # Change/Remove This If you Want
    #     ax.plot(dur, label= 'Score')
    ax.fill_between(np.linspace(1, episode_number, episode_number), dur - dur.std(), dur + dur.std(), alpha=0.2)

    plt.hlines(195, 0, episode_number, colors="red", linestyles=":", label="Win Threshold")

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        last100_mean = means[episode_number - 100].item()
        means = torch.cat((torch.zeros(99), means))
        ax.plot(means.numpy(), label="Last 100 mean")
        print("Episode: ", episode_number, " | Score: ", score, "| Last 100 mean = ", last100_mean)
    ax.legend(loc="upper left")
    fig.savefig("plots/" + graph_name)
    #     if is_ipython:
    #         display.clear_output(wait=True)
    #         display.display(plt.gcf())
    fig.show()


# Training
def optimize_model(memory, policy_net, target_net, device, BATCH_SIZE, GAMMA):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    # torch.cat concatenates tensor sequence
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).type(torch.FloatTensor).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    plt.figure(2)
    print(f"Loss: {loss}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


############################################
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{script_dir}/plots", exist_ok=True)
os.makedirs(f"{script_dir}/save_model", exist_ok=True)

graph_name = "cartpole_vision"
device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

# Settings for GRAYSCALE / RGB
if GRAYSCALE == 0:
    resize = T.Compose([T.ToPILImage(), T.Resize(RESIZE_PIXELS, interpolation=BICUBIC), T.ToTensor()])

    nn_inputs = 3 * FRAMES  # number of channels for the nn
else:
    resize = T.Compose([T.ToPILImage(), T.Resize(RESIZE_PIXELS, interpolation=BICUBIC), T.Grayscale(), T.ToTensor()])
    nn_inputs = FRAMES  # number of channels for the nn


stop_training = False

env = gym.make("CartPole-v1", render_mode="rgb_array")

# Set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# If gpu is to be used

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# Cart location for centering image crop


observation, _ = env.reset()
plt.figure()
if GRAYSCALE == 0:
    plt.imshow(get_screen(observation).cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation="none")
else:
    plt.imshow(get_screen(observation).cpu().squeeze(0).permute(1, 2, 0).numpy().squeeze(), cmap="gray")
plt.title("Example extracted screen")
plt.show(block=False)

env.close()

########################################################
eps_threshold = 0.9  # original = 0.9

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
print("Screen height: ", screen_height, " | Width: ", screen_width)

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if LOAD_MODEL == True:
    policy_net_checkpoint = torch.load("save_model/policy_net_best3.pt")  # best 3 is the default best
    target_net_checkpoint = torch.load("save_model/target_net_best3.pt")
    policy_net.load_state_dict(policy_net_checkpoint)
    target_net.load_state_dict(target_net_checkpoint)
    policy_net.eval()
    target_net.eval()
    stop_training = True  # if we want to load, then we don't train the network anymore

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0


episodes_trajectories = []
episodes_after_stop = 100


# MAIN LOOP
stop_training = False
for j in range(RUNS):
    mean_last = deque([0] * LAST_EPISODES_NUM, LAST_EPISODES_NUM)
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(MEMORY_SIZE)

    count_final = 0

    steps_done = 0
    episode_durations = []
    for i_episode in range(N_EPISODES):
        # Initialize the environment and state
        env.reset()
        init_screen = get_screen()
        screens = deque([init_screen] * FRAMES, FRAMES)
        state = torch.cat(list(screens), dim=1)

        for t in count():

            # Select and perform an action
            action = select_action(state, stop_training)
            state_variables, _, done, truncated, _ = env.step(action.item())

            # Observe new state
            screens.append(get_screen())
            next_state = torch.cat(list(screens), dim=1) if not done else None

            # Reward modification for better stability
            x, x_dot, theta, theta_dot = state_variables
            r1 = (env.unwrapped.x_threshold - abs(x)) / env.unwrapped.x_threshold - 0.8
            r2 = (env.unwrapped.theta_threshold_radians - abs(theta)) / env.unwrapped.theta_threshold_radians - 0.5
            reward = r1 + r2
            reward = torch.tensor([reward], device=device)
            if t >= END_SCORE - 1:
                reward = reward + 20
                done = 1
            else:
                if done:
                    reward = reward - 20

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if done:
                episode_durations.append(t + 1)
                mean_last.append(t + 1)
                mean = 0
                print({"Episode duration": t + 1, "Episode number": i_episode})
                for i in range(LAST_EPISODES_NUM):
                    mean = mean_last[i] + mean
                mean = mean / LAST_EPISODES_NUM
                if mean < TRAINING_STOP and stop_training == False:
                    optimize_model()
                else:
                    stop_training = True
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if stop_training == True:
            count_final += 1
            if count_final >= 100:
                break

    print("Complete")
    stop_training = False
    episodes_trajectories.append(episode_durations)

plt.ioff()
plt.show(block=False)

# Cherry picking best runs
best = []
for i in range(len(episodes_trajectories)):
    best.append(episodes_trajectories[i])

maximum = 0
for i in range(len(best)):
    maximum = max(len(best[i]), maximum)

# Fill the episodes to make them the same length
for i in range(len(best)):
    length = len(best[i])
    for j in range(maximum - len(best[i])):
        best[i].append(best[i][j + length - 100])
    best[i] = np.asarray(best[i])

best = np.asarray(best)

# To numpy
score_mean = np.zeros(maximum)
score_std = np.zeros(maximum)
last100_mean = np.zeros(maximum)
print(best[:, max(0, -99) : 1].mean())
for i in range(maximum):
    score_mean[i] = best[:, i].mean()
    score_std[i] = best[:, i].std()
    last100_mean[i] = best[:, max(0, i - 50) : min(maximum, i + 50)].mean()
print(len(last100_mean))

t = np.arange(0, maximum, 1)

# from scipy.interpolate import make_interp_spline # make smooth version
# interpol = make_interp_spline(t, score_mean, k=3)  # type: BSpline

fig, ax = plt.subplots(figsize=(16, 8))
ax.fill_between(t, np.maximum(score_mean - score_std, 0), np.minimum(score_mean + score_std, 200), color="b", alpha=0.2)
# ax.legend(loc='upper right')
ax.set_xlabel("Episode")
ax.set_ylabel("Score")
# ax.set_title('Inverted Pendulum Training Plot from Pixels')
ax.plot(t, score_mean, label="Score Mean")
ax.plot(t, last100_mean, color="purple", linestyle="dotted", label="Smoothed mean")
ax.legend()
fig.savefig(f"{script_dir}/plots/score.png")
