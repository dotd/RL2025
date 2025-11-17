from ale_py import ALEInterface
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py

ale = ALEInterface()

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
print(f"obs shape: {obs.shape}")
# show obs as image

plt.imshow(obs)
plt.show(block=False)
print(f"reward: {reward}")
print(f"terminated: {terminated}")
print(f"truncated: {truncated}")
print(f"info: {info}")
print("Done")

# save obs as image
plt.imsave("obs.png", obs)
