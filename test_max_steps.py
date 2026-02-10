#!/usr/bin/env python3

from envs.battery_wrapper import make_battery_warehouse

# Create with 200 steps explicitly
env = make_battery_warehouse(max_steps=200)

print(f"Wrapper env type: {type(env.env).__name__}")
print(f"Wrapper env max_steps: {getattr(env.env, 'max_steps', 'NOT FOUND')}")

unwrapped = env.unwrapped
print(f"Unwrapped type: {type(unwrapped).__name__}")
print(f"Unwrapped max_steps: {getattr(unwrapped, 'max_steps', 'NOT FOUND')}")

# Check if there's TimeLimit wrapper
import gymnasium as gym
if isinstance(env, gym.wrappers.TimeLimit):
    print(f"Is TimeLimit wrapper, max_episode_steps: {env.max_episode_steps}")

# Run one episode and count steps
obs, _ = env.reset()
steps = 0
for i in range(300):
    obs, reward, terminated, truncated, info = env.step((0,))  # Just move forward
    steps += 1
    if terminated or truncated:
        print(f"\nEpisode ended at step: {steps}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        break
