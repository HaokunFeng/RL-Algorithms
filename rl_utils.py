from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import gym
import time
import os
import math
from moviepy import ImageSequenceClip, clips_array
import json
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

# ---------------------------------------------------------------------------------------------
# Functions to train on-policy and off-policy agents
# ---------------------------------------------------------------------------------------------

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                reset_output = env.reset()
                state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
                done = False

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                agent.update(transition_dict)

                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                reset_output = env.reset()
                state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
                done = False

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list

# ---------------------------------------------------------------------------------------------
# Functions to compute advantage and moving average
# ---------------------------------------------------------------------------------------------

def compute_advantage(gamma, lmbda, td_error):
    td_error = td_error.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_error[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# ---------------------------------------------------------------------------------------------
# Functions to save and load models
# ---------------------------------------------------------------------------------------------
def save_model(agent, save_path):
    os.makedirs(os.path.dirname(save_path))






# ---------------------------------------------------------------------------------------------------------------
# Functions to display and record agent performance
# ---------------------------------------------------------------------------------------------------------------
def watch_agent(env_name, agent, device, sleep_time=0.01, num_episodes=1):
    env_fn = lambda: gym.make(env_name, render_mode='human')
    for i in range(num_episodes):
        env = env_fn()
        reset_output = env.reset()
        state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        done = False
        while not done:
            env.render()
            with torch.no_grad():
                action = agent.take_action(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state

            time.sleep(sleep_time)
        env.close()


def record_one_episode(env_name, agent, device, save_path='./video', filename='agent_play', fps=30):
    os.makedirs(save_path, exist_ok=True)
    env_fn = lambda: gym.make(env_name, render_mode='rgb_array')
    env = env_fn()
    frames = []

    reset_output = env.reset()
    state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
    done = False

    while not done:
        frame = env.render()
        frames.append(frame)
        with torch.no_grad():
            action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

    env.close()
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_gif(os.path.join(save_path, f'{filename}.gif'), fps=fps)

def record_multiple_episodes(env_name, agent, device, num_episodes=10, videos_per_row=5, save_path='./video', filename='agent_multi_play', fps=30):
    os.makedirs(save_path, exist_ok=True)
    env_fn = lambda: gym.make(env_name, render_mode='rgb_array')
    all_clips = []
    for ep in range(num_episodes):
        env = env_fn()
        frames = []

        reset_output = env.reset()
        state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        done = False

        while not done:
            frame = env.render()
            frames.append(frame)
            with torch.no_grad():
                action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
        
        env.close()

        clip = ImageSequenceClip(frames, fps=fps)
        all_clips.append(clip)
    
    rows = []
    for i in range(0, num_episodes, videos_per_row):
        row = all_clips[i:i+videos_per_row]
        rows.append(row)
    
    final_clip = clips_array(rows)
    final_clip.write_gif(os.path.join(save_path, f'{filename}.gif'), fps=fps)


