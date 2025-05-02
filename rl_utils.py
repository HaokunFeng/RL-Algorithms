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
import pickle

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

def train_on_policy_agent(env,
                          agent,
                          num_episodes):
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


def train_off_policy_agent(env, 
                           agent, 
                           num_episodes, 
                           replay_buffer, 
                           minimal_size, 
                           batch_size):
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

def compute_advantage(gamma, 
                      lmbda, 
                      td_error):
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

def moving_average_score(data, window_size=10):
    return np.mean(data[-window_size:]) if len(data) >= window_size else np.mean(data)

# ---------------------------------------------------------------------------------------------
# Functions to save and load models, checkpoints, return_list, curves and logs
# ---------------------------------------------------------------------------------------------
def save_agent(agent, 
               model_name, 
               save_dir='./agent', 
               score=None):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if score is not None:
        save_path = os.path.join(save_dir, f"{model_name}_score{score:.2f}_{timestamp}.pth")
    else:
        save_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pth")
    torch.save(agent.state_dict(), save_path)
    print(f"✅ Single agent saved at {save_path}")

    _save_log(save_dir, model_name, save_path, score)

def save_multi_agents(agent_dict, 
                      model_name, 
                      save_dir='./agent', 
                      score=None):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if score is not None:
        save_path = os.path.join(save_dir, f"{model_name}_score{score:.2f}_{timestamp}.pth")
    else:
        save_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pth")
    state_dict = {name: agent.state_dict() for name, agent in agent_dict.items()}
    torch.save(state_dict, save_path)
    print(f"✅ Multi-agents saved at {save_path}")
    _save_log(save_dir, model_name, save_path, score)

def save_checkpoint(agent_dict, 
                    optimizer_dict, 
                    episode, 
                    model_name, 
                    save_dir='./agent'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_checkpoint_ep{episode}.pth")
    state = {
        'episode': episode,
        'agent_state_dict': {name: agent.state_dict() for name, agent in agent_dict.items()},
        'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizer_dict.items()}
    }
    torch.save(state, save_path)
    print(f"✅ Checkpoint saved at {save_path}")
    _save_log(save_dir, model_name, save_path, note=f"Checkpoint at episode {episode}")

def save_return_list(return_list, 
                     save_path='./results', 
                     filename='return_list.pkl'):
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(return_list, f)
    print(f"✅ Return list saved at {save_path}")

def save_return_curve(return_list, 
                       model_name, 
                       mv_return=None, 
                       save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(return_list, label='Returns')
    if mv_return is not None:
        plt.plot(mv_return, label='Moving Average Return')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'Return Curve for {model_name}')
    plt.grid(True)
    curve_path = os.path.join(save_dir, f"{model_name}_return_curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"📈 Return curve saved at {curve_path}")

def save_all(agent_dict, 
             return_list, 
             model_name, 
             mv_return=None, 
             score=None, 
             save_dir_agent='./agent', 
             save_dir_result='./results'):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"_score{score:.2f}_{timestamp}" if score is not None else f"_{timestamp}"
    
    # Save model
    os.makedirs(save_dir_agent, exist_ok=True)
    model_filename = f"{model_name}{suffix}.pth"
    model_path = os.path.join(save_dir_agent, model_filename)
    state_dict = {name: agent.state_dict() for name, agent in agent_dict.items()}
    torch.save(state_dict, model_path)
    print(f"✅ Multi-agents saved at {model_path}")
    _save_log(save_dir_agent, model_name, model_path, score)

    # Save return list
    os.makedirs(save_dir_result, exist_ok=True)
    return_list_filename = f"{model_name}{suffix}_return_list.pkl"
    return_list_path = os.path.join(save_dir_result, return_list_filename)
    with open(return_list_path, 'wb') as f:
        pickle.dump(return_list, f)
    print(f"✅ Return list saved at {return_list_path}")

    # Save return curve
    curve_filename = f"{model_name}{suffix}_return_curve.png"
    curve_path = os.path.join(save_dir_result, curve_filename)
    plt.figure()
    plt.plot(return_list, label='Returns')
    if mv_return is not None:
        plt.plot(mv_return, label='Moving Average Return')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'Return Curve for {model_name}')
    plt.grid(True)
    plt.savefig(curve_path)
    plt.close()
    print(f"📈 Return curve saved at {curve_path}")

def load_all(agent_dict, 
              model_name, 
              timestamp, 
              score=None, 
              save_dir_agent='./agent', 
              save_dir_result='./results', 
              device='cpu'):
    suffix = f"_score{score:.2f}_{timestamp}" if score is not None else f"_{timestamp}"

    # Load model
    model_path = os.path.join(save_dir_agent, f"{model_name}{suffix}.pth")
    state_dicts = torch.load(model_path, map_location=device)
    for name, agent in agent_dict.items():
        agent.load_state_dict(state_dicts[name])
    print(f"🔄 Multi-agents loaded from {model_path}")

    # Load return list
    return_list_path = os.path.join(save_dir_result, f"{model_name}{suffix}_return_list.pkl")
    with open(return_list_path, 'rb') as f:
        return_list = pickle.load(f)
    print(f"🔄 Return list loaded from {return_list_path}")

    # Optional: load return curve image path (not data)
    curve_path = os.path.join(save_dir_result, f"{model_name}{suffix}_return_curve.png")
    if os.path.exists(curve_path):
        print(f"📈 Return curve image available at {curve_path}")
    else:
        print("⚠️ Return curve image not found.")

    return agent_dict, return_list



def load_agent(agent, 
               load_path, 
               device):
    state_dict = torch.load(load_path, map_location=device)
    agent.load_state_dict(state_dict)
    print(f"🔄 Single agent loaded from {load_path}")
    return agent

def load_multi_agents(agent_dict, 
                      load_path, 
                      device):
    state_dicts = torch.load(load_path, map_location=device)
    for name, agent in agent_dict.items():
        agent.load_state_dict(state_dicts[name])
    print(f"🔄 Multi-agents loaded from {load_path}")
    return agent_dict

def load_checkpoint(agent_dict, 
                    optimizer_dict, 
                    load_path, 
                    device):
    checkpoint = torch.load(load_path, map_location=device)
    for name, agent in agent_dict.items():
        agent.load_state_dict(checkpoint['agent_state_dict'][name])
    for name, opt in optimizer_dict.items():
        opt.load_state_dict(checkpoint['optimizer_state_dict'][name])
    start_episode = checkpoint['episode']
    print(f"🔄 Checkpoint loaded from {load_path}, starting from episode {start_episode}")
    return agent_dict, optimizer_dict, start_episode

def load_return_list(filepath):
    with open(filepath, 'rb') as f:
        return_list = pickle.load(f)
    print(f"🔄 Return list loaded from {filepath}")
    return return_list

def _save_log(save_dir, 
               model_name, 
               save_path, 
               score=None, 
               note=None):
    log_path = os.path.join(save_dir, "save_log.json")
    log_data = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)

    entry = {
        "model_name": model_name,
        "save_path": save_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if score is not None:
        entry["score"] = score
    if note:
        entry["note"] = note
    
    log_data.append(entry)

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"📝 Save log updated at {log_path}")

# ---------------------------------------------------------------------------------------------------------------
# Functions to display and record agent performance
# ---------------------------------------------------------------------------------------------------------------
def watch_agent(env_name, 
                agent, 
                device, 
                sleep_time=0.01, 
                num_episodes=1):
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


def record_one_episode(env_name, 
                       agent, 
                       device, 
                       save_dir='./video', 
                       filename='agent_play', 
                       score=None, 
                       fps=30):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if score is not None:
        save_path = os.path.join(save_dir, f"{filename}_score{score:.2f}_{timestamp}.gif")
    else:
        save_path = os.path.join(save_dir, f"{filename}_{timestamp}.gif")

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
    clip.write_gif(save_path, fps=fps)
    print(f"✅ Video saved at {save_path}")

def record_multiple_episodes(env_name, 
                             agent, 
                             device, 
                             num_episodes=10, 
                             videos_per_row=5, 
                             save_dir='./video', 
                             filename='agent_multi_play', 
                             score=None, 
                             fps=30):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if score is not None:
        save_path = os.path.join(save_dir, f"{filename}_score{score:.2f}_{timestamp}.gif")
    else:
        save_path = os.path.join(save_dir, f"{filename}_{timestamp}.gif")

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
    final_clip.write_gif(save_path, fps=fps)
    print(f"✅ Video saved at {save_path}")


