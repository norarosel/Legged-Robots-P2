# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3
from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "SAC"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '121321105810'
log_dir = interm_dir + '121123085603'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training

# motor_control_modes: TORQUE, PD, CARTESIAN_PD, CPG
# task_env: FWD_LOCOMOTION, FLAGRUN, LR_COURSE_TASK
# observation_space_mode: DEFAULT, LR_COURSE_OBS, CPG_OBS
# Render slow
env_config = {"motor_control_mode": "CPG", 
              "task_env": "FLAGRUN", 
              "observation_space_mode": "CPG_OBS", 
              "render": False}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
# env_config['competition_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
num_time_steps = 2000
episode_rewards = np.zeros(num_time_steps) #in this array we will save the cumulative reward in each time step
base_positions = np.zeros((num_time_steps, 3)) #base_positions will store the base positions for each time step
base_positions_in_euler = np.zeros((num_time_steps, 3)) #base_positions_euler will store the base_positions in euler angles for each time step
base_linear_velocities = np.zeros((num_time_steps, 3)) #base_linear_velocity will store the velocity of the base in x y z

testing_mode = True  # Set to True when you want to test, False for training

for i in range(num_time_steps):
    action, _states = model.predict(obs, deterministic=(not testing_mode)) # sample at test time? ([TODO]: test) #here we are using our trained model to predict an action based on obs
    obs, rewards, dones, info = env.step(action) #this gives you the new observation based on the action taken 
    episode_reward += rewards #rewards is a scalar that represents the immediate feedback given by the environment for the action taken at the current time step
    base_positions[i] = env.envs[0].env.robot.GetBasePosition()
    episode_rewards[i] = episode_reward
    base_positions_in_euler[i, :] = env.envs[0].env.robot.GetBaseOrientationRollPitchYaw()
    base_linear_velocities[i, :] = env.envs[0].env.robot.GetBaseLinearVelocity()

    if dones:  #boolean flag that represents if the episode is finished or not
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos']) #the base position printed here is the one at the end of the episode?
        episode_reward = 0
         

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    
    
# [TODO] make plots:

# Plotting base positions against time
time_steps = np.arange(num_time_steps)  # Time steps from 0 to num_time_steps - 1

plt.figure(figsize=(10, 6))
plt.plot(time_steps, base_positions[:, 0], label='X-axis')
plt.plot(time_steps, base_positions[:, 1], label='Y-axis')
plt.plot(time_steps, base_positions[:, 2], label='Z-axis')
plt.title('Base Positions vs Time')
plt.xlabel('Time Steps')
plt.ylabel('Base Positions')
plt.legend()
plt.show()

# Plotting cumulative reward over time (we are not interested in knowing the individual value of the reward at a time step, but the reward the robot has accumulated up to that point)

plt.figure(figsize=(10, 6))
plt.plot(time_steps, episode_rewards)
plt.title('Cumulative Reward vs Time Steps')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Reward')
plt.show()

# Plotting the mean of last value of the reward in each episode, which, as time passes
#  should get bigger (this one is already given to us and will only be interesting during training, because we have a really big amount of steps)
#However, the cumulative reward over time is interesting in training because we can analyze the cumulative reward at each time step, and not for each episode


# Plotting base linear velocities against time

plt.figure(figsize=(10, 6))
plt.plot(time_steps, base_linear_velocities[:, 0], label='X-axis')
plt.plot(time_steps, base_linear_velocities[:, 1], label='Y-axis')
plt.plot(time_steps, base_linear_velocities[:, 2], label='Z-axis')
plt.title('Base Linear Velocities vs Time')
plt.xlabel('Time Steps')
plt.ylabel('Base Linear Velocities')
plt.legend()
plt.show()

#Plotting x vs y to see the trajectory (and plot the goal on the same plot to see if the robot is moving towards the objective)