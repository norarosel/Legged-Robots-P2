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

""" Run CPG """
import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv


EXTENSION = True # THIS IS FOR WHEN I IMPLEMENT THE EXTENSION PART. ONLY USED WHEN THAT.
ADD_CARTESIAN_PD = True


TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP, gait="TROT")

# DEFINE WHICH GAIT WE USE. Not used anymore.
#GAIT = 0 # 0 = TROT ; 1 = WALK ; 2 = PACE ; 3 = BOUND. Use index to choose the right omegas.
#cpg.what_gait(GAIT)

TEST_STEPS = int(3/ (TIME_STEP))  # On run 10 secondes et on divise ça par 0.001 (1 ms). 
t = np.arange(TEST_STEPS)*TIME_STEP

# ************************************************************************************************************************************************************** # 
# [TODO] initialize data structures to save CPG and robot states
# Aller chercher les valeurs du cpg : cpg._cequ'onveut. X donne r et theta, X_dot donne rdot et thetadot
time                = np.arange(TEST_STEPS)
r_list              = np.zeros((4,TEST_STEPS))  # Idée : chaque jambe est sauvegardée sur une ligne, à chaque time step, on update la valeur
rdot_list           = np.zeros((4,TEST_STEPS))
theta_list          = np.zeros((4,TEST_STEPS))
thetadot_list       = np.zeros((4,TEST_STEPS))
# Attention : comme on peut ne travailler qu'avec une jambe, je fais 3 étages : 3 coordonnées spatiale et 3 joint angles
des_foot_pos_list   = np.zeros((3,TEST_STEPS))  # Structure : première ligne = x et deuxième ligne = z 
act_foot_pos_list   = np.zeros((3,TEST_STEPS))
des_joint_angle     = np.zeros((3,TEST_STEPS))
act_joint_angle     = np.zeros((3,TEST_STEPS))  # Structure = hip - thigh - calf (ordre des lignes)


# For velocity, duty cycle/ratio, time duration of one step + Cost of Transport

body_velocity_list  = np.zeros((3, TEST_STEPS))  # Structure : dx, dy, dz sur les 3 lignes, à chaque instant

# Power and cost of transport
E_consumed  = 0
evol_E      = np.zeros(TEST_STEPS)
inst_power  = 0

COT_tot     = 0

delta_dist  = 0
distance    = 0
x_list      = np.zeros(TEST_STEPS) # For trajectory
y_list      = np.zeros(TEST_STEPS) 
x_prev      = env.robot.GetBasePosition()[0]
y_prev      = env.robot.GetBasePosition()[1]
dist_list   = np.zeros(TEST_STEPS)

robot_mass = sum(env.robot.GetTotalMassFromURDF())

# Duty cycle/ratio and time duration of one step. Looking only at ONE LEG
ONE_CYCLE       = [False,False,False,False] # Used to only look at ONE cycle
LEG_CYCLE_START = [False,False,False,False] # To detect when a cycle starts
step_prev       = np.zeros(4)
t_swing         = np.zeros(4)
t_stance        = np.zeros(4)

# ************************************************************************************************************************************************************** # 

############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])

# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([10]*3)

for j in range(TEST_STEPS): 
    
  inst_power    = 0
  E_inst        = 0
  delta_dist  = 0
  
  # initialize torque array to send to motors
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()
  
  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q     = env.robot.GetMotorAngles()
  dq    = env.robot.GetMotorVelocities() # Dans quadruped.py, on voit une multiplication entre la valeur "amplitude" et
  # direction, donc il ne faut pas être surpris par les signes.

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])
    
    
    # ************************** FOR DUTY CYCLE PURPOSES *************************************#
    if(cpg.X[0,i] - cpg._mu < 0.01): # On met une tolérance sur la convergence
        if(not ONE_CYCLE[i]): # Si on n'a pas encore fait un cycle complet
            if(not LEG_CYCLE_START[i]): # Si on n'a pas démarré un cycle
                if(leg_xyz[2] > -cpg._robot_height): # Si z décolle du sol. OK ? Dans l'idée, j'ai défini avant qu'on serait en swing si >= mais ici, pour éviter de bug, je dis >.
                    LEG_CYCLE_START[i]  = True
                    t_swing[i]          += 1
                    step_prev[i]        = np.sin( (cpg.X[1,i] % (2*np.pi) ) ) 
            else:
                if( (np.sin( (cpg.X[1,i] % (2*np.pi) ) ) > 0.0) & (step_prev[i] <= 0.0) ):
                    ONE_CYCLE[i]    = True
                elif(np.sin( (cpg.X[1,i] % (2*np.pi) ) ) > 0): # Si on vole
                    t_swing[i]      += 1
                    step_prev[i]    = np.sin( (cpg.X[1,i] % (2*np.pi) ) )
                else:
                    t_stance[i]     += 1
                    step_prev[i]    = np.sin( (cpg.X[1,i] % (2*np.pi) ) )
    # ************************** FOR DUTY CYCLE PURPOSES *************************************#
    
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz) 
    # Add joint PD contribution to tau for leg i (Equation 4)
    # Avec des arrays, on peut facilement faire les opérations oklm dessus.
    
    tau += kp * (leg_q - q[i*3:(i+1)*3]) - kd * dq[i*3:(i+1)*3] # [TODO]
    
    
    # Body velocity: used for later computations
    body_vel = env.robot.GetBaseLinearVelocity() # (dx, dy, dz)  
    
    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      # [TODO] 
      J, foot_pos_leg_frame = env.robot.ComputeJacobianAndPosition(i)
      
      # Get current foot velocity in leg frame (Equation 2)
      # [TODO] 
      foot_vel_leg_frame = J @ dq[i*3:(i+1)*3]
      
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      transpose_J = np.transpose(J)
      
      tau += np.dot(transpose_J, np.dot(kpCartesian, (leg_xyz - foot_pos_leg_frame)) - np.dot(kdCartesian, foot_vel_leg_frame)) # [TODO]

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # ************************************************************************************************************************************************************** # 
  # [TODO] save any CPG or robot states
    # Compute power and Cost Of Transport:
    inst_power  += np.abs(tau @ np.transpose(dq[i*3:(i+1)*3]))

  delta_dist    = np.sqrt((env.robot.GetBasePosition()[0] - x_prev) **2 + (env.robot.GetBasePosition()[1] - y_prev) **2)
  x_list[j]     = x_prev
  y_list[j]     = y_prev
  
  x_prev        = env.robot.GetBasePosition()[0] # Ici on actualise 
  y_prev        = env.robot.GetBasePosition()[1]
  E_inst        = inst_power * TIME_STEP
  
  
  E_consumed    += E_inst
  distance      += delta_dist
  dist_list[j]  = distance
  evol_E[j]     = E_consumed
          
  
  # UPDATE LISTS CONTENT: r, rdot, theta, thetadot
  for leg in range(4):
      r_list[leg, j]        = cpg.X[0,leg]
      rdot_list[leg, j]     = cpg.X_dot[0,leg]
      theta_list[leg, j]    = cpg.X[1,leg] 
      thetadot_list[leg, j] = cpg.X_dot[1,leg]
  
  # Je travaille toujours sur la dernière jambe, c'est-à-dire Rear Left (arrière gauche)
  # UPDATE LISTS CONTENT: desired and actual foot position & desired and actual joint angles and body velocity
  for foot in range(3):
      des_foot_pos_list[foot, j] = leg_xyz[foot]
      act_foot_pos_list[foot, j] = foot_pos_leg_frame[foot]
      des_joint_angle[foot, j]   = leg_q[foot]
      act_joint_angle[foot, j]   = q[9+foot] # Attention : dimension 12
      
      body_velocity_list[foot,j] = body_vel[foot]
      
  
  # ************************************************************************************************************************************************************** #   

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

duty_cycle      = np.zeros(4)
duty_ratio      = np.zeros(4)
duty_ratio_real = np.zeros(4)
for i in range(4):
    duty_cycle[i]       = t_swing[i] + t_stance[i]
    duty_ratio[i]       = t_stance[i]/t_swing[i]
    duty_ratio_real[i]  = (t_swing[i]/duty_cycle[i]) * 100


COT_tot         = E_consumed / (robot_mass * 9.81 * distance)

##################################################### 
# PLOTS
#####################################################
print(" #****************************************#" + '\n')
print(" OBTAINED VALUE FOR THE GAIT " + '\n')
print(" #****************************************#" + '\n')

print(f"Time in swing mode : {t_swing} [ms]")
print(f"Time in stance mode : {t_stance} [ms] (time duration of one step)")
print(f"Duty cycle : {duty_cycle} [ms]")
print(f"Duty ratio : {duty_ratio} [/]")
print(f"Duty cycle in percentage : {duty_ratio_real} [%]")
print(f"CoT of the gait : {COT_tot} [/]")

"""
for i in range(4):
    fig, axs = plt.subplots(2, 2)
    if(i == 0):
        fig.suptitle('Results for Front Right leg')
    elif(i == 1):
        fig.suptitle('Results  for Front Left leg')    
    elif(i == 2):
        fig.suptitle('Results  for Rear Right leg')
    else:
        fig.suptitle('Results  for Rear Left leg')
    axs[0, 0].plot(time, r_list[i,:])
    axs[0, 0].set_title('Time evolution of r')
    axs[0, 1].plot(time, rdot_list[i,:], 'tab:orange')
    axs[0, 1].set_title('Time evolution of r_dot')
    axs[1, 0].plot(time, theta_list[i,:], 'tab:green')
    axs[1, 0].set_title('Time evolution of theta')
    axs[1, 1].plot(time, thetadot_list[i,:], 'tab:red')
    axs[1, 1].set_title('Time evolution of theta_dot')
    
    for ax in axs.flat:
        ax.set(xlabel='time [ms]', ylabel='Amplitude')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
   
fig = plt.figure()
plt.plot(time, np.c_[des_foot_pos_list[0,:], act_foot_pos_list[0,:]], label=['Desired', 'Actual'])
plt.title("Desired vs actual x foot position of the Rear Left leg in time")
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude [m]")
plt.legend()

fig = plt.figure()
plt.plot(time, np.c_[des_foot_pos_list[2,:], act_foot_pos_list[2,:]], label=['Desired', 'Actual'])
plt.title("Desired vs actual z foot position of the Rear Left leg in time")
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude [m]")
plt.legend()



plt.show()    """