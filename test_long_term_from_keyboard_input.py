# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import sys
import matplotlib.pyplot as plt
from long_term.dataset_locomotion_avatar import dataset, long_term_weights_path
from long_term.locomotion_utils import build_extra_features, compute_splines, compute_splines_from_keyboard_input
from long_term.pose_network_long_term import PoseNetworkLongTerm
from common.spline import Spline
from common.visualization import render_animation
from long_term.pace_network import PaceNetwork
import numpy as np

import keyboard  # using module keyboard
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch


default_subject = 'S1'

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

class agent():
    def __init__(self, delta_time):
        self.delta_time = delta_time
        self.cur_vel = np.array([0.0, 0.0])
        self.max_vel_norm = 100.0
        self.fric_facotr0 = 500.0
        self.fric_facotr1 = 10.0
        self.acc_factor = 1000.0
        self.start = False
        self.quit = False
        self.location = np.array([0.0, 0.0])
        self.visual_len = 30
        self.his_cnt = 0
        self.his_traj = np.array([np.array([0., 0.]) for i in range(self.visual_len)])
        self.vel_s0 = 20
        self.vel_s1 = 5

        self.orientation = 0.0
        self.cur_ang_vel = 0.0
        self.ang_fric_facotr0 = 1000.0
        self.ang_fric_facotr1 = 1000.0
        self.ang_vel_s0 = 72
        self.ang_vel_s1 = 18
        self.ang_acc_factor = 1000.0
        self.max_ang_vel = 100.0
        self.his_ang_traj = np.array([0. for i in range(self.visual_len)])
        self.his_speed = []
        
        # self.cur_acc = np.array([0.0, 0.0])
        # self.max_acc_norm = 100.0
        self.cur_anim = []

    def calc_angular_velocity(self, key):
        if key is None:
            ## the angular speed of this character should slow down to 0.
            self.cur_ang_vel *= 0.1
        else:
            ## now we should calculate the acc from the input
            cur_want_orient = None
            if key == 'w':
                cur_want_orient = 90.0
            if key == 's':
                cur_want_orient = 270.0
            if key == 'a':
                cur_want_orient = 180.0
            if key == 'd':
                cur_want_orient = 0.0
                
            ang_diff = cur_want_orient - self.orientation
            if np.abs(ang_diff) > 180.0:
                if ang_diff > 0:
                    self.orientation += 360.0
                    ang_diff = cur_want_orient - self.orientation
                else:
                    cur_want_orient += 360.0
                    ang_diff = cur_want_orient - self.orientation
            if np.abs(ang_diff) < 90.0:
                self.cur_ang_vel = ang_diff * 10.0
            else:
                self.cur_ang_vel = np.sign(ang_diff) * 900.0

    def calc_linear_velocity(self, key):
        tmp_acc = np.array([0.0, 0.0])
        tmp_fric = np.array([0.0, 0.0])
        if key is None:
            ## the speed of this character should slow down to 0.
            ## now there is only friction
            vel_norm = np.linalg.norm(self.cur_vel, 2) + 1e-9
            if vel_norm > self.vel_s0:
                tmp_fric += -(self.cur_vel / vel_norm) * self.fric_facotr0
            elif vel_norm > self.vel_s1:
                tmp_fric += -self.cur_vel * self.fric_facotr1
            else:
                tmp_fric += -self.cur_vel / self.delta_time
            
            ## pre speed check
            total_acc = tmp_acc + tmp_fric
            self.cur_vel += total_acc * self.delta_time
        else:
            ## the speed of this character should speed up to max_vel_norm 
            ## the friction is not related to the vel_norm any more
            vel_norm = np.linalg.norm(self.cur_vel, 2) + 1e-9
            if vel_norm > 20:
                tmp_fric += -(self.cur_vel / vel_norm) * self.fric_facotr0
            elif vel_norm > 5:
                tmp_fric += -self.cur_vel * self.fric_facotr1
            else:
                tmp_fric += -self.cur_vel / self.delta_time

            ## now we should calculate the acc from the input
            if key == 'w':
                tmp_acc += np.array([0.0, self.acc_factor])
            if key == 's':
                tmp_acc += np.array([0.0, -self.acc_factor])
            if key == 'a':
                tmp_acc += np.array([-self.acc_factor, 0.0])
            if key == 'd':
                tmp_acc += np.array([self.acc_factor, 0.0])

            ## pre speed check
            total_acc = tmp_acc + tmp_fric
            pre_cal_vel = total_acc * self.delta_time + self.cur_vel
            pre_cal_vel_norm = np.linalg.norm(pre_cal_vel, 2)
            cur_cal_vel_norm = np.linalg.norm(self.cur_vel, 2)
            ## if speed direction is not the same as acc, acc is not 0
            if cos_sim(tmp_acc, self.cur_vel) < 1.0:
                pass
            ## if speed direction is the same as acc
            else:
                ## and if the speed larger than max_vel
                if cur_cal_vel_norm >= self.max_vel_norm: 
                    total_acc = 0.0
                else:
                    if pre_cal_vel_norm > self.max_vel_norm:
                        total_acc *= 0.1
            self.cur_vel += total_acc * self.delta_time

            ## sanity check 
            cur_cal_vel_norm = np.linalg.norm(self.cur_vel, 2)
            if cur_cal_vel_norm > self.max_vel_norm:
                self.cur_vel = self.cur_vel / cur_cal_vel_norm * self.max_vel_norm

def keyboard_control(agent):
    ## initialize the model with pretrained weights
    ## load the data and compute the extra feature
    if torch.cuda.is_available():
        print('CUDA detected. Using GPU.')
        dataset.cuda()
    else:
        print('CUDA not detected. Using CPU.')

    ## remove unnecessary data to save time 
    for key in dataset['S1'].copy():
        if key.split('0')[0] != 'Idle':
            del dataset['S1'][key]

    dataset.compute_positions()
    build_extra_features(dataset)
    compute_splines(dataset)
    pace_net = PaceNetwork()
    pace_net.load_weights('weights_pace_network.bin')
    model = PoseNetworkLongTerm(30, dataset.skeleton())
    if torch.cuda.is_available():
        model.cuda()
    model.load_weights(long_term_weights_path) # Load pretrained model

    add_point_flag = False
    spline = None
    fout = open('tangent.txt', 'w')
    while not agent.quit:  # making a loop
        cur_key = None
        if keyboard.is_pressed('q'):  # if key 'q' is pressed 
            print('Quit this program!')
            agent.quit = True  # finishing the loop
        if keyboard.is_pressed('a'):  # if key 'a' is pressed 
            cur_key = 'a'
        if keyboard.is_pressed('w'):  # if key 'w' is pressed 
            cur_key = 'w'
        if keyboard.is_pressed('s'):  # if key 's' is pressed 
            cur_key = 's'
        if keyboard.is_pressed('d'):  # if key 'd' is pressed 
            cur_key = 'd'
        agent.calc_linear_velocity(cur_key)
        agent.location += agent.delta_time * agent.cur_vel 
        agent.his_traj[:-1] = agent.his_traj[1:].copy()
        agent.his_traj[-1] = agent.location

        agent.calc_angular_velocity(cur_key)
        agent.orientation += agent.delta_time * agent.cur_ang_vel
        agent.orientation %= 360.0
        agent.his_ang_traj[:-1] = agent.his_ang_traj[1:].copy()
        agent.his_ang_traj[-1] = agent.orientation / 180.0 * np.pi
        if cur_key is not None:
            agent.his_cnt += 1
            if agent.his_cnt > agent.visual_len: 
                agent.start = True  
        # fout.write(str(agent.location[0]) + ' ' + str(agent.location[1]) + ' ' + str(agent.orientation / 180.0 * np.pi) + '\n')
        if my_agent.start:
            if not add_point_flag:
                to_plot_traj = np.array(my_agent.his_traj)
                to_plot_ang_traj = np.array(my_agent.his_ang_traj)
                ## dirty trick: the minus symbol
                input_seq = np.concatenate([-to_plot_traj / 25.0, np.expand_dims(to_plot_ang_traj, -1)], -1)
                spline = compute_splines_from_keyboard_input(input_seq, 0.05)
                annotated_spline = pace_net.predict(spline)
                animation = model.generate_motion(annotated_spline, dataset[default_subject]['Idle001_1_d0'], False)
            else:
                ## dirty trick: the minus symbol
                spline.update_points(-my_agent.his_traj[-1] / 25.0, 500, 1)
                # print(spline.get_track('tangent')[-1])
                annotated_spline = pace_net.predict(spline, repara=False)
                # print(annotated_spline.start_length(), annotated_spline.length())
                fout.write(str(spline.get_track('tangent')[-1][0]) + ' ' + \
                           str(spline.get_track('tangent')[-1][1]) + ' ' + \
                           str(annotated_spline.get_track('tangent')[-1][0]) + ' ' + \
                           str(annotated_spline.get_track('tangent')[-1][1]) + '\n')
                animation = model.generate_motion(annotated_spline, dataset[default_subject]['Idle001_1_d0'], True)
                for i in range(len(animation)):
                    my_agent.cur_anim.append(animation[i])
                if len(my_agent.cur_anim) > 30:
                    my_agent.cur_anim = my_agent.cur_anim[-30:]
            add_point_flag = True
        time.sleep(agent.delta_time)# sleep 30 ms

if __name__ == '__main__':

    my_agent = agent(0.03)
    t1 = threading.Thread(target=keyboard_control, args=(my_agent,))
    t1.start()

    ## animation configuration set up
    x = 0
    y = 1
    z = 2
    radius = torch.max(dataset.skeleton().offsets()).item() * 2 # Heuristic that works well with many skeletons
    skeleton_parents = dataset.skeleton().parents()
    plt.ion()
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20., azim=30)

    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.set_aspect('auto')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5
    initialized = False
    traj_initialized = False
    lines = []

    while not my_agent.quit:
        if len(my_agent.cur_anim) > 0:
            data = np.array(my_agent.cur_anim).copy()
            trajectory = data[:, 0, [0, 2]].copy()
            avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
            draw_offset = int(1/avg_segment_length)
            if not traj_initialized:
                if trajectory.shape[0] == 30:
                    spline_line, = ax.plot(trajectory[:,0], trajectory[:,0] * 0.0, trajectory[:,1])
                    traj_initialized = True
            height_offset = np.min(data[:, :, 1]) # Min height
            data = data.copy()
            data[:, :, 1] -= height_offset
            frame = len(data) - 1
            positions_world = data[frame]
            for i in range(positions_world.shape[0]):
                if skeleton_parents[i] == -1:
                    continue
                if not initialized:
                    col = 'red' if i in dataset.skeleton().joints_right() else 'black' # As in audio cables :)
                    lines.append(ax.plot([positions_world[i, x], positions_world[skeleton_parents[i], x]],
                            [positions_world[i, y], positions_world[skeleton_parents[i], y]],
                            [positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y', c=col))
                else:
                    lines[i-1][0].set_xdata(np.array([positions_world[i, x], positions_world[skeleton_parents[i], x]]))
                    lines[i-1][0].set_ydata(np.array([positions_world[i, y], positions_world[skeleton_parents[i], y]]))
                    lines[i-1][0].set_3d_properties([positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y')
            if traj_initialized:
                if trajectory.shape[0] == 30:
                    spline_line.set_xdata(trajectory[:, 0])
                    spline_line.set_ydata(np.zeros_like(trajectory[:, 0]))
                    spline_line.set_3d_properties(trajectory[:, 1], zdir='y')
                    traj_initialized = True
            initialized = True

            xmin = np.min(positions_world[:, 0])
            ymin = np.min(positions_world[:, 2])
            zmin = np.min(positions_world[:, 1])
            xmax = np.max(positions_world[:, 0])
            ymax = np.max(positions_world[:, 2])
            zmax = np.max(positions_world[:, 1])
            scale = np.max([xmax - xmin, ymax - ymin, zmax - zmin]) * 2
            xmid = (xmax + xmin) / 2
            ymid = (ymax + ymin) / 2
            zmid = (zmax + zmin) / 2
            ax.set_xlim3d(xmid - scale / 2, xmid + scale / 2)
            ax.set_ylim3d(ymid - scale / 2, ymid + scale / 2)
            ax.set_zlim3d(zmid - scale / 2, zmid + scale / 2)
            ax.set_aspect('auto')

            plt.draw()
            plt.pause(0.001)
        plt.pause(0.001)
    plt.show()
