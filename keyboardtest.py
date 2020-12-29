import keyboard  # using module keyboard
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import os

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
        self.quit = False
        self.location = np.array([0.0, 0.0])
        self.visual_len = 30
        self.his_traj = np.array([np.array([0., 0.]) for i in range(self.visual_len)])
        self.vel_s0 = 20
        self.vel_s1 = 5

        self.orientation = 0.0
        self.cur_ang_vel = 0.0
        self.ang_fric_facotr0 = 500.0
        self.ang_fric_facotr1 = 10.0
        self.ang_vel_s0 = 72
        self.ang_vel_s1 = 18
        self.ang_acc_factor = 1000.0
        self.max_ang_vel = 100.0
        self.his_ang_traj = np.array([np.array([0., 0.]) for i in range(self.visual_len)])
        
        # self.cur_acc = np.array([0.0, 0.0])
        # self.max_acc_norm = 100.0

    def calc_angular_velocity(self, key):
        tmp_acc = 0.0
        tmp_fric = 0.0
        if key is None:
            ## the speed of this character should slow down to 0.
            ## now there is only friction
            if self.cur_ang_vel > self.ang_vel_s0:
                tmp_fric += - self.ang_fric_facotr0
            elif self.cur_ang_vel > self.ang_vel_s1:
                tmp_fric += - self.cur_ang_vel * self.ang_fric_facotr1
            else:
                tmp_fric += - self.cur_ang_vel / self.delta_time
            
            ## pre speed check
            total_acc = tmp_acc + tmp_fric
            self.cur_ang_vel += total_acc * self.delta_time
        else:
            ## now we should calculate the acc from the input
            tmp_acc = 0
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
                    
            self.cur_ang_vel = ang_diff * 10.0

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
    while not agent.quit:  # making a loop
        try:  # used try so that if user pressed other than the given key error will not be shown
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
            my_agent.calc_linear_velocity(cur_key)
            my_agent.location += my_agent.delta_time * my_agent.cur_vel 
            # print('before', np.array(my_agent.his_traj)[:,0])
            my_agent.his_traj[:-1] = my_agent.his_traj[1:].copy()
            my_agent.his_traj[-1] = my_agent.location
            # print('after', np.array(my_agent.his_traj)[:,0])

            my_agent.calc_angular_velocity(cur_key)
            my_agent.orientation += my_agent.delta_time * my_agent.cur_ang_vel
            my_agent.orientation %= 360.0
            my_agent.his_ang_traj[:-1] = my_agent.his_ang_traj[1:].copy()
            my_agent.his_ang_traj[-1] = np.array([10.0 * np.cos(my_agent.orientation / 180.0 * np.pi), 10.0 * np.sin(my_agent.orientation / 180.0 * np.pi)])
            print(my_agent.orientation) 
        except:
            break  # if user pressed a key other than the given key the loop will break
        time.sleep(agent.delta_time)# sleep 30 ms
    

if __name__ == "__main__":
    my_agent = agent(0.03)
    t1 = threading.Thread(target=keyboard_control, args=(my_agent,))
    # t2 = threading.Thread(target=display_screen, args=(my_agent,))
    t1.start()
    # t2.start()

    to_plot = np.array(my_agent.his_traj)
    to_plot -= to_plot[14:15]
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    my_line, = ax.plot(to_plot[:,0], to_plot[:,1])
    
    my_ang_line = []
    ang_to_plot = np.array(my_agent.his_ang_traj)
    for i in range(my_agent.visual_len):
        xdata = np.array([to_plot[i,0], to_plot[i,0] + ang_to_plot[i,0]])
        ydata = np.array([to_plot[i,1], to_plot[i,1] + ang_to_plot[i,1]])
        ang_line, = ax.plot(xdata, ydata, 'g')
        my_ang_line.append(ang_line)
    
    while not my_agent.quit:
        try:
            # print(my_agent.cur_vel, np.linalg.norm(my_agent.cur_vel))
            to_plot = np.array(my_agent.his_traj)
            to_plot -= to_plot[-1:]
            my_line.set_xdata(to_plot[:,0])
            my_line.set_ydata(to_plot[:,1])
            
            ang_to_plot = np.array(my_agent.his_ang_traj)
            for i in range(my_agent.visual_len):
                xdata = np.array([to_plot[i,0], to_plot[i,0] + ang_to_plot[i,0]])
                ydata = np.array([to_plot[i,1], to_plot[i,1] + ang_to_plot[i,1]])
                my_ang_line[i].set_xdata(xdata)
                my_ang_line[i].set_ydata(ydata)

            my_line.axes.set_xlim([-50.0, 50.0])
            my_line.axes.set_ylim([-50.0, 50.0])
            plt.draw()
            plt.pause(0.001)
        except:
            break
        # time.sleep(my_agent.delta_time)# sleep 30 ms
    plt.show()
    # os.system('\n')

    
