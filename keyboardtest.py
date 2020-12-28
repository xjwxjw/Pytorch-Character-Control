import keyboard  # using module keyboard
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import time

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
        self.fric_facotr = 500.0
        self.acc_factor = 1000.0
        self.quit = False
        self.location = np.array([0.0, 0.0])
        self.visual_len = 30
        self.his_traj = np.array([np.array([0., 0.]) for i in range(self.visual_len)])
        # self.cur_acc = np.array([0.0, 0.0])
        # self.max_acc_norm = 100.0
    
    def calcvelocity(self, key):
        tmp_acc = np.array([0.0, 0.0])
        tmp_fric = np.array([0.0, 0.0])
        if key is None:
            ## the speed of this character should slow down to 0.
            ## now there is only friction
            vel_norm = np.linalg.norm(self.cur_vel, 2) + 1e-9
            if vel_norm > 20:
                tmp_fric += -(self.cur_vel / vel_norm) * self.fric_facotr
            elif vel_norm > 5:
                tmp_fric += -self.cur_vel * 10
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
                tmp_fric += -(self.cur_vel / vel_norm) * self.fric_facotr
            elif vel_norm > 5:
                tmp_fric += -self.cur_vel * 10
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
            my_agent.calcvelocity(cur_key)
            my_agent.location += my_agent.delta_time * my_agent.cur_vel 
            print('before', np.array(my_agent.his_traj)[:,0])
            my_agent.his_traj[:-1] = my_agent.his_traj[1:].copy()
            my_agent.his_traj[-1] = my_agent.location
            print('after', np.array(my_agent.his_traj)[:,0])
        except:
            break  # if user pressed a key other than the given key the loop will break
        time.sleep(agent.delta_time)# sleep 30 ms

def display_screen(agent):
    pass
    

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
    
    while not my_agent.quit:
        try:
            # print(my_agent.cur_vel, np.linalg.norm(my_agent.cur_vel))
            to_plot = np.array(my_agent.his_traj)
            
            my_line.set_xdata(to_plot[:,0])
            my_line.set_ydata(to_plot[:,1])
            
            
            my_line.axes.set_xlim([np.min(to_plot[:,0]) - 10.0, np.max(to_plot[:,0]) + 10.0])
            my_line.axes.set_ylim([np.min(to_plot[:,1]) - 10.0, np.max(to_plot[:,1]) + 10.0])
            plt.draw()
            plt.pause(0.001)
        except:
            break
        # time.sleep(my_agent.delta_time)# sleep 30 ms
    plt.show()


    
