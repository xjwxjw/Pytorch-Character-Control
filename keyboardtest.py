import keyboard  # using module keyboard
import time
import numpy as np

class agent():
    def __init__(self, delta_time):
        self.delta_time = delta_time
        self.cur_vel = np.array([0.0, 0.0])
        self.max_vel_norm = 100.0
        self.fric_facotr = 500.0
        self.acc_factor = 1000.0
        # self.cur_acc = np.array([0.0, 0.0])
        # self.max_acc_norm = 100.0
    
    def calcvelocity(self, key):
        tmp_acc = np.array([0.0, 0.0])
        if key == 'w':
            tmp_acc += np.array([0.0, self.acc_factor])
        if key == 's':
            tmp_acc += np.array([0.0, -self.acc_factor])
        if key == 'a':
            tmp_acc += np.array([self.acc_factor, 0.0])
        if key == 'd':
            tmp_acc += np.array([-self.acc_factor, 0.0])
        
        vel_norm = np.linalg.norm(self.cur_vel, 2) + 1e-9
        if vel_norm > 20:
            cur_fric = -(self.cur_vel / vel_norm) * self.fric_facotr
        elif vel_norm > 5:
            cur_fric = -self.cur_vel * 30
        else:
            cur_fric = -self.cur_vel / 0.03
        tmp_acc += cur_fric

        ## pre speed check
        pre_cal_vel = tmp_acc * self.delta_time + self.cur_vel
        pre_cal_vel_norm = np.linalg.norm(pre_cal_vel, 2)
        if pre_cal_vel_norm > self.max_vel_norm:
            tmp_acc = 0
        self.cur_vel += tmp_acc * self.delta_time
        

            
        
        
if __name__ == "__main__":
    my_agent = agent(0.03)

    while True:  # making a loop
        cur_key = ''
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                print('Quit this program!')
                break  # finishing the loop
            if keyboard.is_pressed('a'):  # if key 'a' is pressed 
                cur_key = 'a'
            if keyboard.is_pressed('w'):  # if key 'w' is pressed 
                cur_key = 'w'
            if keyboard.is_pressed('s'):  # if key 's' is pressed 
                cur_key = 's'
            if keyboard.is_pressed('d'):  # if key 'd' is pressed 
                cur_key = 'd'
            my_agent.calcvelocity(cur_key)
            print(my_agent.cur_vel)
        except:
            break  # if user pressed a key other than the given key the loop will break
            
        time.sleep(0.03)# sleep 30 ms
