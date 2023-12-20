import numpy as np
import random
import math
import queue


class Offload:

    def __init__(self, num_iot, num_fog, num_time, max_delay, task_arrive_prob):

        # Number of IoT and Fogs       
        self.n_iot = num_iot # M, number of iot
        self.n_fog = num_fog #N, number of fog

        # Timeslots
        self.max_delay = max_delay # \tau, deadline time slots
        self.n_time = num_time # T, No. of timeslots in one episode
        self.duration = 0.1 # \delta, time of 1 timeslot

        # Drop count
        self.drop_trans_count = 0
        self.drop_fog_count = 0
        self.drop_iot_count = 0

        # Transmission rate calculation between IoT and UAV
        self.height = self.ground_length = self.ground_width = 100
        self.bandwidth_nums = 2
        self.B =  self.bandwidth_nums * 10 ** 6
        self.p_noisy_los = 10 ** (-13)
        self.p_noisy_nlos = 10 ** (-11)
        self.r = 10 ** (-27)
        self.p_uplink = 0.1
        self.alpha0 = 1e-5
        self.t_move = 0.02
        self.v_ue = 1
        #self.block_flag_list = np.random.randint(0, 2, self.n_iot)

        # Location of UE and UAV
        self.loc_ue_list = np.random.randint(0, 101, size=[2 * self.n_iot])     #Coordinates of ith iot are in indices 2*iot and 2*(iot+1)
        self.loc_uav_list = np.random.randint(0, 101, size=[2 * self.n_fog])

        # Computation capacity 
        self.comp_cap_iot = 2.5 * np.ones(self.n_iot) * self.duration  # 1 Gigacycles per second  * duration
        self.comp_cap_fog =  15 * np.ones([self.n_fog]) * self.duration  # Gigacycles per second * duration
        self.comp_cap_fog[0] = 20 * self.duration

        # Transmission rate between IoT and Satellite
        self.tran_cap_sat = 14  * self.duration # 8 Mbps * duration
        self.propagation_sat = 0.2 # 2 timeslots for propagation

	######################################################## 25 June
        self.p_fog = 0.1 * np.ones([self.n_fog])
        self.p_fog[0] = 0.3
        self.coeff = 10 ** -27
        self.w1 = 0.5
        self.w2 = 0.5
        
        # Task size
        self.task_arrive_prob = task_arrive_prob
        self.max_bit_arrive = 5 # Mbits
        self.min_bit_arrive = 2 # Mbits
        self.bitArrive_set = np.arange(self.min_bit_arrive, self.max_bit_arrive, 0.1)
        self.bitArrive = np.zeros([self.n_time, self.n_iot])

        #Computation density
        self.comp_density = 0.297 * np.ones([self.n_iot])  # 0.297 Gigacycles per Mbits

        # ACTION: localand fogs
        self.n_actions = 1 + num_fog

        # STATE: [p_m_x, p_m_y, q_1 to q_n - both x and y, A, t^{comp}, t^{tran}, [B^{fog}]]
        self.n_features = 2 + (2 * self.n_fog) + 1 + 1 + 1 + num_fog

        # LSTM STATE
        self.n_lstm_state = self.n_fog  # [fog1, fog2, ...., fogn]
        # print(f'\nFOG ENV n_lstm_state dim: {self.n_lstm_state}')

        # TIME COUNT
        self.time_count = int(0)

        # QUEUE INITIALIZATION: size -> task size; time -> arrive time
        #Each iot has 1 comp queue, 1 tran queue and 1 queue at all n fogs
        self.Queue_iot_comp = list()
        self.Queue_iot_tran = list()
        self.Queue_fog_comp = list()

        for iot in range(self.n_iot):
            self.Queue_iot_comp.append(queue.Queue())
            self.Queue_iot_tran.append(queue.Queue())
            self.Queue_fog_comp.append(list())
            for fog in range(self.n_fog):
                self.Queue_fog_comp[iot].append(queue.Queue())

        # QUEUE INFO INITIALIZATION
        self.t_iot_comp = - np.ones([self.n_iot]) #l^comp
        self.t_iot_tran = - np.ones([self.n_iot]) #l^tran
        self.b_fog_comp = np.zeros([self.n_iot, self.n_fog]) # q^edge

        #self.t_iot_comp_energy = - np.ones([self.n_iot]) #l^comp
        #self.t_iot_tran_energy = - np.ones([self.n_iot]) #l^tran

        # TASK INDICATOR
        self.task_on_process_local = list()
        self.task_on_transmit_local = list()
        self.task_on_process_fog = list()
        self.fog_iot_m = np.zeros(self.n_fog) # B_n
        self.fog_iot_m_observe = np.zeros(self.n_fog)

        for iot in range(self.n_iot):
            self.task_on_process_local.append({'size': np.nan, 'time': np.nan, 'remain': np.nan})
            self.task_on_transmit_local.append({'size': np.nan, 'time': np.nan, 'fog': np.nan, 'remain': np.nan})
            self.task_on_process_fog.append(list())
            for fog in range(self.n_fog):
                self.task_on_process_fog[iot].append({'size': np.nan, 'time': np.nan, 'remain': np.nan, 'energy':np.nan})

        # TASK DELAY
        self.process_delay = np.zeros([self.n_time, self.n_iot])    # total delay
        self.process_delay_unfinish_ind = np.zeros([self.n_time, self.n_iot])  # unfinished indicator
        self.process_delay_trans = np.zeros([self.n_time, self.n_iot])  # transmission delay (if applied)

        #################################################################################################################3 25the June - TASK Energy
        self.process_energy = np.zeros([self.n_time, self.n_iot])    # total energy 
        self.process_energy_trans = np.zeros([self.n_time, self.n_iot])  # transmission energy (if applied)

        self.fog_drop = np.zeros([self.n_iot, self.n_fog])

    # reset the network scenario
    def reset(self, bitArrive):

        # test
        self.drop_trans_count = 0
        self.drop_fog_count = 0
        self.drop_iot_count = 0

        # Location of UEs
        self.loc_ue_list = np.random.randint(0, 101, size=[2 * self.n_iot])

        # BITRATE
        self.bitArrive = bitArrive

        # TIME COUNT
        self.time_count = int(0)

        # QUEUE INITIALIZATION
        self.Queue_iot_comp = list()
        self.Queue_iot_tran = list()
        self.Queue_fog_comp = list()

        for iot in range(self.n_iot):
            self.Queue_iot_comp.append(queue.Queue())
            self.Queue_iot_tran.append(queue.Queue())
            self.Queue_fog_comp.append(list())
            for fog in range(self.n_fog):
                self.Queue_fog_comp[iot].append(queue.Queue())

        # QUEUE INFO INITIALIZATION
        self.t_iot_comp = - np.ones([self.n_iot])
        self.t_iot_tran = - np.ones([self.n_iot])
        self.b_fog_comp = np.zeros([self.n_iot, self.n_fog])

        #self.t_iot_comp_energy = - np.ones([self.n_iot])
        #self.t_iot_tran_energy = - np.ones([self.n_iot])

        # TASK INDICATOR
        self.task_on_process_local = list()
        self.task_on_transmit_local = list()
        self.task_on_process_fog = list()

        for iot in range(self.n_iot):
            self.task_on_process_local.append({'size': np.nan, 'time': np.nan, 'remain': np.nan})
            self.task_on_transmit_local.append({'size': np.nan, 'time': np.nan, 'fog': np.nan, 'remain': np.nan})
            self.task_on_process_fog.append(list())
            for fog in range(self.n_fog):
                self.task_on_process_fog[iot].append({'size': np.nan, 'time': np.nan, 'remain': np.nan,'energy':np.nan})

        # TASK DELAY
        self.process_delay = np.zeros([self.n_time, self.n_iot])
        self.process_delay_unfinish_ind = np.zeros([self.n_time, self.n_iot])  # unfinished indicator
        self.process_delay_trans = np.zeros([self.n_time, self.n_iot])  # transmission delay (if applied)

#################################################################################################################3 25the June - TASK Energy
        self.process_energy = np.zeros([self.n_time, self.n_iot])    # total energy
        self.process_energy_trans = np.zeros([self.n_time, self.n_iot])  # transmission energy (if applied)

        self.fog_drop = np.zeros([self.n_iot, self.n_fog])

        # INITIAL
        observation_all = np.zeros([self.n_iot, self.n_features])
        for iot_index in range(self.n_iot):
            if self.bitArrive[self.time_count, iot_index] != 0:
                observation_all[iot_index, :] = np.hstack([
                    self.loc_ue_list[2 * iot_index], self.loc_ue_list[(2 * iot_index) + 1], 
                    self.loc_uav_list,
                    #self.t_iot_comp_energy[iot_index],
                    #self.t_iot_tran_energy[iot_index],
                    self.bitArrive[self.time_count, iot_index],
                    self.t_iot_comp[iot_index],
                    self.t_iot_tran[iot_index],
                    np.squeeze(self.b_fog_comp[iot_index, :])])
        ####################### new concept added ##############

        lstm_state_all = np.zeros([self.n_iot, self.n_lstm_state])

        return observation_all, lstm_state_all

        ####################### new concept added ##############
    def calc_tran(self, fog_cur, iot_ind):
      dx = self.loc_uav_list[2 * fog_cur] - self.loc_ue_list[2 * iot_ind]
      dy = self.loc_uav_list[(2 * fog_cur) + 1] - self.loc_ue_list[(2 * iot_ind) + 1]
      dh = self.height
      dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
      p_noise = self.p_noisy_los
      #if self.block_flag_list[iot_ind] == 1:
      #  p_noise = self.p_noisy_nlos
      g_uav_ue = abs(self.alpha0 / (dist_uav_ue) ** 2)
      iot_tran_cap = self.duration* self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)* 1e-6
      return iot_tran_cap
        ####################### new concept added ##############

    # perform action, observe state and delay (several steps later)
    def step(self, action):

        # EXTRACT ACTION FOR EACH IOT
        iot_action_local = np.zeros([self.n_iot], np.int32)
        iot_action_fog = np.zeros([self.n_iot], np.int32)
        for iot_index in range(self.n_iot):
            iot_action = action[iot_index]
            iot_action_fog[iot_index] = int(iot_action - 1)
            if iot_action == 0:
                iot_action_local[iot_index] = 1

            #file_name = 'Local_Action.txt'
            #with open(file_name, 'a') as file_obj:
            #  file_obj.write("\nUE-" + '{:f}'.format(iot_index) + "Timecount-" + '{:f}'.format(self.time_count) + ", Action action:" + '{:f}'.format(iot_action) + ", Edge:" + '{:f}'.format(iot_action_fog[iot_index]) )
              #file_obj.write("\nUE-" + '{:f}'.format(iot_index) + ", Local:" + '{:.3f}'.format(iot_action_local[iot_index]) + ", Edge:" + '{:.3f}'.format(iot_action_fog[iot_index]) )
        # COMPUTATION QUEUE UPDATE ===================
        for iot_index in range(self.n_iot):

            iot_bitarrive = np.squeeze(self.bitArrive[self.time_count, iot_index])
            iot_comp_cap = np.squeeze(self.comp_cap_iot[iot_index])
            iot_comp_density = self.comp_density[iot_index]

            # INPUT
            if iot_action_local[iot_index] == 1:
                tmp_dict = {'size': iot_bitarrive, 'time': self.time_count}
                self.Queue_iot_comp[iot_index].put(tmp_dict)

            # TASK ON PROCESS
            if math.isnan(self.task_on_process_local[iot_index]['remain']) \
                    and (not self.Queue_iot_comp[iot_index].empty()):
                while not self.Queue_iot_comp[iot_index].empty():
                    # only put the non-zero task to the processor
                    get_task = self.Queue_iot_comp[iot_index].get()
                    # since it is at the beginning of the time slot, = self.max_delay is acceptable
                    if get_task['size'] != 0:
                        if self.time_count - get_task['time'] + 1 <= self.max_delay:
                            self.task_on_process_local[iot_index]['size'] = get_task['size']
                            self.task_on_process_local[iot_index]['time'] = get_task['time']
                            self.task_on_process_local[iot_index]['remain'] \
                                = self.task_on_process_local[iot_index]['size']
                            break
                        else:
                            self.process_delay[get_task['time'], iot_index] = self.max_delay
                            self.process_energy[get_task['time'], iot_index] = self.max_delay * self.coeff * ((iot_comp_cap * (10 ** 9)) ** 3)
                            self.process_delay_unfinish_ind[get_task['time'], iot_index] = 1

            # PROCESS
            if self.task_on_process_local[iot_index]['remain'] > 0:
                self.task_on_process_local[iot_index]['remain'] = self.task_on_process_local[iot_index]['remain'] - iot_comp_cap / iot_comp_density
                # if no remain, compute processing delay
                if self.task_on_process_local[iot_index]['remain'] <= 0:
                    self.process_delay[self.task_on_process_local[iot_index]['time'], iot_index] =  (self.time_count - self.task_on_process_local[iot_index]['time'] + 1) 
                    self.process_energy[self.task_on_process_local[iot_index]['time'], iot_index] = (self.coeff * (((iot_comp_cap/self.duration) * (10 ** 9)) ** 2) * (iot_comp_density * (10 ** 9)) * self.task_on_process_local[iot_index]['size'] ) * (10**-2)
                    self.task_on_process_local[iot_index]['remain'] = np.nan
                    #file_name = 'Local_Delay,energy_Processed.txt'
                    #with open(file_name, 'a') as file_obj:
                    #    file_obj.write("\n UE-" + '{:f}'.format(iot_index) + "Time_arrived:" + '{:.3f}'.format(self.task_on_process_local[iot_index]['time']) +  "Time_processed:" + '{:.3f}'.format(self.time_count) +", Tasksize:" + '{:.3f}'.format(self.task_on_process_local[iot_index]['size']  ) + ", Delay:" + '{:.3f}'.format(self.process_delay[self.task_on_process_local[iot_index]['time'], iot_index] ) + ", Energy:" + '{:.3f}'.format(self.process_energy[self.task_on_process_local[iot_index]['time'], iot_index]) )
  

                elif self.time_count - self.task_on_process_local[iot_index]['time'] + 1 == self.max_delay:
                    self.process_delay[self.task_on_process_local[iot_index]['time'], iot_index] = self.max_delay
                    self.process_energy[self.task_on_process_local[iot_index]['time'], iot_index] = self.max_delay * self.coeff * ((iot_comp_cap * (10 ** 9))** 3)
                    self.process_delay_unfinish_ind[self.task_on_process_local[iot_index]['time'], iot_index] = 1
                    self.task_on_process_local[iot_index]['remain'] = np.nan
                    self.drop_iot_count = self.drop_iot_count + 1

                    #file_name = 'Local_Delay,energy_Dropped.txt'
                    #with open(file_name, 'a') as file_obj:
                    #  file_obj.write("\nTimecount:" + '{:.3f}'.format(self.time_count) + ", UE-" + '{:f}'.format(iot_index) + ", Delay:" + '{:.3f}'.format(self.process_delay[self.task_on_process_local[iot_index]['time'], iot_index] ) + ", Energy:" + '{:.3f}'.format(self.process_energy[self.task_on_process_local[iot_index]['time'], iot_index]) )


            if iot_bitarrive != 0:
                tmp_tilde_t_iot_comp = np.max([self.t_iot_comp[iot_index] + 1, self.time_count])
                self.t_iot_comp[iot_index] = np.min([tmp_tilde_t_iot_comp
                                                    + math.ceil(iot_bitarrive * iot_action_local[iot_index]
                                                     / (iot_comp_cap / iot_comp_density)) - 1,
                                                    self.time_count + self.max_delay - 1])

                #self.t_iot_comp_energy[iot_index] = self.coeff * (((iot_comp_cap/self.duration) * (10 ** 9)) ** 2) * (iot_comp_density * (10 ** 9)) * iot_bitarrive

        # FOG QUEUE UPDATE =========================
        for iot_index in range(self.n_iot):

            iot_comp_density = self.comp_density[iot_index]

            for fog_index in range(self.n_fog):

                # TASK ON PROCESS
                if math.isnan(self.task_on_process_fog[iot_index][fog_index]['remain']) \
                        and (not self.Queue_fog_comp[iot_index][fog_index].empty()):
                    while not self.Queue_fog_comp[iot_index][fog_index].empty():
                        get_task = self.Queue_fog_comp[iot_index][fog_index].get()
                        if self.time_count - get_task['time'] + 1 <= self.max_delay:
                            self.task_on_process_fog[iot_index][fog_index]['size'] \
                                = get_task['size']
                            self.task_on_process_fog[iot_index][fog_index]['time'] \
                                = get_task['time']
                            self.task_on_process_fog[iot_index][fog_index]['remain'] \
                                = self.task_on_process_fog[iot_index][fog_index]['size']
                            self.task_on_process_fog[iot_index][fog_index]['energy']\
                                = get_task['energy']
                            break
                        else:
                            self.process_delay[get_task['time'], iot_index] = self.max_delay 
                            self.process_energy[get_task['time'], iot_index] =  self.max_delay *  self.coeff * ((self.comp_cap_fog[fog_index] * (10 ** 9)) ** 3)                       
                            self.process_delay_unfinish_ind[get_task['time'], iot_index] = 1

                # PROCESS
                self.fog_drop[iot_index, fog_index] = 0
                if self.task_on_process_fog[iot_index][fog_index]['remain'] > 0:
                    self.task_on_process_fog[iot_index][fog_index]['remain'] = self.task_on_process_fog[iot_index][fog_index]['remain'] \
                        - self.comp_cap_fog[fog_index] / iot_comp_density / self.fog_iot_m[fog_index]
                    
                    if self.task_on_process_fog[iot_index][fog_index]['remain'] <= 0:
                        self.process_delay[self.task_on_process_fog[iot_index][fog_index]['time'],iot_index] \
                            = (self.time_count - self.task_on_process_fog[iot_index][fog_index]['time'] + 1)
                        self.task_on_process_fog[iot_index][fog_index]['remain'] = np.nan
                        self.process_energy[self.task_on_process_fog[iot_index][fog_index]['time'],iot_index] = (self.task_on_process_fog[iot_index][fog_index]['energy'] + \
                            self.coeff * (((self.comp_cap_fog[fog_index]/self.duration) * (10 ** 9)) ** 2) * (iot_comp_density * (10 ** 9)) * self.task_on_process_fog[iot_index][fog_index]['size']) * (10**-2)

                        #file_name = 'Offload_Delay,energy_Processed.txt'
                        #with open(file_name, 'a') as file_obj:
                        #    file_obj.write("\n UE-" + '{:f}'.format(iot_index) + "Edge-" + '{:f}'.format(fog_index) + "Time_arrived:" + '{:.3f}'.format(self.task_on_process_fog[iot_index][fog_index]['time']) +  "Time_processed:" + '{:.3f}'.format(self.time_count) +", Tasksize:" + '{:.3f}'.format(self.task_on_process_fog[iot_index][fog_index]['size'] ) + ", Delay:" + '{:.3f}'.format(self.process_delay[self.task_on_process_fog[iot_index][fog_index]['time'],iot_index]) )

                    elif self.time_count - self.task_on_process_fog[iot_index][fog_index]['time'] + 1 == self.max_delay:
                        self.process_delay[self.task_on_process_fog[iot_index][fog_index]['time'], iot_index] = self.max_delay
                        self.process_energy[self.task_on_process_fog[iot_index][fog_index]['time'], iot_index] = self.max_delay *  self.coeff * ((self.comp_cap_fog[fog_index] * (10 ** 9)) ** 3) 
                        self.process_delay_unfinish_ind[self.task_on_process_fog[iot_index][fog_index]['time'], iot_index] = 1
                        self.fog_drop[iot_index, fog_index] = self.task_on_process_fog[iot_index][fog_index]['remain']
                        self.task_on_process_fog[iot_index][fog_index]['remain'] = np.nan
                        self.drop_fog_count = self.drop_fog_count + 1

                # OTHER INFO
                if self.fog_iot_m[fog_index] != 0:
                    self.b_fog_comp[iot_index, fog_index] \
                        = np.max([self.b_fog_comp[iot_index, fog_index]
                                  - self.comp_cap_fog[fog_index] / iot_comp_density / self.fog_iot_m[fog_index]
                                  - self.fog_drop[iot_index, fog_index], 0])

        # TRANSMISSION QUEUE UPDATE ===================
        for iot_index in range(self.n_iot):

            #iot_tran_cap = np.squeeze(self.tran_cap_iot[iot_index,:])
            iot_bitarrive = np.squeeze(self.bitArrive[self.time_count, iot_index])

            # INPUT
            if iot_action_local[iot_index] == 0:
                tmp_dict = {'size': self.bitArrive[self.time_count, iot_index], 'time': self.time_count, 'fog': iot_action_fog[iot_index]}
                self.Queue_iot_tran[iot_index].put(tmp_dict)

            # TASK ON PROCESS
            if math.isnan(self.task_on_transmit_local[iot_index]['remain']) \
                    and (not self.Queue_iot_tran[iot_index].empty()):
                while not self.Queue_iot_tran[iot_index].empty():
                    get_task = self.Queue_iot_tran[iot_index].get()
                    if get_task['size'] != 0:
                        if self.time_count - get_task['time'] + 1 <= self.max_delay:
                            self.task_on_transmit_local[iot_index]['size'] = get_task['size']
                            self.task_on_transmit_local[iot_index]['time'] = get_task['time']
                            self.task_on_transmit_local[iot_index]['fog'] = int(get_task['fog'])
                            self.task_on_transmit_local[iot_index]['remain'] = \
                                self.task_on_transmit_local[iot_index]['size']
                            break
                        else:
                            self.process_delay[get_task['time'], iot_index] = self.max_delay
                            self.process_energy[get_task['time'], iot_index] = self.max_delay * 0.1
                            self.process_delay_unfinish_ind[get_task['time'], iot_index] = 1

            # PROCESS
            if self.task_on_transmit_local[iot_index]['remain'] > 0:
              if self.task_on_transmit_local[iot_index]['fog'] == 0:
                self.task_on_transmit_local[iot_index]['remain'] = self.task_on_transmit_local[iot_index]['remain'] - self.tran_cap_sat
              else:
                fog_cur = self.task_on_transmit_local[iot_index]['fog']
                iot_tran_val = self.calc_tran(fog_cur, iot_index)             
                self.task_on_transmit_local[iot_index]['remain'] = self.task_on_transmit_local[iot_index]['remain'] - iot_tran_val

                  # UPDATE FOG QUEUE
              if self.task_on_transmit_local[iot_index]['remain'] <= 0:
                  self.process_delay_trans[self.task_on_transmit_local[iot_index]['time'], iot_index] = (self.time_count - self.task_on_transmit_local[iot_index]['time'] + 1) 
                  if fog_index == 0:
                    self.process_energy_trans[self.task_on_transmit_local[iot_index]['time'], iot_index] = self.p_fog[fog_index] * self.task_on_transmit_local[iot_index]['size'] / (self.tran_cap_sat/self.duration)
                  else:
                    fog_cur = self.task_on_transmit_local[iot_index]['fog']
                    iot_tran_val = self.calc_tran(fog_cur, iot_index)
                    self.process_energy_trans[self.task_on_transmit_local[iot_index]['time'], iot_index] = self.p_fog[fog_cur] * self.task_on_transmit_local[iot_index]['size'] / (iot_tran_val/self.duration)              
                  self.task_on_transmit_local[iot_index]['remain'] = np.nan

                  tmp_dict = {'size': self.task_on_transmit_local[iot_index]['size'],'time': self.task_on_transmit_local[iot_index]['time'], 'energy': self.process_energy_trans[self.task_on_transmit_local[iot_index]['time'], iot_index] }
                  self.Queue_fog_comp[iot_index][self.task_on_transmit_local[iot_index]['fog']].put(tmp_dict)
                  fog_index = self.task_on_transmit_local[iot_index]['fog']
                  self.b_fog_comp[iot_index, fog_index] = self.b_fog_comp[iot_index, fog_index] + self.task_on_transmit_local[iot_index]['size']

                  #file_name = 'Transmission_Delay, energy_processed.txt'
                  #if fog_index == 0:
                  #  with open(file_name, 'a') as file_obj:
                  #      file_obj.write("\n UE-" + '{:f}'.format(iot_index) + ", Edge-" + '{:f}'.format(self.task_on_transmit_local[iot_index]['fog']) + ", Transmission rate per duration-" + '{:f}'.format(self.tran_cap_sat) + "Time_arrived:" + '{:.3f}'.format(self.task_on_transmit_local[iot_index]['time']) +  "Time_sent:" + '{:.3f}'.format(self.time_count) + ", Tasksize:" + '{:.3f}'.format(self.task_on_transmit_local[iot_index]['size']) + ", Delay:" + '{:.3f}'.format(self.process_delay_trans[self.task_on_transmit_local[iot_index]['time'], iot_index]) + ", Energy:" + '{:.3f}'.format(self.process_energy_trans[self.task_on_transmit_local[iot_index]['time'], iot_index]) )
                  #else: 
                  #  with open(file_name, 'a') as file_obj:
                  #      file_obj.write("\nUE-" + '{:f}'.format(iot_index) + ", Edge-" + '{:f}'.format(self.task_on_transmit_local[iot_index]['fog']) + ", Transmission rate per duration-" + '{:f}'.format(iot_tran_val) +  "Time_arrived:" + '{:.3f}'.format(self.task_on_transmit_local[iot_index]['time']) +  "Time_sent:" + '{:.3f}'.format(self.time_count) + ", Tasksize:" + '{:.3f}'.format(self.task_on_transmit_local[iot_index]['size']) + ", Delay:" + '{:.3f}'.format(self.process_delay_trans[self.task_on_transmit_local[iot_index]['time'], iot_index]) + ", Energy:" + '{:.3f}'.format(self.process_energy_trans[self.task_on_transmit_local[iot_index]['time'], iot_index]) )


              elif self.time_count - self.task_on_transmit_local[iot_index]['time'] + 1 == self.max_delay:
                  self.process_delay[self.task_on_transmit_local[iot_index]['time'], iot_index] = self.max_delay
                  self.process_energy[self.task_on_transmit_local[iot_index]['time'], iot_index] = self.max_delay * self.p_fog[fog_index]
                  self.process_delay_trans[self.task_on_transmit_local[iot_index]['time'], iot_index] = self.max_delay
                  self.process_energy_trans[self.task_on_transmit_local[iot_index]['time'], iot_index] = self.max_delay * self.p_fog[fog_index]
                  self.process_delay_unfinish_ind[self.task_on_transmit_local[iot_index]['time'], iot_index] = 1
                  self.task_on_transmit_local[iot_index]['remain'] = np.nan
                  self.drop_trans_count = self.drop_trans_count + 1
                  #file_name = 'Transmission_Delay, energy_Dropped.txt'
                  #with open(file_name, 'a') as file_obj:
                  #    file_obj.write("\nTimecount:" + '{:.3f}'.format(self.time_count) + ", UE-" + '{:f}'.format(iot_index) + ", Delay:" + '{:.3f}'.format(self.process_delay[self.task_on_transmit_local[iot_index]['time'], iot_index] ) + ", Energy:" + '{:.3f}'.format(self.process_energy[self.task_on_transmit_local[iot_index]['time'], iot_index]) )

            # OTHER INFO
            if iot_bitarrive != 0:
                tmp_tilde_t_iot_tran = np.max([self.t_iot_tran[iot_index] + 1, self.time_count])

                fog_cur = iot_action_fog[iot_index]
                iot_tran_val = self.calc_tran(fog_cur, iot_index)

                if iot_action_fog[iot_index] == 0:
                    self.t_iot_tran[iot_index] = np.min([tmp_tilde_t_iot_tran + math.ceil(iot_bitarrive * (1 - iot_action_local[iot_index]) / self.tran_cap_sat) + (2*self.propagation_sat) - 1, self.time_count + self.max_delay - 1])
                    #self.t_iot_tran_energy[iot_index] =  self.p_fog[fog_cur] * iot_bitarrive/self.tran_cap_sat
                    
                else:
                    self.t_iot_tran[iot_index] = np.min([tmp_tilde_t_iot_tran + math.ceil(iot_bitarrive * (1 - iot_action_local[iot_index]) / iot_tran_val) - 1, self.time_count + self.max_delay - 1])
                    #self.t_iot_tran_energy[iot_index] =  self.p_fog[fog_cur] * iot_bitarrive/iot_tran_val

                

            # print('iot_action_fog[iot_index]:', iot_action_fog[iot_index])

        # COMPUTE CONGESTION (FOR NEXT TIME SLOT)
        self.fog_iot_m_observe = self.fog_iot_m
        self.fog_iot_m = np.zeros(self.n_fog)
        for fog_index in range(self.n_fog):
            for iot_index in range(self.n_iot):
                if (not self.Queue_fog_comp[iot_index][fog_index].empty()) \
                        or self.task_on_process_fog[iot_index][fog_index]['remain'] > 0:
                    self.fog_iot_m[fog_index] += 1

        # TIME UPDATE
        self.time_count = self.time_count + 1
        done = False
        if self.time_count >= self.n_time:
            done = True
            # set all the tasks' processing delay and unfinished indicator
            for time_index in range(self.n_time):
                for iot_index in range(self.n_iot):
                    iot_comp_cap = np.squeeze(self.comp_cap_iot[iot_index])
                    if self.process_delay[time_index, iot_index] == 0 and self.bitArrive[time_index, iot_index] != 0:
                        self.process_delay[time_index, iot_index] = (self.time_count - 1) - time_index + 1
                        self.process_energy[time_index, iot_index] = ((self.time_count - 1) - time_index + 1) * self.coeff * ((iot_comp_cap * (10 ** 9)) ** 3)
                        self.process_delay_unfinish_ind[time_index, iot_index] = 1



        ####################### new concept added ##############

        #UPDATE POSITIONS OF IOT
        # self.loc_ue_list = np.random.randint(0, 101, size=[2 * self.n_iot])
        for i in range(self.n_iot):
          tmp = np.random.rand(2)
          theta_ue = tmp[0] * np.pi * 2
          dis_ue = tmp[1] * self.t_move * self.v_ue
          self.loc_ue_list[2 * i] = self.loc_ue_list[2 * i] + math.cos(theta_ue) * dis_ue
          self.loc_ue_list[(2 * i) + 1] = self.loc_ue_list[(2 * i) + 1] + math.sin(theta_ue) * dis_ue
          self.loc_ue_list[2 * i] = np.clip(self.loc_ue_list[2 * i], 0, self.ground_length)
          self.loc_ue_list[(2 * i) + 1] = np.clip(self.loc_ue_list[(2 * i) + 1], 0, self.ground_width)
        ####################### new concept added ##############


        # OBSERVATION
        observation_all_ = np.zeros([self.n_iot, self.n_features])
        lstm_state_all_ = np.zeros([self.n_iot, self.n_lstm_state])
        if not done:
            for iot_index in range(self.n_iot):
                if self.bitArrive[self.time_count, iot_index] != 0:
                    observation_all_[iot_index, :] = np.hstack([
                        self.loc_ue_list[2 * iot_index], self.loc_ue_list[(2 * iot_index) + 1], 
                        self.loc_uav_list,
                        #self.t_iot_comp_energy[iot_index],
                        #self.t_iot_tran_energy[iot_index],
                        self.bitArrive[self.time_count, iot_index],
                        self.t_iot_comp[iot_index] - self.time_count + 1,
                        self.t_iot_tran[iot_index] - self.time_count + 1,
                        self.b_fog_comp[iot_index, :]])
        ####################### new concept added ##############

                lstm_state_all_[iot_index, :] = np.hstack(self.fog_iot_m_observe)

        return observation_all_, lstm_state_all_, done
