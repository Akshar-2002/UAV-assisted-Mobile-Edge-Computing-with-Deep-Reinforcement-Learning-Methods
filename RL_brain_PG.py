"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
from collections import deque

# reproducible
np.random.seed(1)
tf.compat.v1.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            n_time,
            n_lstm_features,
            learning_rate=0.01,
            reward_decay=0.95,
            memory_size=500,
            batch_size=32,
            n_lstm_step=10,
            N_lstm=20,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size 
        self.n_time = n_time
        # lstm
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step       # step_size in lstm
        self.n_lstm_state = n_lstm_features  # [fog1, fog2, ...., fogn, M_n(t)]

        self.ep_obs = np.zeros((memory_size, self.n_features))
        self.ep_obs_lstm = np.zeros((memory_size, self.n_lstm_state))
        self.ep_obs_ = np.zeros((memory_size, self.n_features))
        self.ep_obs_lstm_ = np.zeros((memory_size, self.n_lstm_state))
        self.ep_as = np.zeros(memory_size)
        self.ep_rs = np.zeros(memory_size)

        self._build_net()

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()

        # print('PG_v2 n_lstm_state dim: ')
        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for ii in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_prob_weights = list()

    def _build_net(self):

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.name_scope('inputs'):
            self.tf_obs = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_obs_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name="observations_")
            self.tf_obs_lstm = tf.compat.v1.placeholder(tf.float32, [None, self.n_lstm_step, self.n_lstm_state], name="lstm_observations")
            self.tf_obs_lstm_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_lstm_step, self.n_lstm_state], name="lstm_observations_")
            self.tf_acts = tf.compat.v1.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.compat.v1.placeholder(tf.float32, [None, ], name="actions_value")

        #lstm 
        # Define your LSTM cell and operations here
        with tf.compat.v1.variable_scope('lstm', reuse=tf.compat.v1.AUTO_REUSE):
            lstm_dnn = tf.compat.v1.nn.rnn_cell.LSTMCell(self.N_lstm)
            lstm_dnn.zero_state(1, tf.float32)
            lstm_output, lstm_state = tf.compat.v1.nn.dynamic_rnn(lstm_dnn, self.tf_obs_lstm, dtype=tf.float32)
            lstm_output_reduced = tf.reshape(lstm_output[-1, :], shape=[self.N_lstm*10])   

        #lstm_
        # Define your LSTM_ cell and operations here
        lstm_dnn_ = tf.compat.v1.nn.rnn_cell.LSTMCell(self.N_lstm)
        lstm_dnn_.zero_state(1, tf.float32)
        lstm_output_, lstm_state_ = tf.compat.v1.nn.dynamic_rnn(lstm_dnn_, self.tf_obs_lstm_, dtype=tf.float32)
        lstm_output_reduced_ = tf.reshape(lstm_output_[-1, :], shape=[self.N_lstm*10])   
       
        exp_tensor = tf.convert_to_tensor(lstm_output_reduced, dtype=tf.float32)
        expanded_tensor = tf.reshape(exp_tensor, shape=[1, 200])  
        exp_tensor_ = tf.convert_to_tensor(lstm_output_reduced_, dtype=tf.float32)
        expanded_tensor_ = tf.reshape(exp_tensor_, shape=[1, 200])  

        lstm_or = tf.tile(expanded_tensor, [tf.shape(self.tf_obs)[0], 1])
        lstm_or_ = tf.tile(expanded_tensor_, [tf.shape(self.tf_obs_)[0], 1])

        concat_lstm = tf.concat([lstm_or, lstm_or_], axis=1)
        concat_ann = tf.concat([self.tf_obs, self.tf_obs_], axis=1)
        concat_input = tf.concat([concat_ann, concat_lstm], axis=1)

        # print(f'\nlstm_or shape: {lstm_or.shape}\n')
        # print(f'\nself.tf_obs shape: {self.tf_obs.shape}\n')
        # print(f'\nexpanded_tensor shape: {expanded_tensor.shape}\n')
        # print(f'\n\nCONCAT SHAPE: {concat_input.shape}\n\n')

        # fc1
        layer = tf.compat.v1.layers.dense(
            inputs=concat_input,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.compat.v1.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.compat.v1.constant_initializer(0.1),
            name='fc1'
        )    
        # fc2
        all_act = tf.compat.v1.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.compat.v1.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.compat.v1.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.compat.v1.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.compat.v1.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        lstm_observation = np.array(self.lstm_history)
        observation_ = np.zeros_like(observation)
        lstm_observation_ = np.zeros_like(lstm_observation)
        # print(f'\n\nLSTM OBS SHAPE: {lstm_observation.shape}')
        # prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :], self.tf_obs_lstm: lstm_observation.reshape(self.n_lstm_step,
        #                                                                                    self.n_lstm_state),
        #                                                            })
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :], 
                                                                   self.tf_obs_: observation_[np.newaxis, :],
                                                                   self.tf_obs_lstm: lstm_observation[np.newaxis, :, :],
                                                                   self.tf_obs_lstm_: lstm_observation_[np.newaxis, :, :],
                                                                   })
        
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        
        self.store_prob_weights.append({'observation': observation, 'q_value': prob_weights})

        return action
    
    def choose_action1(self, observation):
        action = 0
        return action

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        index = self.memory_counter % self.memory_size
        self.ep_obs[index, :] = s
        self.ep_obs_lstm[index, :] = lstm_s
        self.ep_obs_[index, :] = s_
        self.ep_obs_lstm_[index, :] = lstm_s_
        self.ep_as[index] = a
        self.ep_rs[index] = r

        self.memory_counter += 1

    def update_lstm(self, lstm_s):

        self.lstm_history.append(lstm_s)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # randomly pick [batch_size] memory from memory np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)

        batch_memory = self.ep_obs[sample_index, :]
        batch_memory_ = self.ep_obs_[sample_index, :]

        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state])
        for ii in range(len(sample_index)):
            for jj in range(self.n_lstm_step):
                lstm_batch_memory[ii,jj,:] = self.ep_obs_lstm[sample_index[ii]+jj, :]

        lstm_batch_memory_ = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state])
        for ii in range(len(sample_index)):
            for jj in range(self.n_lstm_step):
                lstm_batch_memory_[ii,jj,:] = self.ep_obs_lstm_[sample_index[ii]+jj, :]

        action_memory = self.ep_as[sample_index]
        reward_memory = discounted_ep_rs_norm[sample_index]

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: batch_memory,  # shape=[None, n_obs]
             self.tf_obs_lstm: lstm_batch_memory,
             self.tf_obs_: batch_memory_,  # shape=[None, n_obs]
             self.tf_obs_lstm_: lstm_batch_memory_,
             self.tf_acts: action_memory,  # shape=[None, ]
             self.tf_vt: reward_memory,  # shape=[None, ]
        })

        # return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
    
    def do_store_reward(self, episode, time, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self,episode,time, action):
        while episode >= len(self.action_store):
            self.action_store.append(- np.ones([self.n_time]))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay

    def do_store_energy(self, episode, time, energy):
        while episode >= len(self.energy_store):
            self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time] = energy



