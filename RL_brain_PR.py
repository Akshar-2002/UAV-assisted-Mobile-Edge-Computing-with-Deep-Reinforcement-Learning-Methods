"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
from collections import deque
np.random.seed(1)
tf.compat.v1.set_random_seed(1)
n_lstm_step = 10


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Store data with its priority in the tree.
    """
    data_pointer = 0
    seq_no = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity
        self.seq_list = np.zeros(capacity, dtype=int)
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.seq_list[self.data_pointer] = self.seq_no       
        self.update(tree_idx, p)  # update tree_frame

        self.seq_no += 1
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1

        #Modified: Check for corresponding lstm step
        lstm_bool = False
        curr_seq = self.seq_list[data_idx]
        next_seq = self.seq_list[(data_idx + n_lstm_step) % self.capacity]
        if (curr_seq + n_lstm_step) == next_seq:
            lstm_bool = True

        return leaf_idx, self.tree[leaf_idx], self.data[data_idx], lstm_bool

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        # pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            # a, b = pri_seg * i, pri_seg * (i + 1)

            while True:
                # v = np.random.uniform(a, b)
                v = np.random.uniform(0, self.tree.total_p)
                idx, p, data, lstm_bool = self.tree.get_leaf(v)
                if lstm_bool:
                    break

            prob = p / self.tree.total_p
            if min_prob == 0:
                ISWeights[i, 0] = np.power(prob/1e-4, -self.beta)
            else:
                ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features,
            n_lstm_features,
            n_time,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=0.00025,
            output_graph=False,
            prioritized=True,
            sess=None,
            N_L1=20,
            N_lstm=20,
            seed=0
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.prioritized = prioritized    # decide to use double q or not
        self.learn_step_counter = 0
        self.N_L1 = N_L1
        self.seed = seed       
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step       # step_size in lstm
        self.n_lstm_state = n_lstm_features
        self._build_net(seed=self.seed)
        t_params = tf.compat.v1.get_collection('target_net_params')
        e_params = tf.compat.v1.get_collection('eval_net_params')
        self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]

        tf.compat.v1.disable_eager_execution()
        
        # Do we need to add LSTM in this?
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        if sess is None:
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)

        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()

        self.cost_his = []
        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for ii in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_q_value = list()

    def _build_net(self, seed=0):
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(seed)
        def build_layers(s,lstm_s, c_names, n_l1,n_lstm, w_initializer, b_initializer, trainable):
            # lstm for load levels
            with tf.compat.v1.variable_scope('l0'):
                lstm_dnn = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm)
                lstm_dnn.zero_state(self.batch_size, tf.float32)
                lstm_output,lstm_state = tf.compat.v1.nn.dynamic_rnn(lstm_dnn, lstm_s, dtype=tf.float32)
                lstm_output_reduced = tf.reshape(lstm_output[:, -1, :], shape=[-1, n_lstm])

            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [n_lstm + self.n_features, n_l1], initializer=w_initializer, collections=c_names, trainable=trainable, use_resource=False)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable, use_resource=False)
                l1 = tf.nn.relu(tf.matmul(tf.concat([lstm_output_reduced, s],1), w1) + b1)

            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,  trainable=trainable, use_resource=False)
                b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names,  trainable=trainable, use_resource=False)
                out = tf.matmul(l1, w2) + b2
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s') 
        self.lstm_s = tf.compat.v1.placeholder(tf.float32,[None,self.n_lstm_step,self.n_lstm_state], name='lstm1_s') 
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.compat.v1.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.compat.v1.variable_scope('eval_net'):
            c_names, n_l1,n_lstm, w_initializer, b_initializer = \
                ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], self.N_L1, self.N_lstm,\
                tf.compat.v1.random_normal_initializer(0., 0.3), tf.compat.v1.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s,self.lstm_s, c_names, n_l1,n_lstm, w_initializer, b_initializer, True)

        with tf.compat.v1.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.math.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval))
        with tf.compat.v1.variable_scope('train'):
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_') 
        self.lstm_s_ = tf.compat.v1.placeholder(tf.float32,[None,self.n_lstm_step,self.n_lstm_state], name='lstm1_s_')   # input
        with tf.compat.v1.variable_scope('target_net'):
            c_names = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, self.lstm_s_, c_names, n_l1, n_lstm, w_initializer, b_initializer, False)

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def update_lstm(self, lstm_s):

        self.lstm_history.append(lstm_s)

    def choose_action(self, observation, inference=False):
        observation = observation[np.newaxis, :]
        if inference or np.random.uniform() < self.epsilon:

            # lstm only contains history, there is no current observation
            lstm_observation = np.array(self.lstm_history)

            actions_value = self.sess.run(self.q_eval,
                                          feed_dict={self.s: observation,
                                                     self.lstm_s: lstm_observation.reshape(1, self.n_lstm_step,
                                                                                           self.n_lstm_state),
                                                     })
            self.store_q_value.append({'observation': observation, 'q_value': actions_value})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def choose_action1(self, observation, inference=False):
        action = 4
        return action


    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
          
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])
        
        for ii in range(self.batch_size):
            data_idx = tree_idx[ii] - self.memory.tree.capacity + 1
            for jj in range(self.n_lstm_step):
                lstm_batch_memory[ii,jj,:] = self.memory.tree.data[(data_idx+jj) % self.memory.tree.capacity][self.n_features+1+1+self.n_features:]
        
        # for ii in range(len(sample_index)):
        #     for jj in range(self.n_lstm_step):
        #         # Self.Memory not subscriptable, find another way to access it 
        #         lstm_batch_memory[ii,jj,:] = self.memory[sample_index[ii]+jj,
        #                                       self.n_features+1+1+self.n_features:]


        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],  # output
            feed_dict={
                # [s, a, r, s_]
                # input for target_q (last)
                self.s_: batch_memory[:, -self.n_features:], self.lstm_s_: lstm_batch_memory[:,:,self.n_lstm_state:],
                # input for eval_q (last)
                self.s: batch_memory[:, -self.n_features:], self.lstm_s: lstm_batch_memory[:,:,self.n_lstm_state:],
            }
        )

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.lstm_s: lstm_batch_memory[:, :, :self.n_lstm_state],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.lstm_s: lstm_batch_memory[:, :, :self.n_lstm_state],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

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