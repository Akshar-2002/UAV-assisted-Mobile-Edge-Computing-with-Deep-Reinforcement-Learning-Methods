# UAV-assisted-Mobile-Edge-Computing-with-Deep-Reinforcement-Learning-Methods

This repository contains the code for the training and the environment of a multi-agent edge-computing framework involving User Systems, Unmanned Aerial Vehicles (Edge Devices), and a Satellite.

The Deep Reinforcement Learning Algorithms of Dueling Double DQN, Prioritized DQN, and Policy Gradient-Softmax have been used in this project to minimize the time latency of processing and optimize the energy consumption of the UAVs.

## Environment

The 'fog_env.py' file contains the 'Offload' class, which consists of the environment setting, including the properties and number of UAVs, User Systems, and Satellites. It also contains the properties of the three types of task queues - the arrival queue, the UAV waiting queue and the User System waiting queue.
The 'step' function contains the code for updating the environment based on the decision of the RL algorithms and calculating the corresponding time latency and energy consumption values.

## RL Algorithms

The RL_brain files contain the code for the RL algorithms, the memory replay buffer, and their integration with the RNN architectures. 

1. RL_brain.py - Dueling Double DQN Algorithm
2. RL_brain_PR.py - Prioritized DQN Algorithm
3. RL_barin_PG.py - Policy Gradient-Softmax Algorithm

The above files use an 'LSTM' layer for the RNN architecture. Thus can be replaced with other architectures as seen fit.

## Training Files

The training python files incorprate the Environment and the Algorithms to run the model with the given hyper-parameters and generate the results.

1. train_DQN.py - Training code for DQN-based algorithms
2. train_PG.py - Training code for Policy Gradient-based algorithms
