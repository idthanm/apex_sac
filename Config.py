from __future__ import print_function
import numpy as np
import torch
import math

class GeneralConfig:
  buffer_size_max = 100000
  buffer_queue_size = 100
  para_queue_size = 3




  num_inputs = 5
  num_control = 2
  num_batchsize = 256
  num_iteration = 3000
  num_presimulation = 10
  num_simulation = 1000
  num_buffersize = 50000
  frequency_time = 5
  frequency_simulation = 200
  frequency_learn = frequency_simulation/frequency_time
  num_agent = 256
  num_agent_cons = 50
  num_step = np.floor(np.random.uniform(40, 320, [num_agent, 1])).astype(int)#0 + np.floor(np.random.uniform(50, 800, [num_agent, 1]))
  num_step[num_agent-10:num_agent,0] = 10
  num_step[0] = 120
  row_length = num_iteration
