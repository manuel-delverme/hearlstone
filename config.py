class DQNAgent:
  use_target = False
  use_double_q = True
  gamma = 0.99
  lr = 1e-6
  l2_decay = 0
  batch_size = 128
