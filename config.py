class DQNAgent:
  buffer_size = int(1e5)
  training_steps = int(1e6)
  warmup_steps = int(1e3)
  use_target = False
  use_double_q = True
  gamma = 0.99
  lr = 1e-5
  l2_decay = 0
  batch_size = 128
