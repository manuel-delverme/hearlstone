import torch.optim as optim

enjoy = True


class DQNAgent:
  silly = True
  target_update = 500
  nr_epochs = 1
  buffer_size = int(1e5)
  training_steps = int(1e6)
  warmup_steps = 0  # int(5e3)

  epsilon_decay = None
  beta_decay = min(training_steps / 6, int(1e5))
  use_target = False
  # optimizer = optim.RMSprop

  optimizer = optim.Adam
  use_double_q = True
  gamma = 0.99
  lr = 1e-5
  l2_decay = 0
  # batch_size = 256
  batch_size = 32


class VanillaHS:
  normalize = False
  starting_hp = 30
  max_cards_in_board = 7
