import torch.optim as optim

enjoy = False
use_gpu = False


class DQNAgent:
  nr_parallel_envs = 1
  tau = 1
  gradient_clip = 1
  silly = False
  target_update = 10000 / nr_parallel_envs
  nr_epochs = 2
  buffer_size = int(1e6) * nr_parallel_envs
  training_steps = int(1e7)
  # warmup_steps = int(5e3)
  warmup_steps = 0

  epsilon_decay = None
  beta_decay = max(training_steps / 3, int(1e5))
  # optimizer = optim.RMSprop

  optimizer = optim.Adam
  gamma = 0.99
  lr = 1e-5
  l2_decay = 0
  # batch_size = 256
  batch_size = 256


class VanillaHS:
  debug = False
  normalize = False
  starting_hp = 30
  max_cards_in_board = 7
  max_cards_in_hand = 10


class PPOAgent:
  nr_parallel_envs = 30
  num_steps = 20
  lr = 3e-4
  use_gpu = True
  max_frames = 15000
  gamma = 0.995
  tau = 0.97
  seed = 543
  batch_size = 5000
  entropy_coeff = 0.0
  clip_epsilon = 0.2
  use_joint_pol_val = True


class A2CAgent:
  num_workers = 32
  optimizer = optim.Adam
  lr = 1e-5
  discount = 0.99
  use_gae = True
  gae_tau = 0.95
  entropy_weight = 0.01
  rollout_length = 20
  gradient_clip = 0.5
  # defaults
  training_steps = int(1e7)

  # Hyper params:
  hidden_size = 256
  ddlr = 3e-4
  # num_steps = 5
