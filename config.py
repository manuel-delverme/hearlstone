import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import torch.optim as optim

enjoy = False
use_gpu = False


class VanillaHS:
  debug = False
  normalize = True
  starting_hp = 30
  max_cards_in_board = 5
  max_cards_in_hand = 5
  # opponent = agents.heuristic.hand_coded.PassingAgent
  # opponent = agents.heuristic.random_agent.RandomAgent
  opponent = agents.heuristic.hand_coded.HeuristicAgent


class DQNAgent:
  nr_parallel_envs = 1
  nr_epochs = nr_parallel_envs
  tau = 1
  gradient_clip = 1
  silly = False
  target_update = 1000
  buffer_size = int(1e6)
  training_steps = int(1e7)
  # warmup_steps = int(5e3)
  warmup_steps = 0

  epsilon_decay = None
  beta_decay = max(training_steps / 3, int(1e5))
  # optimizer = optim.RMSprop

  optimizer = optim.Adam
  gamma = 0.99
  # lr = 1e-5
  lr = 1e-3
  l2_decay = 0
  # batch_size = 256
  batch_size = 32


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
  num_workers = 64
  optimizer = optim.Adam
  lr = 1e-5
  discount = 0.99
  use_gae = True
  gae_tau = 0.95
  checkpoint_every = int(2e5)
  entropy_weight = 0.001
  rollout_length = 5
  gradient_clip = 0.5
  # defaults
  training_steps = int(1e7)

  # Hyper params:
  hidden_size = 32
