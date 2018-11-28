import torch.nn as nn
import config

from agents.learning.models import noisy_networks
import torch.nn.functional as F
import torch


class DQN(nn.Module):
  def __init__(self, num_inputs, num_actions, use_cuda):
    self.num_inputs = num_inputs
    self.num_actions = num_actions
    self.use_cuda = use_cuda
    super(DQN, self).__init__()

  def build_network(self):
    if config.DQNAgent.silly:
      self.silly_fc = noisy_networks.NoisyLinear(69, 1)
    else:
      # each card is summarized in 12 numbers
      self.conv_minions = nn.Conv1d(1, 2, kernel_size=2, stride=2,)
      self.fc_heroes = nn.Linear(4, 4)

      self.action_fc1 = nn.Linear(self.num_actions, 64)

      # 1 is the Q(s, a) value
      self.value_fc1 = nn.Linear(67, 256)
      self.value_fc2 = nn.Linear(256, 256)
      self.value_nosiy_fc3 = noisy_networks.NoisyLinear(256, 1)

    if self.use_cuda:
      self.cuda()

  def state_features(self, board):
    # entity_size = int(board.shape[1] / ((config.VanillaHS.max_cards_in_board + 1) * 2))
    # board_side_size = int(board.shape[1] / 2)

    # my_board = board[:, :board_side_size]
    # my_minions = my_board[:, :-entity_size]
    # my_hero = my_board[:, -entity_size:]

    # his_board = board[:, board_side_size:]
    # his_minions = his_board[:, :-entity_size]
    # his_hero = his_board[:, -entity_size:]

    # minions_x = torch.cat((my_minions, his_minions), dim=1).unsqueeze(dim=1)
    # heroes_x = torch.cat((my_hero, his_hero), dim=1)

    # minions_features = self.conv_minions(minions_x)
    # minions_features = minions_features.view(minions_features.size(0), -1)

    # heroes_features = self.fc_heroes(heroes_x)

    h = torch.cat((minions_features, heroes_features), dim=1)
    h = F.leaky_relu(h)
    # TODO: further strcture, my board my hero self opponent board, opponent hero

    return h

  def action_features(self, x):
    h = self.action_fc1(x)
    h = F.leaky_relu(h)
    return h

  def value_network(self, x):
    h = self.value_fc1(x)
    h = F.leaky_relu(h)
    h = self.value_fc2(h)
    h = F.leaky_relu(h)
    h = self.value_nosiy_fc3(h)
    return h

  def silly_network(self, x):
    h = self.silly_fc(x)
    return torch.tanh(h)

  def forward(self, x):
    x = torch.FloatTensor(x)
    if self.use_cuda:
      x = x.cuda()

    if config.DQNAgent.silly:
      q_val = self.silly_network(x)
    else:
      board = x[:, :-2]
      action = x[:, -2:]
      # h_state = self.state_features(board)
      # h_action = self.action_features(action)
      h_state = board
      h_action = action
      h_value = torch.cat((h_state, h_action), dim=1)
      q_val = self.value_network(h_value)

    if self.use_cuda:
      q_val = q_val.cpu()
    return q_val
