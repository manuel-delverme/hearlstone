import numpy as np
import torch
import torch.nn as nn

import environments.base_env
import hs_config
import shared.constants as C
import specs
from shared.utils import init


class ActorCritic(nn.Module):
  def __init__(self, num_inputs: int, num_actions: int, device: torch.device):
    super(ActorCritic, self).__init__()
    assert num_inputs > 0
    assert num_actions > 0

    self.num_inputs = num_inputs
    self.num_possible_actions = num_actions

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2) / 100)

    self.card_embedding = nn.Embedding(hs_config.Environment.num_possible_cards, hs_config.PPOAgent.card_embedding_size)

    entities = [-1, ] + list(C.MINIONS) + list(C.SPELLS)
    self.card_id_to_index = {v: k for k, v in enumerate(entities)}

    self.actor = nn.Sequential(
        init_(nn.Linear(438, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        nn.Linear(hs_config.PPOAgent.hidden_size, self.num_possible_actions),
    )
    self.critic = nn.Sequential(
        init_(nn.Linear(438, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        nn.Linear(hs_config.PPOAgent.hidden_size, 1),
    )
    self.reset_actor()
    self.reset_critic()
    self.train()
    self.to(device)

  def to(self, *args, **kwargs):
    self.actor.to(*args, **kwargs)
    self.critic.to(*args, **kwargs)
    self.critic_regression.to(*args, **kwargs)
    self.card_embedding.to('cpu')

  def reset_actor(self):
    logits = list(self.actor.children())[-1]
    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.1)
    logits.apply(init_)

  def reset_critic(self):
    value_fn = list(self.critic.children())[-1]
    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.001)
    self.critic_regression = init_(value_fn)

  def forward(self, observations: torch.FloatTensor, possible_actions: torch.FloatTensor,
      deterministic: bool = False) -> (torch.FloatTensor, torch.LongTensor, torch.FloatTensor):
    if isinstance(observations, np.ndarray):
      observations = torch.from_numpy(observations).float()
      observations = observations.unsqueeze(0)
    if len(possible_actions.shape) == 1:
      possible_actions = torch.from_numpy(possible_actions)
      possible_actions = possible_actions.unsqueeze(0)

    assert specs.check_observation(self.num_inputs, observations)
    assert specs.check_possible_actions(self.num_possible_actions, possible_actions)
    assert observations.shape[0] == possible_actions.shape[0]
    assert isinstance(deterministic, bool)

    if not isinstance(observations, torch.Tensor):
      observations = torch.tensor(observations)

    action_distribution, value = self.actor_critic(observations, possible_actions)

    if deterministic:
      action = action_distribution.probs.argmax(dim=-1, keepdim=True)
    else:
      action = action_distribution.sample().unsqueeze(-1)

    action_log_probs = self.action_log_prob(action, action_distribution)

    assert value.size(1) == 1
    assert specs.check_action(action)
    assert action_log_probs.size(1) == 1
    assert value.size(0) == action.size(0) == action_log_probs.size(0)

    return value, action, action_log_probs

  def action_log_prob(self, action, action_distribution):
    assert action.size(1) == 1

    action_log_probs = action_distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)

    assert action_log_probs.size(1) == 1
    return action_log_probs

  # def critic(self, inputs):
  #   value_features = self.critic_features(inputs)
  #   value = self.critic_regression(value_features)
  #   return value

  def actor_critic(self, inputs, possible_actions):
    features = self.extract_features(inputs)
    value = self.critic(features)
    logits = self.actor(features)

    action_distribution = self._get_action_distribution(possible_actions, logits)
    return action_distribution, value

  def extract_features(self, observation):
    batch_size = observation.shape[0]
    offset, board, hand, mana, hero, board_size, deck = environments.base_env.RenderableEnv.render_player(observation, preserve_types=True)
    offset, o_board, _, o_mana, o_hero, o_board_size, _, = environments.base_env.RenderableEnv.render_player(observation, offset,
                                                                                                             show_hand=False,
                                                                                                             preserve_types=True)
    board_repr = self.parse_zone(board)
    hand_repr = self.parse_zone(hand)
    o_board_repr = self.parse_zone(o_board)
    deck_repr = self.parse_zone(deck)

    assert mana.shape == (batch_size, 1)
    assert hero.shape == (batch_size, 4)
    assert board_repr.flatten(start_dim=1).shape == (
      batch_size, hs_config.Environment.max_cards_in_board * (hs_config.PPOAgent.card_embedding_size + 3))
    assert hand_repr.flatten(start_dim=1).shape == (
      batch_size, hs_config.Environment.max_cards_in_hand * (hs_config.PPOAgent.card_embedding_size + 3))
    assert deck_repr.flatten(start_dim=1).shape == (
      batch_size, hs_config.Environment.max_cards_in_deck * (hs_config.PPOAgent.card_embedding_size + 3))
    assert o_board_repr.flatten(start_dim=1).shape == (
      batch_size, hs_config.Environment.max_cards_in_board * (hs_config.PPOAgent.card_embedding_size + 3))

    assert o_mana.shape == (batch_size, 1)
    features = torch.cat((
      mana,
      hero,
      board_repr.flatten(start_dim=1),
      hand_repr.flatten(start_dim=1),
      deck_repr.flatten(start_dim=1),
      o_board_repr.flatten(start_dim=1),
      o_mana,
    ), dim=1)
    return features

  def parse_zone(self, zone):
    zone = torch.stack(zone)
    original_device = zone.device
    zone = zone.cpu().permute(1, 0, 2)  # position, batch, dims -> batch, position, dim
    sparse = zone[:, :, -1]
    dense = zone[:, :, :-1]

    ids = []
    for card_id in sparse.reshape(-1):
      card_index = self.card_id_to_index[int(card_id)]
      ids.append(card_index)

    sparse = torch.LongTensor(ids).view(sparse.shape)
    card_embeddings = self.card_embedding(sparse.long())
    zone = torch.cat((dense, card_embeddings), dim=-1)
    return zone.flatten(start_dim=1).to(original_device)  # turn into batch_size x zone

  def evaluate_actions(self, observations: torch.FloatTensor, action: torch.LongTensor,
      possible_actions: torch.FloatTensor) -> (
      torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):

    assert specs.check_observation(self.num_inputs, observations)
    assert specs.check_possible_actions(self.num_possible_actions, possible_actions)
    assert specs.check_action(action)
    assert action.size(0) == observations.size(0) == possible_actions.size(0)

    action_distribution, value = self.actor_critic(observations, possible_actions)
    action_log_probs = self.action_log_prob(action, action_distribution)
    dist_entropy = action_distribution.entropy().mean()

    assert value.size(1) == 1
    assert action_log_probs.size(1) == 1
    assert not dist_entropy.size()
    assert value.size(0) == action_log_probs.size(0)

    return value, action_log_probs, dist_entropy

  @staticmethod
  def _get_action_distribution(possible_actions, logits):
    logits -= ((1 - possible_actions) * hs_config.PPOAgent.BIG_NUMBER).float()
    return torch.distributions.Categorical(logits=logits)
