import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import environments.base_env
import hs_config
import shared.constants as C
import specs
from shared.utils import init


class ActorCritic(nn.Module):
  actor_parameters = None
  critic_parameters = None

  def __init__(self, num_inputs: int, num_actions: int):
    super(ActorCritic, self).__init__()
    assert num_inputs > 0
    assert num_actions > 0

    self.num_inputs = C.STATE_SPACE  # num_inputs
    self.num_possible_actions = num_actions

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2) / 100)

    self.inactive_card_summarizer = nn.Conv1d(1, 64, kernel_size=C.INACTIVE_CARD_ENCODING_SIZE, stride=C.INACTIVE_CARD_ENCODING_SIZE)
    self.deck_summarizer = nn.Conv1d(1, 16, kernel_size=C.INACTIVE_CARD_ENCODING_SIZE, stride=C.INACTIVE_CARD_ENCODING_SIZE)
    self.active_card_summarizer = nn.Conv1d(1, 64, kernel_size=C.ACTIVE_CARD_ENCODING_SIZE, stride=C.ACTIVE_CARD_ENCODING_SIZE)

    self.actor = nn.Sequential(
        init_(nn.Linear(678, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        nn.Linear(hs_config.PPOAgent.hidden_size, self.num_possible_actions),
    )
    self.critic_head = nn.Sequential(
        init_(nn.Linear(678, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
        nn.ReLU(),
        nn.Linear(hs_config.PPOAgent.hidden_size, 1),
    )
    logits = list(self.actor.children())[-1]
    logits.apply(lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.1))

    value_fn = list(self.critic_head.children())[-1]
    value_fn.apply(lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.001))

    # This is necessary for the optimizer.
    # TODO: should we have 3 optimizers?
    # Actor, critic, features so that features has 0.5 learning rate.
    self.actor_parameters = [
      *list(self.inactive_card_summarizer.parameters()),
      *list(self.deck_summarizer.parameters()),
      *list(self.active_card_summarizer.parameters()),
      *list(self.actor.parameters()),
    ]
    self.critic_parameters = [
      *list(self.inactive_card_summarizer.parameters()),
      *list(self.deck_summarizer.parameters()),
      *list(self.active_card_summarizer.parameters()),
      *list(self.critic_head.parameters()),
    ]

    self.train()

  def forward(self, observations: torch.FloatTensor, possible_actions: torch.FloatTensor,
      deterministic: bool = False) -> (torch.FloatTensor, torch.LongTensor, torch.FloatTensor):
    if isinstance(observations, np.ndarray):
      observations = torch.from_numpy(observations).float()
      observations = observations.unsqueeze(0)
    if len(possible_actions.shape) == 1:
      possible_actions = torch.from_numpy(possible_actions)
      possible_actions = possible_actions.unsqueeze(0)

    # assert specs.check_observation(self.num_inputs, observations)
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

  def critic(self, observation):
    features = self.extract_features(observation)
    return self.critic_head(features)

  def actor_critic(self, observation, possible_actions):
    features = self.extract_features(observation)

    value = self.critic_head(features)
    logits = self.actor(features)

    action_distribution = self._get_action_distribution(possible_actions, logits)
    return action_distribution, value

  def extract_features(self, observation):
    batch_size = observation.shape[0]
    observation.to(hs_config.device)
    offset, board, hand, mana, hero, deck = environments.base_env.render_player(observation, preserve_types=True)
    deck = observation[:, offset: offset + hs_config.Environment.max_cards_in_deck * C.INACTIVE_CARD_ENCODING_SIZE]
    deck = deck.view(batch_size, 1, -1)
    _, o_board, _, o_mana, o_hero, _ = environments.base_env.render_player(observation, offset, current_player=False, preserve_types=True)
    # TODO @manuel something is wrong here with tensor allocation
    board_repr = self.parse_active_zone(batch_size, board).to(observation.device)
    hand_repr = self.parse_inactive_zone(batch_size, hand).flatten(start_dim=1).to(observation.device)
    o_board_repr = self.parse_active_zone(batch_size, o_board).to(observation.device)
    deck_repr = self.deck_summarizer(deck).flatten(start_dim=1).to(observation.device)

    assert mana.shape == (batch_size, 1)
    assert hero.shape == (batch_size, 4)
    assert board_repr.flatten(start_dim=1).shape == (batch_size, 64)
    assert hand_repr.flatten(start_dim=1).shape == (batch_size, 64)
    assert deck_repr.flatten(start_dim=1).shape == (batch_size, 16 * hs_config.Environment.max_cards_in_deck)
    assert o_board_repr.flatten(start_dim=1).shape == (batch_size, 64)
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

  def parse_inactive_zone(self, batch_size, zone):
    if zone:
      zone = torch.cat(zone).view(batch_size, 1, -1)
      zone = self.inactive_card_summarizer(zone)
      hand_repr = F.max_pool1d(zone, kernel_size=zone.shape[-1], stride=zone.shape[-1])
    else:
      hand_repr = torch.ones((batch_size, 1, self.active_card_summarizer.out_channels)) * -1
    return hand_repr

  def parse_active_zone(self, batch_size, zone):
    if zone:
      zone = torch.cat(zone).view(batch_size, 1, -1)
      zone = self.active_card_summarizer(zone)
      return F.max_pool1d(zone, kernel_size=zone.shape[-1], stride=zone.shape[-1])
    else:
      return torch.ones((batch_size, 1, self.active_card_summarizer.out_channels)) * -1

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
