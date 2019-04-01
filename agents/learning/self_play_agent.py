from abc import ABC

import agents.base_agent
import game_utils
import hs_config


class SelfPlayAgent(agents.base_agent.Agent, ABC):
  def self_play(self, game_manger: game_utils.GameManager, checkpoint_file):
    # levels, updates = (3, 5), (100, hs_config.PPOAgent.num_updates - 100)
    levels, updates = (5,), (hs_config.PPOAgent.num_updates,)

    updates_so_far = 0
    for level, num_updates in zip(*(levels, updates)):
      print("RUNNING: level {} for {} updates".format(level, num_updates))
      game_manger.set_opponent(level=level)
      checkpoint_file = self.train(game_manger, checkpoint_file=checkpoint_file, num_updates=num_updates,
                                   updates_offset=updates_so_far)
      updates_so_far += num_updates

    assert checkpoint_file
    # https://openai.com/blog/openai-five/

    # for self_play_iter in range(hs_config.SelfPlay.num_opponent_updates):
    #   game_manger.update_opponent(checkpoint_file)
    #   self.train(game_manger, checkpoint_file)  # , checkpoint_file=latest_checkpoint)
