import hs_config


class GameManager(object):
  def __init__(self, seed=None, env_id=None, log_dir=None):
    self.seed = seed
    self.game_class = hs_config.VanillaHS.get_game_mode()
    self.opponent_class = hs_config.VanillaHS.get_opponent()
    self.opponent = self.opponent_class()

  def __call__(self, extra_seed):
    hs_game = self.game_class(seed=self.seed, extra_seed=extra_seed)
    hs_game.set_opponent(opponent=self.opponent)
    return hs_game

  def set_opponent(self, opponent=None, level=5):
    if opponent is None:
      opponent = self.opponent_class
    self.opponent = opponent(level=level)

  def update_learning_opponent(self, checkpoint_file):
    self.opponent = self.opponent_class()
    self.opponent.load_model(checkpoint_file)
    self.opponent.eval()
    self.opponent.freeze()
