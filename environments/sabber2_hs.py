import os

import environments.sabber_hs
import sb_env.SabberStone_python_client.python_pb2 as sabberstone_protobuf
import sb_mm_env.SabberStone_python_client.sabber_protocol.server as mm_server


class _GameRef(environments.sabber_hs._GameRef):
  def __init__(self, game_ref):
    self.game_ref = game_ref

  @property
  def CurrentPlayer(self):
    return self.game_ref.current_player

  @property
  def CurrentOpponent(self):
    return self.game_ref.current_opponent


class Stub2:
  def __init__(self, stub: mm_server.SabberStoneServer):
    self.stub = stub
    self._cards = None

  def NewGame(self, deck1, deck2):
    game = _GameRef(self.stub.new_game(deck1, deck2))
    return game

  def Reset(self, game):
    return _GameRef(self.stub.reset(game.game_ref))

  def Process(self, game, selected_action):
    return _GameRef(self.stub.process(game.game_ref, selected_action))

  def GetOptions(self, game):
    return self.stub.options(game.game_ref)

  def LoadCards(self):
    self._cards = self.stub.GetCardDictionary(sabberstone_protobuf.Empty()).cards

  def GetCard(self, idx):
    if self._cards is None:
      self.LoadCards()
    return self._cards[idx]


class Sabberstone2(environments.sabber_hs.Sabberstone):
  def __init__(self, *args, **kwargs):
    try:
      os.mkdir('server_files/')
    except FileExistsError:
      pass
    super().__init__(*args, **kwargs)

  def connect(self, address, rank):
    # print("sleeping", rank)
    # time.sleep(int(rank.split('.')[-1]) + 1)
    # print("waking", rank)
    server = mm_server.SabberStoneServer(id=f"sabberstone2_{rank}")
    self.stub = Stub2(server)

  def close(self):
    self.logger.warning("Not closing cleanly, restart the server")
