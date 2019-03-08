import multiprocessing
import multiprocessing.connection
from typing import Callable


def worker_process(remote: multiprocessing.connection.Connection, make_env: Callable, kwargs):
  game = make_env()
  while True:
    cmd, data = remote.recv()
    if cmd == "step":
      remote.send(game.step(data))
    elif cmd == "reset":
      remote.send(game.reset())
    elif cmd == "close":
      remote.close()
      break
    else:
      raise NotImplementedError


class Worker:
  def __init__(self, kwargs):
    self.child, parent = multiprocessing.Pipe()
    self.process = multiprocessing.Process(target=worker_process, args=(parent, kwargs))
    self.process.start()
