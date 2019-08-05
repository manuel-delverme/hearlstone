import itertools
import random
import time

import numpy as np
import torch
import tqdm

import agents.heuristic.random_agent
import environments.sabber_hs
# from environments.vanilla_hs import VanillaHS
import game_utils
import hs_config
# env = VanillaHS(skip_mulligan=True)
import shared.env_utils


# env.set_opponents([SabberAgent(level=6)])


def HSenv_test():
  env = environments.sabber_hs.Sabbertsone('localhost:50052')
  s0, reward, terminal, info = env.reset()
  avg_time = []
  for _ in range(int(1e4)):
    done = False
    r = None
    while not done:
      random_act = random.choice(info['possible_actions'])
      start = time.time()
      s, r, done, info = env.step(int(random_act))
      delta = time.time() - start
      avg_time.append(1 / delta)
      # assert env.game.CurrentPlayer.id == 1
    # assert r != 0.0
    print(np.mean(avg_time))


def test_multiprocessFPS():
  print("[Train] Loading training environments")
  game_manager = game_utils.GameManager(seed=None, address='0.0.0.0:50052')
  game_manager.use_heuristic_opponent = False

  no_subprocess = hs_config.Environment.no_subprocess
  # hs_config.Environment.no_subprocess = False
  nr_processes = 10

  envs = shared.env_utils.make_vec_envs(game_manager, nr_processes, -1, None, None, False)

  hs_config.Environment.no_subprocess = no_subprocess

  _, reward, terminal, info = envs.reset()

  pbar = tqdm.tqdm()
  for _ in itertools.count():
    pbar.update(nr_processes)
    pa = info['possible_actions']
    actions = []
    for i in range(nr_processes):
      rows = np.argwhere(pa[i, :])
      assert rows.shape[0] == 1
      rows = rows[0]
      random_act = random.choice(rows)
      # [ai.tolist() for a in actions for ai in a]
      actions.append(int(random_act))
    _, r, done, info = envs.step(torch.tensor(actions).unsqueeze(-1))


def test_wrapperFPS():
  env = environments.sabber_hs.Sabbertsone('localhost:50052')
  env.set_opponents([agents.heuristic.random_agent.RandomAgent()])
  s0, reward, terminal, info = env.reset()
  for _ in tqdm.tqdm(itertools.count()):
    pa = info['possible_actions']
    rows = np.argwhere(pa)  # row is always 0
    random_act = random.choice(rows)
    s, r, done, info = env.step(int(random_act))
    if done:
      env.reset()


def test_loss():
  env.reset()
  for _ in range(3):
    done = False
    r = None
    while not done:
      s, r, done, info = env.step(0)
      env.render()
    assert r == -1


def test_add_opponent():
  game_manager = game_utils.GameManager(seed=None, address='0.0.0.0:50052')
  game_manager.use_heuristic_opponent = False
  for _ in range(10000):
    game_manager.add_learning_opponent("/home/esac/projects/hearlstone/ppo_save_dir/id=d10c4n3:steps=192:inputs=92.pt")


if __name__ == '__main__':
  test_add_opponent()
  # test_multiprocessFPS()
  # test_wrapperFPS()
  # test_loss()
  # HSenv_test()
