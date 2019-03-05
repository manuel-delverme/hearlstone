import subprocess
import torch.nn
import agents.base_agent
from agents.learning.models import actor_critic

import time
import tensorboardX
from deep_rl.component import env_utils
from deep_rl.component import replay

import hs_config
import torch
import numpy as np
import os
import tqdm


class A2CAgent(agents.base_agent.Agent):
  def choose(self, observation, possible_actions):
    pass

  def __init__(self, num_inputs, num_actions, should_flip_board=False, model_path="checkpoints/checkpoint.pth.tar",
               record=True, opponent=None, ) -> None:
    self.model_path = model_path
    self.network = actor_critic.A2C(num_inputs, num_actions)
    self.network.build_network()
    self.optimizer = hs_config.A2CAgent.optimizer(
      self.network.parameters(),
      lr=hs_config.A2CAgent.lr
    )
    experiment_name = time.strftime("%Y_%m_%d-%H_%M_%S")
    log_dir = 'runs/{}'.format(experiment_name)

    if record:
      self.summary_writer = tensorboardX.SummaryWriter(log_dir=log_dir, flush_secs=10)
      print("Experiment name:", experiment_name)
      cmd = "find {} -name '*.py' | grep -v venv | tar -cvf {}/code.tar --files-from -".format(os.getcwd(), log_dir)
      subprocess.check_output(cmd, shell=True)
    else:
      self.summary_writer = tensorboardX.SummaryWriter(log_dir='/tmp/trash/', flush_secs=999999999999)

  def compute_returns(self, next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
      R = rewards[step] + gamma * R * masks[step]
      returns.insert(0, R)
    return returns

  def train(self, make_env):
    envs = env_utils.Task(make_env, hs_config.A2CAgent.num_workers, single_process=False)
    progress_bar = tqdm.tqdm(total=hs_config.A2CAgent.training_steps)
    states, _, _, possible_actions = envs.reset()

    frame_idx = 0
    while frame_idx < hs_config.A2CAgent.training_steps:
      log_probs = []
      values = []
      rewards = []
      masks = []
      entropy = 0

      for _ in range(hs_config.A2CAgent.rollout_length):
        dist, value = self.network(states, possible_actions)
        actions = dist.sample()
        next_state, reward, done, possible_actions = envs.step(actions)

        log_prob = dist.log_prob(actions)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)

        # rewards.append(reward)
        rewards.append(torch.Tensor(reward))  # .unsqueeze(1))
        # masks.append(1 - done)
        masks.append(torch.Tensor(1 - done))  # .unsqueeze(1))

        for env_idx, (env_done, env_reward) in enumerate(zip(done, reward)):
          if env_done:
            game_value = (int(env_reward) + 1) / 2
            self.summary_writer.add_scalar('game_stats/game_value', game_value,
                                           frame_idx * hs_config.A2CAgent.num_workers + env_idx)

        states = next_state
        progress_bar.update(len(states))

        if frame_idx % hs_config.A2CAgent.checkpoint_every == 0 and frame_idx > 0:
          torch.save(self.network.state_dict(), self.model_path)

        frame_idx += 1

      next_state = torch.FloatTensor(next_state)  # .to(device)

      _, next_value = self.network(next_state, possible_actions)
      returns = self.compute_returns(next_value, rewards, masks)

      log_probs = torch.cat(log_probs)
      returns = torch.cat(returns).detach()
      values = torch.cat(values)

      advantage = returns - values

      actor_loss = -(log_probs * advantage.detach()).mean()
      critic_loss = advantage.pow(2).mean()

      loss = actor_loss + 0.5 * critic_loss - hs_config.A2CAgent.entropy_weight * entropy

      self.summary_writer.add_scalar('loss/actor_loss', actor_loss, (frame_idx + 1) * hs_config.A2CAgent.num_workers)
      self.summary_writer.add_scalar('loss/critic_loss', 0.5 * critic_loss,
                                     (frame_idx + 1) * hs_config.A2CAgent.num_workers)
      self.summary_writer.add_scalar('loss/entropy', hs_config.A2CAgent.entropy_weight * entropy,
                                     (frame_idx + 1) * hs_config.A2CAgent.num_workers)
      self.summary_writer.add_scalar('loss/total_loss', loss, (frame_idx + 1) * hs_config.A2CAgent.num_workers)
      progress_bar.set_description(str((frame_idx + 1) * hs_config.A2CAgent.num_workers))

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
