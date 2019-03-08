import warnings
import collections
import random

class MCTS(object):
    def __init__(self, environment, player_neural_network, epsilon=0.1):
        self.network = player_neural_network
        self.epsilon = epsilon
        warnings.warn("mcts-init method not implemented")

    def reset(self):
        warnings.warn("mcts-reset method not implemented")

    def pick_action(self, state, possible_actions):
        if False and random.random() > self.epsilon:
            policy, value = self.network.policy_value_fn(state, possible_actions)
        else:
            action = random.choice(possible_actions)
        return action

class UCT(object):
    def __init__(self):
         self.score = collections.defaultdict(0)
         self.visits = collections.defaultdict(0)
         self.differential = collections.defaultdict(0)
         self.total = 1


    def heuristic_value(self, state):
         N = total
         Ni = _visits[state] + 1e-5
         V = score[state] * 1.0 / Ni
         return V + C*(np.log(N)/Ni)

    def record(self, state, score):
         self.total += 1
         self.visits[state] += 1
         differential[state] += score

