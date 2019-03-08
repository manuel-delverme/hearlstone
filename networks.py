import torch.nn
import torch.autograd
import numpy as np
import torch.optim
import tqdm


class SimpleModel(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_size, action_size, bias=True)

    def forward(self, state):
        action = self.fc1(state)
        return action


class NeuralNetwork(object):
    def __init__(self, state_size, action_size):
        self.model = SimpleModel(state_size, action_size)

    def train_on_samples(self, samples, nr_epochs=100, batch_size=16):
        # TODO: this should be moved out of the class

        optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()

        boards, current_player, pis, vs = zip(*samples)
        boards = np.array(boards)
        pis = np.array(pis)
        vs = np.array(vs)

        for epoch_nr in range(nr_epochs):
            sample_ids = list(range(len(samples)))
            np.random.shuffle(sample_ids)

            for start in range(0, len(samples) // batch_size, batch_size):
                # mini_batch = samples[sample_ids[start: start + batch_size]]
                mini_batch_boards = boards[sample_ids[start: start + batch_size]]
                mini_batch_pis = pis[sample_ids[start: start + batch_size]]
                mini_batch_vs = vs[sample_ids[start: start + batch_size]]

                boards = torch.FloatTensor(np.array(mini_batch_boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(mini_batch_pis))
                target_vs = torch.FloatTensor(np.array(mini_batch_vs).astype(np.float64))

                boards, target_pis, target_vs = torch.autograd.Variable(boards), torch.autograd.Variable(
                    target_pis), torch.autograd.Variable(target_vs)

                out_pi, out_v = self.model(boards)
                l_pi = loss_pi(target_pis, out_pi)
                l_v = loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()


def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]


def loss_v(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
