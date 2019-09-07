import torch
import torch.nn as nn
import torch.autograd as autograd

import environments.sabber_hs as sabber_hs

class PointerNetwork(nn.Module):
    # https://arxiv.org/abs/1506.03134
    def __init__(self, out_features=128):
        super(PointerNetwork, self).__init__()

        action_lookup = torch.Tensor(list(sabber_hs.enumerate_actions().keys()))
        self.action_lookup = action_lookup
        n_actions, action_size = action_lookup.shape
        self.v = autograd.Variable(
            torch.rand((n_actions, 1), requires_grad=True))
        self.option_encoding = nn.Linear(in_features=action_size,
                                         out_features=out_features)

    def forward(self, x, options):
        x = x.unsqueeze(1)
        oe = self.option_encoding(self.action_lookup).unsqueeze(0)  
        oe = oe * options.unsqueeze(-1)
        logits = (self.v * torch.tanh(oe + x)).sum(dim=-1)
        return logits

