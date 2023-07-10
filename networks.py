import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from utils import bin_concrete_sample


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, feat_dim, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.W = nn.Parameter(torch.Tensor(feat_dim, num_classes))
        self.bias = nn.Parameter(torch.Tensor(num_classes))

        torch.nn.init.xavier_normal_(self.W)
        self.bias.data.fill_(0.1)

    def forward(self, features, mask=None):

        if mask is not None:
            features = features * mask

        out = nn.functional.linear(features, self.W.T, self.bias)

        return out


def bernoulli_kl(p, q, eps=1e-7):
    return (p * ((p + eps).log() - (q + eps).log())) + (1. - p) * ((1. - p + eps).log() - (1. - q + eps).log())


class DiscoveryMechanism(nn.Module):

    def __init__(self, feat_dim, cdim, prior):
        super(DiscoveryMechanism, self).__init__()
        self.W = nn.Parameter(torch.Tensor(feat_dim, cdim))
        self.bias = nn.Parameter(torch.Tensor(cdim))

        self.register_buffer('temp', torch.tensor(0.1))
        self.register_buffer('temptest', torch.tensor(.01))

        self.register_buffer('prior', torch.tensor(prior))

        torch.nn.init.xavier_normal_(self.W)
        self.bias.data.fill_(0.0)

    def forward(self, features, probs_only=False):

        logits = nn.functional.linear(features, self.W.T, self.bias)
        kl = 0.

        if self.training:

            # print(out)
            out = bin_concrete_sample(logits, self.temp).clamp(1e-6, 1.-1e-6)
            kl = bernoulli_kl(torch.sigmoid(logits), self.prior).sum(1).mean()

        else:

            if probs_only:
                out = torch.sigmoid(logits)
            else:
                out = RelaxedBernoulli(self.temptest, logits=logits).sample()

        return out, kl
