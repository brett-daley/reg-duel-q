import torch
from torch import nn

from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead

import reg_duel_q


def semi_gradient_mse(targets, predictions, batch_accumulator):
    # Make sure all inputs have consistent, 1-dim shapes to avoid bugs
    targets = targets.squeeze(axis=-1)
    predictions = predictions.squeeze(axis=-1)
    assert targets.ndim == 1
    assert predictions.ndim == 1
    assert targets.shape == predictions.shape
    return nn.functional.mse_loss(predictions, targets.detach(), reduction=batch_accumulator) / 2.0


class SharedQVNetwork(nn.Module):
    def __init__(self, backbone: nn.Module, num_outputs: int, share_v: bool, no_flow_q: bool):
        super(SharedQVNetwork, self).__init__()
        self.share_v = share_v
        self.no_flow_q = no_flow_q

        assert isinstance(backbone, nn.Sequential)
        assert isinstance(backbone[-2], nn.Linear)
        assert isinstance(backbone[-1], nn.ReLU)
        num_features = backbone[-2].in_features
        hidden_size = backbone[-2].out_features
        self.backbone = backbone = backbone[:-2]  # Remove last layer + activation

        self.stream_q = nn.Sequential(
            init_chainer_default(nn.Linear(num_features, hidden_size)),
            nn.ReLU(),
            init_chainer_default(nn.Linear(hidden_size, num_outputs, bias=False)),
            reg_duel_q.nn_utils.SingleSharedBias(),
        )
        self.stream_v = nn.Sequential(
            init_chainer_default(nn.Linear(num_features, hidden_size)),
            nn.ReLU(),
            init_chainer_default(nn.Linear(hidden_size, 1)),
        )
        self.head = DiscreteActionValueHead()

    def forward(self, x, subtract_mean_adv=False):
        f = self.backbone(x)
        q = self.stream_q(f.detach() if self.no_flow_q else f)
        v = self.stream_v(f)
        if self.share_v:
            adv = q  # Use Q-head for advantage
            if subtract_mean_adv:
                adv -= torch.mean(adv, axis=-1, keepdim=True)  # Subtract mean for identifiability
            q = adv + (v.detach() if self.no_flow_q else v).expand_as(adv)
        else:
            assert not subtract_mean_adv
        return self.head(q), self.head(v)

    def forward_q(self, x, **kwargs):
        q, _ = self.forward(x, **kwargs)
        return q

    def forward_v(self, x, **kwargs):
        _, v = self.forward(x, **kwargs)
        return v


class ModifiedDuelingNetwork(SharedQVNetwork):
    def forward(self, x, return_adv=False):
        assert self.share_v
        f = self.backbone(x)
        adv = self.stream_q(f.detach() if self.no_flow_q else f)  # Use Q-head for advantage
        v = self.stream_v(f)
        q = adv + (v.detach() if self.no_flow_q else v).expand_as(adv)

        returns = (self.head(q), self.head(v))
        if return_adv:
            returns = (*returns, self.head(adv))
        return returns

