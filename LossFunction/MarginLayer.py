import torch
import torch.nn as nn


class InnerProduct(nn.Module):
    def __init__(self, features_in, features_out):
        super(InnerProduct, self).__init__()
        self.features_in = features_in
        self.features_out = features_out
        self.weight = nn.Parameter(torch.Tensor(features_in, features_out))

    def forward(self, x):
        output = x @ self.weight
        return output


class CosineMarginProduct(nn.Module):
    def __init__(self, features_in, features_out, s=30, m=0.4):
        super(CosineFaceMargin, self).__init__()
        self.features_in = features_in
        self.features_out = features_out
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(features_in, features_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        w1 = torch.norm(x, 2, 1)
        w2 = torch.norm(self.weight, 2, 1)
        cosine = self.weight.t() @ x / torch.ger(w1, w2).clamp(min=1e-8)
