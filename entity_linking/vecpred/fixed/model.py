import torch
from torch import nn


ACTIVATION_FUNCS = {'relu': torch.relu, 'tanh': torch.tanh}


class FCNN(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, activation_function: str = 'relu'):
        super().__init__()
        self.af = ACTIVATION_FUNCS[activation_function]
        # Learnable layers
        self.linear1 = nn.Linear(in_dims, in_dims)
        self.linear2 = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        # x.size() = (N, in_dims)
        x = self.af(self.linear1(x))
        # x.size() = (N, in_dims)
        x = self.linear2(x)
        # x.size() = (N, DBPEDIA_EMBEDDING_DIMENSIONS)
        return x


class FCNN2(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, activation_function: str = 'relu'):
        super().__init__()
        self.af = ACTIVATION_FUNCS[activation_function]
        # Learnable layers
        self.linear1 = nn.Linear(in_dims, in_dims)
        self.linear2 = nn.Linear(in_dims, in_dims)
        self.linear3 = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        # x.size() = (N, in_dims)
        x = self.af(self.linear1(x))
        # x.size() = (N, in_dims)
        x = self.af(self.linear2(x))
        # x.size() = (N, in_dims)
        x = self.linear3(x)
        # x.size() = (N, DBPEDIA_EMBEDDING_DIMENSIONS)
        return x


class FCNN3(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, activation_function: str = 'relu'):
        super().__init__()
        self.af = ACTIVATION_FUNCS[activation_function]
        # Learnable layers
        self.linear1 = nn.Linear(in_dims, in_dims)
        self.linear2 = nn.Linear(in_dims, in_dims)
        self.linear3 = nn.Linear(in_dims, in_dims)
        self.linear4 = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        # x.size() = (N, in_dims)
        x = self.af(self.linear1(x))
        # x.size() = (N, in_dims)
        x = self.af(self.linear2(x))
        # x.size() = (N, in_dims)
        x = self.af(self.linear3(x))
        # x.size() = (N, in_dims)
        x = self.linear4(x)
        # x.size() = (N, DBPEDIA_EMBEDDING_DIMENSIONS)
        return x


class FCNN4(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, activation_function: str = 'relu'):
        super().__init__()
        self.af = ACTIVATION_FUNCS[activation_function]
        # Learnable layers
        self.linear1 = nn.Linear(in_dims, in_dims)
        self.linear2 = nn.Linear(in_dims, in_dims)
        self.linear3 = nn.Linear(in_dims, in_dims)
        self.linear4 = nn.Linear(in_dims, in_dims)
        self.linear5 = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        # x.size() = (N, in_dims)
        x = self.af(self.linear1(x))
        # x.size() = (N, in_dims)
        x = self.af(self.linear2(x))
        # x.size() = (N, in_dims)
        x = self.af(self.linear3(x))
        # x.size() = (N, in_dims)
        x = self.af(self.linear4(x))
        # x.size() = (N, in_dims)
        x = self.linear5(x)
        # x.size() = (N, DBPEDIA_EMBEDDING_DIMENSIONS)
        return x
