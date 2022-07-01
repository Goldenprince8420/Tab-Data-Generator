import torch
from torch import nn
import torch.nn.functional as f


class Generator(nn.Module):
    def __init__(self, input_dims,
                 continuous_cols,
                 discrete_cols,
                 gumbel_softmax_tau = 0.2):
        super(Generator, self).__init__()
        self._in_dims = input_dims
        self._cont_cols = continuous_cols
        self._dis_cols = discrete_cols
        self._num_cont_cols = len(self._cont_cols)
        self.tau = gumbel_softmax_tau

        self.dense1 = nn.Linear(in_features = self._in_dims,
                                out_features = self._in_dims)
        self.dense_num_ = nn.Linear(in_features = self._in_dims,
                                    out_features = self._num_cont_cols)
        self.dense_cat_ = nn.ModuleDict()

        for key, val in self._dis_cols.items():
            self.dense_cat_[key] = nn.Linear(in_features = self._in_dims,
                                             out_features = val)

    def forward(self, input_data):
        x = input_data
        x = self.dense1(x)
        x = f.relu(x)
        x_num = self.dense_num_(x)
        x_num = f.relu(x_num)

        x_cat = []
        for key in self.dense_cat_:
            temp = f.gumbel_softmax(self.dense_cat_[key](x), tau = self.tau)
            x_cat.append(temp)

        output = torch.cat([x_num, x_cat], dim = 1)
        return output
