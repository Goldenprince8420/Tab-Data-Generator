from torch import nn
import torch.nn.functional as f


class Critic(nn.Module):
    def __init__(self,
                 input_dim,
                 do_classify = False,
                 classifier_dim_if_do_classify = 256,
                 drop_rate_if_do_classify = 0.2):
        super(Critic, self).__init__()
        self._in_dims = input_dim
        self.do_classify = do_classify
        self._classifier_dim = classifier_dim_if_do_classify
        self._drop_rate = drop_rate_if_do_classify

        if self.do_classify:
            self._predense = nn.Linear(in_features = self._in_dims,
                                       out_features = self._classifier_dim)

        self._dense1 = nn.Linear(in_features = self._in_dims,
                                 out_features = self._in_dims)
        self._dense2 = nn.Linear(in_features = self.In_dims,
                                 out_features = self._in_dims)

        # For singular value classification
        if self.do_classify:
            self._dense3 = nn.Linear(in_features = self._classifier_dim,
                                     out_features = 1)
            self.drop = nn.Dropout(p = self._drop_rate)
            self._classifier_activation = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data
        if self.do_classify:
            x = self._predense(x)
        x = self._dense1(x)
        x = f.leaky_relu(x)

        x = self._dense2(x)
        x = f.leaky_relu(x)

        if self.do_classify:
            x = self.drop(x)
            x = self._dense3(x)
            x = f.leaky_relu(x)
            x = self._classifier_activation(x)

        return x
