import torch
from torch import nn


class FairLoss(nn.Module):
    def __init__(self,
                 S_start_index,
                 Y_start_index,
                 underprivileged_index,
                 privileged_index,
                 undesire_index,
                 desire_index,
                 lambda_rate):
        super(FairLoss, self).__init__()
        self._s_start_idx = S_start_index
        self._y_start_idx = Y_start_index

        self._underpriv_idx = underprivileged_index
        self._priv_idx = privileged_index

        self._undesire_idx = undesire_index
        self._desire_idx = desire_index

        self._lamb_rate = lambda_rate

        self.G = None
        self.I = None
        self._fake_pred = None
        self.f_loss = None

    def forward(self,
                input_data,
                fake_pred):
        x = input_data
        self._fake_pred = fake_pred
        self.G = x[:, self._s_start_idx:self._s_start_idx + 2]
        self.I = x[:, self._y_start_idx:self._y_start_idx + 2]
        self.f_loss = -1.0 * self._lamb_rate * \
                      (torch.mean(self.G[:, self._underpriv_idx] *
                                  self.I[:, self._desire_idx]) / (
                          x[:, self._s_start_idx + self._underpriv_idx].sum()) -
                       torch.mean(self.G[:, self._priv_idx] * self.I[:, self._desire_idx]) /
                       (x[:, self._s_start_idx + self._priv_idx].sum())) - 1.0 * torch.mean(fake_pred)
        return self.f_loss


def get_generator_loss(fake_pred):
    gen_loss = -1.0 * torch.mean(fake_pred)
    return gen_loss


def get_critic_loss(fake_pred,
                    real_pred,
                    gp,
                    c_lambda):
    crit_loss = torch.mean(fake_pred) - torch.mean(real_pred) +\
                c_lambda * gp
    return crit_loss
