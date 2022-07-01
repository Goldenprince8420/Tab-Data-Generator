from data import *
from generator import *
from critic import *
from losses import *
from utils import *

import torch
from torch import nn
import matplotlib.pyplot as plt


class Trainer(nn.Module):
    def __init__(self,
                 data,
                 epochs = 500,
                 batch_size = 64,
                 fair_epochs = 10,
                 lambda_rate = 0.5,
                 lr_gen_opt = 0.0002,
                 lr_gen_opt_fair = 0.0001,
                 lr_crit_opt = 0.0002,
                 display_step = 50,
                 params = None,
                 device = "cpu"):
        super(Trainer, self).__init__()
        self.data = data
        self._epochs = epochs
        self._batch_size = batch_size
        self._fair_epochs = fair_epochs
        self._lambda_rate = lambda_rate
        self._lr_gen_opt = lr_gen_opt
        self._lr_gen_opt_fair = lr_gen_opt_fair
        self._lr_crit_opt = lr_crit_opt
        self._display_step = display_step
        self._device = device
        self._params = params
        self._train_params_dic = {}

        self._preprocess_handler = PreProcessor(self.data,
                                                self._params)
        self._data_handler = Dataset(self.data,
                                     self._batch_size,
                                     self._params)

    def _handle_preprocess(self):
        self._preprocess_handler.preprocess()
        self._ohe = self._preprocess_handler.ohe
        self._scaler = self._preprocess_handler.scaler
        self._disc_cols = self._preprocess_handler.discrete_cols_orderdict
        self._cont_cols = self._preprocess_handler.cont_col_list
        self._s_start_idx = self._preprocess_handler.s_start_index
        self._y_start_idx = self._preprocess_handler.y_start_index
        self._underpriv_idx = self._preprocess_handler.underpriv_index
        self._priv_idx = self._preprocess_handler.priv_index
        self._undesire_idx = self._preprocess_handler.undesire_index
        self._desire_idx = self._preprocess_handler.desire_index

    def _handle_dataset(self):
        self._train_dl, self._test_dl, self._data_train, self._data_test = \
            self._data_handler.get_training_data()
        self._in_dims = self._data_handler.input_dims

    def _get_generator(self):
        self._gen = Generator(self._in_dims,
                              self._cont_cols,
                              self._disc_cols).to(self._device)

    def _get_critic(self):
        self._crit = Critic(self._in_dims).to(self._device)
        self._second_crit = FairLoss(S_start_index = self._s_start_idx,
                                     Y_start_index = self._y_start_idx,
                                     underprivileged_index = self._underpriv_idx,
                                     privileged_index = self._priv_idx,
                                     undesire_index = self._undesire_idx,
                                     desire_index = self._desire_idx,
                                     lambda_rate = self._lambda_rate
                                     )

    def _get_optimizer(self):
        self._gen_opt = torch.optim.Adam(self._gen.parameters(),
                                         lr = self._lr_gen_opt,
                                         betas = (0.5, 0.999))
        self._gen_opt_fair = torch.optim.Adam(self._gen.parameters(),
                                              lr = self._lr_gen_opt_fair,
                                              betas = (0.5, 0.999))
        self._crit_opt = torch.optim.Adam(self._crit.parameters(),
                                          lr = self._lr_crit_opt,
                                          betas = (0.5, 0.999))

    def _train(self):
        self._handle_preprocess()
        self._handle_dataset()

        self._get_generator()
        self._get_critic()

        self._get_optimizer()
        critic_losses = []
        generator_losses = []
        cur_step = 0

        for epoch in range(self._epochs):
            print("Epoch {}".format(epoch + 1))
            if epoch + 1 <= (self._epochs - self._fair_epochs):
                print("Training for Accuracy...")
            if epoch + 1 > (self._epochs - self._fair_epochs):
                print("Training for Fairness...")

            for data in self._train_dl:
                data[0] = data[0].to(self._device)
                critic_rep = 4
                mean_iter_crit_loss = 0

                for _rep in range(critic_rep):
                    self._crit_opt.zero_grad()
                    fake_noise = torch.randn(size = (self._batch_size, self._in_dims),
                                             device = self._device).float()
                    fake = self._gen(fake_noise)

                    crit_fake_pred = self._crit(fake.detach())
                    crit_real_pred = self._crit(self.data[0])

                    epsilon = torch.randn(self._batch_size,
                                          self._in_dims,
                                          device = self._device,
                                          requires_grad = True)
                    grad = get_gradient(self._crit,
                                        self.data[0],
                                        fake.detach(),
                                        epsilon = epsilon)
                    grad_penalty = gradient_penalty(grad)

                    self._crit_loss = get_critic_loss(crit_fake_pred,
                                                      crit_real_pred,
                                                      grad_penalty,
                                                      c_lambda = 10)
                    mean_iter_crit_loss += self._crit_loss.item() / critic_rep
                    self._crit_loss.backward(retain_graph = True)
                    self._crit_opt.step()

                if cur_step > 50:
                    critic_losses += [mean_iter_crit_loss]

                if epoch + 1 <= (self._epochs - self._fair_epochs):
                    self._gen_opt.zero_grad()
                    fake_noise_2 = torch.randn(size = (self._batch_size, self._in_dims),
                                               device = self._device).float()
                    fake_2 = self._gen(fake_noise_2)
                    crit_fake_pred = self._crit(fake_2)

                    self._gen_loss = get_generator_loss(crit_fake_pred)
                    self._gen_loss.backward()
                    self._gen_opt.step()

                if epoch + 1 > (self._epochs - self._fair_epochs):
                    self._gen_opt_fair.zero_grad()
                    fake_noise_2 = torch.randn(size = (self._batch_size, self._in_dims),
                                               device = self._device).float()
                    fake_2 = self._gen(fake_noise_2)
                    crit_fake_pred = self._second_crit(fake_2)

                    self._gen_fair_loss = self._second_crit(fake_2,
                                                            crit_fake_pred,
                                                            self._lambda_rate)
                    self._gen_fair_loss.backward()
                    self._gen_opt_fair.step()
                    
                if cur_step > 50:
                    if epoch + 1 <= (self._epochs - self._fair_epochs):
                        generator_losses += [self._gen_loss.item()]
                    if epoch + 1 > (self._epochs - self._fair_epochs):
                        generator_losses += [self._gen_fair_loss.item()]

                        print("Current Step: {}".format(cur_step))

                if cur_step % self._display_step == 0 and cur_step > 0:
                    gen_mean = sum(generator_losses[-self._display_step:]) / self._display_step
                    crit_mean = sum(critic_losses[-self._display_step:]) / self._display_step
                    print("Step {}; Generator Loss: {}; Critic Loss: {}".format(cur_step, gen_mean, crit_mean))
                    step_bins = 20
                    num_examples = (len(generator_losses) // step_bins) * step_bins
                    plt.plot(range(num_examples // step_bins),
                             torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                             label = "Generator Loss")
                    plt.plot(range(num_examples // step_bins),
                             torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                             label = "Critic Loss")
                    plt.legend()
                    plt.show()

                cur_step += 1

    def get_train_params(self, is_trained = True):
        if is_trained:
            self._train_params_dic['Generator'] = self._gen
            self._train_params_dic['Critic'] = self._crit
        else:
            self.train()
            self._train_params_dic['Generator'] = self._gen
            self._train_params_dic['Critic'] = self._crit
        return self._train_params_dic

    @property
    def gen(self):
        return self._gen

    @property
    def preprocess_handler(self):
        return self._preprocess_handler
