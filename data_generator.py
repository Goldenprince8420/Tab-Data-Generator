from train import Trainer
from data import *
import torch


class DataGenerator:
    def __init__(self,
                 data,
                 epochs=500,
                 fair_epochs=10,
                 batch_size=64,
                 lambda_rate=0.5):
        super(DataGenerator, self).__init__()
        self._data = data
        self._trainer = Trainer(data=data,
                                epochs=epochs,
                                fair_epochs=fair_epochs,
                                batch_size=batch_size,
                                lambda_rate=lambda_rate)
        self._generator = None
        self._fake_df = None
        self._fake_arr = None

    def generate_new_data(self,
                          size_generate,
                          input_dims,
                          device,
                          is_trained=True):
        if not is_trained:
            self._trainer.train()
        else:
            self._generator = self._trainer.gen
            self._fake_arr = self._generator(torch.randn(size=(size_generate,
                                                               input_dims),
                                                         device=device))
            self._fake_df = get_generated_data(self._fake_arr,
                                               self._data,
                                               self._trainer.preprocess_handler.ohe,
                                               self._trainer.preprocess_handler.scaler)
            self._fake_df = self._fake_df[self._data.columns]
            return self._fake_df
