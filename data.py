from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import PreProcessor

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class Dataset:
    def __init__(self, data, batch_size, params):
        super(Dataset, self).__init__()
        self.arr = None
        self._input_dims = None
        self.batch_size = batch_size
        self.params = params
        self.preprocessor = PreProcessor(data, self.params)
        self.x_train = None
        self.x_test = None
        self.torch_tensor_train = None
        self.torch_tensor_test = None
        self.train_ds = None
        self.train_dl = None
        self.test_ds = None
        self.test_dl = None

    def preprocess_data(self):
        self.preprocessor.process_numerical()
        self.preprocessor.process_categorical()
        self.preprocessor.set_indexes()
        self.arr = self.preprocessor.get_final_array()

    def split_data(self):
        self._input_dims = self.arr.shape[1]
        self.x_train, self.x_test = train_test_split(self.arr,
                                                     test_size = 0.1,
                                                     shuffle = True)

    def create_tensors(self):
        self.torch_tensor_train = torch.from_numpy(self.x_train.copy()).float()
        self.train_ds = TensorDataset(self.torch_tensor_train)
        self.train_dl = DataLoader(self.train_ds,
                                   batch_size = self.batch_size,
                                   drop_last = True)

        self.torch_tensor_test = torch.from_numpy(self.x_train.copy()).float()
        self.test_ds = TensorDataset(self.torch_tensor_test)
        self.test_dl = DataLoader(self.test_ds,
                                   batch_size=1,
                                   drop_last=True)

    def get_training_data(self):
        self.preprocess_data()
        self.split_data()
        self.create_tensors()
        return self.train_dl, self.test_dl, self.x_train.copy(), self.x_test.copy()

    @property
    def input_dims(self):
        return self._input_dims


def get_generated_data(df_transformed, df_orig, ohe, scaler):
    df_ohe_int = df_transformed[:, :df_orig.select_dtypes(['float', 'integer']).shape[1]]
    df_ohe_int = scaler.inverse_transform(df_ohe_int)
    df_ohe_cats = df_transformed[:, df_orig.select_dtypes(['float', 'integer']).shape[1]:]
    df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
    # df_income = df_transformed[:,-1]
    # df_ohe_cats = np.hstack((df_ohe_cats, df_income.reshape(-1,1)))
    df_int = pd.DataFrame(df_ohe_int, columns=df_orig.select_dtypes(['float', 'integer']).columns)
    df_cat = pd.DataFrame(df_ohe_cats, columns=df_orig.select_dtypes('object').columns)
    return pd.concat([df_int, df_cat], axis=1)
