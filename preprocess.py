from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer


class PreProcessor:
    def __init__(self, data, params):
        self.frame = data
        self.cont_col_list = list(self.frame.select_dtypes(['float', 'integer']).columns)
        self.cat_col_list = list(self.frame.select_dtypes('object').columns)
        self.scaler = QuantileTransformer(n_quantiles = 2000,
                                          output_distribution = 'uniform')
        self.ohe = OneHotEncoder()
        self.frame_int = None
        self.frame_cat = None
        self.frame_cat_encoded = None
        self.cat_lens = None
        self.discrete_cols_orderdict = None
        self.s_start_index = None
        self.y_start_index = None
        self.S = params['S']
        self.Y = params['Y']
        self.s_under = params['S_under']
        self.y_desire = params['Y_desire']
        self.underpriv_index = None
        self.priv_index = None
        self.desire_index = None
        self.undesire_index = None
        self.encoded_array = None
        self.df_ohe_int = None
        self.df_ohe_cats = None
        self.fake_int = None
        self.fake_cat = None
        self.fake_frame = None

    def process_numerical(self):
        self.frame_int = self.scaler.fit_transform(self.frame_int)

    def process_categorical(self):
        self.frame_cat_encoded = self.ohe.fit_transform(self.frame_cat)

    def set_indexes(self):
        self.cat_lens = [i.shape[0] for i in self.ohe.categories_]
        self.discrete_cols_orderdict = OrderedDict(zip(self.cat_col_list, self.cat_lens))
        self.s_start_index = len(self.cont_col_list) + \
                             sum(list(self.discrete_cols_orderdict.values()))[
                                 :list(self.discrete_cols_orderdict.keys()).index(self.S)
                             ]
        self.y_start_index = len(self.cont_col_list) + \
                             sum(list(self.discrete_cols_orderdict.values()))[
                             :list(self.discrete_cols_orderdict.keys()).index(self.Y)
                             ]
        if self.ohe.categories_[
            list(self.discrete_cols_orderdict.keys()).index(self.S)
        ][0] == self.s_under:
            self.underpriv_index = 0
            self.priv_index = 1
        else:
            self.underpriv_index = 1
            self.priv_index = 0

        if self.ohe.categories_[
            list(self.discrete_cols_orderdict.keys()).index(self.Y)
        ][0] == self.y_desire:
            self.desire_index = 0
            self.underpriv_index = 1
        else:
            self.desire_index = 1
            self.underpriv_index = 0

    def get_final_array(self):
        self.encoded_array = np.hstack((self.frame_int, self.frame_cat_encoded.toarray()))
        return self.encoded_array

    def get_original_dataset(self, transformed_data):
        self.df_ohe_int = transformed_data[:, :self.frame.select_dtypes([
            'float', 'integer']
        ).shape[1]]
        self.df_ohe_int = self.scaler.inverse_transform(self.df_ohe_int)
        self.df_ohe_cats = transformed_data[:, self.frame.select_dtypes(
            ['float', 'integer']
        ).shape[1]:]
        self.df_ohe_cats = self.ohe.inverse_transform(self.df_ohe_cats)
        self.fake_int = pd.DataFrame(self.df_ohe_int,
                                     columns = self.frame.select_dtypes(
                                         ['float', 'integer']).columns
                                     )
        self.fake_cat = pd.DataFrame(self.df_ohe_cats,
                                     columns = self.frame.select_dtypes('object').columns
                                     )
        self.fake_frame = pd.concat([self.fake_int, self.fake_cat], axis = 1)

    def preprocess(self):
        self.process_numerical()
        self.process_categorical()
        self.set_indexes()
        self.get_final_array()
