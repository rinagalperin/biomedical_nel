import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#from bert.umls_classification.cfg import is_baseline


class ContextualRelevance:
    def __init__(self, file, is_baseline=True):
        self.train, self.test = self.init_data(file, is_baseline)

    def init_data(self, data_path, is_baseline):
        with open(data_path, encoding="utf8") as json_file:
            data = json.load(json_file)
        if is_baseline:
            data = data['baseline']
        else:
            data = data['data']

        return pd.DataFrame(data['train']), pd.DataFrame(data['test'])

    def get_data(self):
        return self.train, self.test

#
# dataloder = ContextualRelevance('../../training_data/training_data_4.json')
# train, test = dataloder.get_data()
#
# print(len(train['Labels']))
# print(test)
# print(test['Annotations_UMLS'].tolist())
# print(len(train['Annotations_UMLS'].tolist()))