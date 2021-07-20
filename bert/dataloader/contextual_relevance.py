import json
import pandas as pd


class ContextualRelevance:
    def __init__(self, file):
        self.train, self.test, self.false_negatives_test = self.init_data(file)

    def init_data(self, data_path):
        with open(data_path, encoding="utf8") as json_file:
            data = json.load(json_file)

        return pd.DataFrame(data['train']), pd.DataFrame(data['test']), int(data['false_negatives_test'])

    def get_data(self):
        return self.train, self.test, self.false_negatives_test
