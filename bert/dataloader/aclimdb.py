import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import re


class aclImdb:
    def __init__(self):
        train, test = self.download_and_load_datasets()
        self.train = train.sample(5000)
        self.test = test.sample(5000)

    def get_data(self):
        return self.train, self.test

    # Load all files from a directory in a DataFrame.
    def load_directory_data(self, directory):
        data = {}
        data["sentence"] = []
        data["sentiment"] = []
        for file_path in os.listdir(directory):
            with tf.compat.v1.gfile.GFile(os.path.join(directory, file_path), "r") as f:
                data["sentence"].append(f.read())
                data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
        return pd.DataFrame.from_dict(data)

    # Merge positive and negative examples, add a polarity column and shuffle.
    def load_dataset(self, directory):
        pos_df = self.load_directory_data(os.path.join(directory, "pos"))
        neg_df = self.load_directory_data(os.path.join(directory, "neg"))
        pos_df["polarity"] = 1
        neg_df["polarity"] = 0
        return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

    # Download and process the dataset files.
    def download_and_load_datasets(self, force_download=False):
        dataset = tf.keras.utils.get_file(
            fname="aclImdb.tar.gz",
            origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            extract=True)

        train_df = self.load_dataset(os.path.join(os.path.dirname(dataset),
                                             "aclImdb", "train"))
        test_df = self.load_dataset(os.path.join(os.path.dirname(dataset),
                                            "aclImdb", "test"))

        return train_df, test_df
