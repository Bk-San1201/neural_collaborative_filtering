""""
Created on Aug 8, 2016
Processing datasets.

@author: Lam San
"""

import numpy as np
import scipy.sparse as sp


class Dataset(object):
    def __init__(self, path):
        """
            Constructor
        """
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")

        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        mat = np.loadtxt(filename, dtype=int)
        user_col = mat[..., 0]
        item_col = mat[..., 1]
        rating_col = mat[..., 2]
        num_users = np.max(user_col)
        num_items = np.max(item_col)
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        for i in range(len(user_col)):
            if rating_col[i] > 0:
                mat[user_col[i], item_col] = 1.0
        return mat

    def load_rating_file_as_list(self, filename):
        mat = np.loadtxt(filename, dtype=int)
        return mat[..., :2]

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList


