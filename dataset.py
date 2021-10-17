import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import pywt
from tqdm import tqdm

from utils import dotdict


@dataclass
class Subdataset():
    X_train: List[np.ndarray]
    X_test: List[np.ndarray]
    y_train: List[int]
    y_test: List[int]


@dataclass
class Dataset():
    X_train: List[np.ndarray]
    X_test: List[np.ndarray]
    y_train: List[int]
    y_test: List[int]
    X6_train: List[np.ndarray]
    X6_test: List[np.ndarray]
    y6_train: List[int]
    y6_test: List[int]

    def normal(self) -> Subdataset:
        return Subdataset(self.X_train, self.X_test, self.y_train, self.y_test)

    def six(self) -> Subdataset:
        return Subdataset(self.X6_train, self.X6_test, self.y6_train,
                          self.y6_test)


def haar_transform(settings: dotdict,
                   sequence_array: pd.DataFrame) -> np.ndarray:
    x, y, z = (np.concatenate(
        pywt.wavedec(sequence_array.iloc[:, axis],
                     settings.wavelet_name,
                     level=3)) for axis in range(3))
    xyz = np.concatenate((x[:8], y[:8], z[:8]))
    return xyz


def read_dataset(settings: dotdict) -> Dataset:
    X_all: List[np.ndarray] = []
    y_all: List[int] = []
    X_six: List[np.ndarray] = []
    y_six: List[int] = []
    for user in tqdm(settings.users):
        user_path = os.path.join(settings.path, user)
        for class_num, gesture in enumerate(settings.gestures):
            path = os.path.join(user_path, gesture)
            for sequence in os.listdir(path):
                sequence_array = pd.read_csv(os.path.join(path, sequence),
                                             sep=" ",
                                             header=None,
                                             usecols=[3, 4, 5])
                X_all.append(haar_transform(settings, sequence_array))
                y_all.append(class_num)
                if class_num <= 5:
                    X_six.append(haar_transform(settings, sequence_array))
                    y_six.append(class_num)
    return Dataset(*split_dataset(X_all, y_all, settings.split_point),
                   *split_dataset(X_six, y_six, settings.split_point_6))


def split_dataset(
    X: List[np.ndarray], y: List[int], split_point: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    X_train, X_test = X[:split_point], X[split_point + 1:]
    y_train, y_test = y[:split_point], y[split_point + 1:]
    return (X_train, X_test, y_train, y_test)
