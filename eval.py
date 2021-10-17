from dataclasses import dataclass
from typing import Callable, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.base import ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm

from dataset import Dataset, Subdataset


@dataclass
class SklearnEvaluator:
    model_generator: Callable[[int], ClassifierMixin]
    model6_generator: Callable[[int], ClassifierMixin]
    parameter_name: str
    index: int
    dataset: Dataset
    best_accuracy: List[float]
    best_accuracy6: List[float]

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, param_range: Iterable[int]):
        params = list(param_range)
        accTest: List[float] = []
        accTest6: List[float] = []
        f1Test: List[float] = []
        f1Test6: List[float] = []
        for param in tqdm(params):
            model = self.model_generator(param)
            model6 = self.model6_generator(param)
            self._predict_accuracy_and_f1(model, accTest, f1Test,
                                          self.dataset.normal())
            self._predict_accuracy_and_f1(model6, accTest6, f1Test6,
                                          self.dataset.six())
        best_param = params[accTest6.index(max(accTest6))]
        best_model = self.model6_generator(best_param)
        best_model.fit(self.dataset.X6_train, self.dataset.y6_train)
        self._print_and_plot_results(best_model, accTest, accTest6, f1Test,
                                     f1Test6, params)

    def _predict_accuracy_and_f1(self, model: ClassifierMixin,
                                 accTest: List[float], f1Test: List[float],
                                 dataset: Subdataset):
        model.fit(dataset.X_train, dataset.y_train)
        y_pred = model.predict(dataset.X_test)
        acc = metrics.accuracy_score(dataset.y_test, y_pred)
        accTest.append(acc)
        f1 = metrics.f1_score(dataset.y_test,
                              y_pred,
                              average="macro",
                              labels=np.unique(y_pred))
        f1Test.append(f1)

    def _print_and_plot_results(self, model: ClassifierMixin,
                                accTest: List[float], accTest6: List[float],
                                f1Test: List[float], f1Test6: List[float],
                                params: List[int]):
        max_val_index = accTest.index(max(accTest))
        max_val6_index = accTest6.index(max(accTest6))
        max_acc = accTest[max_val_index]
        max_acc6 = accTest6[max_val6_index]
        self.best_accuracy[self.index] = max_acc
        self.best_accuracy6[self.index] = max_acc6

        plt.figure(figsize=(14, 6))

        plt.subplot(2, 2, 1)
        plt.plot(params, f1Test6, label="Using the first 6 gestures")
        plt.plot(params, f1Test, label="Using all gestures")
        plt.title("F1 Macro")
        plt.ylabel("F1 Macro")
        plt.xlabel(self.parameter_name)
        plt.legend(loc="lower right")

        plt.subplot(2, 2, 3)
        plt.plot(params, accTest6, label="Using the first 6 gestures")
        plt.plot(params, accTest, label="Using all gestures")
        plt.title("Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel(self.parameter_name)
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        y_pred = model.predict(self.dataset.X6_test)
        cm = confusion_matrix(self.dataset.y6_test, y_pred)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(
            cm,
            index=range(1, 7),
            columns=range(1, 7),
        )
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                     rotation=0,
                                     ha="right",
                                     fontsize=14)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                     rotation=45,
                                     ha="right",
                                     fontsize=14)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title("Confusion Matrix using 6 gestures\n and %s set to %d " %
                  (self.parameter_name, params[max_val6_index]))

        plt.tight_layout(pad=0.4, w_pad=3, h_pad=1)
        plt.show()

        print(
            "Max accuracy is {:.3f},".format(max_acc),
            "with F1 score {:.3f},".format(f1Test[max_val_index]),
            "using all gestures and",
            self.parameter_name,
            "set to",
            params[max_val_index],
            end=".\n",
        )
        print(
            "Max accuracy is {:.3f},".format(max_acc6),
            "with F1 score {:.3f},".format(f1Test6[max_val6_index]),
            "using 6 gestures and",
            self.parameter_name,
            "set to",
            params[max_val6_index],
            end=".",
        )
