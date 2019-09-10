from unittest import main, TestCase

import torch
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import GridSearchCV

from .pytorchbridge import TorchEstimator



class TestAPI(TestCase):


    # def test_api(self):
    #     check_estimator(TorchEstimator())


    def test_numpy_arrays(self):
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        est = TorchEstimator()
        est.fit(X, y).predict(X)


    def test_torch_tensors(self):
        # pylint: disable=no-member
        X = torch.from_numpy(np.random.rand(10, 2))
        y = torch.from_numpy(np.random.rand(10))
        est = TorchEstimator()
        est.fit(X, y).predict(X)


    def grid_search(self):
        pass


if __name__ == '__main__':
    main()
