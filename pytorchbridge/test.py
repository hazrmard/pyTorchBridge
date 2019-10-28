from unittest import main, TestCase

import torch
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import GridSearchCV

from .pytorchbridge import TorchEstimator
from .bpt import RecurrentTorchEstimator


# pylint: disable=no-member
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


    def test_recurrent_estimator(self):
        X = torch.from_numpy(np.random.rand(10, 5, 2))
        y = torch.from_numpy(np.random.rand(10, 5, 1))
        est = RecurrentTorchEstimator(bpt_every=3, bpt_for=2)
        pred = est.fit(X, y).predict(X)
        self.assertSequenceEqual(pred.shape, (10, 5, 1))


    def test_truncated_bpt(self):
        X = torch.ones(10, 1, 1)
        y = np.cumsum(X, axis=0)
        est = RecurrentTorchEstimator(bpt_every=3, bpt_for=1)

    def grid_search(self):
        pass


if __name__ == '__main__':
    main()
