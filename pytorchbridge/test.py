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


    def create_arrays(self, recurrent=False, batch_first=True, tensor=True):
        t, n, fin, fout = 1, 10, 2, 1      # time, batch, features
        if recurrent:
            if batch_first:
                X = np.random.rand(n, t, fin)
                y = np.random.rand(n, fout)
            else:
                X = np.random.rand(t, n, fin)
                y = np.random.rand(n, fout)
        else:
            X = np.random.rand(n, fin)
            y = np.random.rand(n, fout)
        if tensor:
            X = torch.as_tensor(X)
            y = torch.as_tensor(y)
        return X, y


    def test_numpy_dense(self):
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        est = TorchEstimator()
        est.fit(X, y).predict(X)


    def test_numpy_recurrent(self):
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
