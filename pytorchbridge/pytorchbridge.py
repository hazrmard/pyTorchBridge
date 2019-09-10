"""
Defines the `TorchEstimator` class which provides a Scikit-learn Estimator API
for pytorch modules.
"""
from typing import Iterable, Tuple, Iterator, Union

import numpy as np
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange, tqdm



class TorchEstimator(BaseEstimator):
    """
    Wraps a `torch.nn.Module` instance with a scikit-learn `Estimator` API.
    """

    def __init__(self, module: nn.Module=None,
                 optimizer: optim.Optimizer=None,
                 loss: nn.modules.loss._Loss=None, epochs: int=10, verbose=False,
                 batch_size: int=8, cuda=True):
        """        
        Keyword Arguments:
            module {torch.nn.Module} -- A `nn.Module` describing the neural network,
            optimizer {torch.optim.Optimizer} -- An `Optimizer` instance which
                iteratively modifies weights,
            loss {torch.nn._Loss} -- a `_Loss` instance which calculates the loss metric,
            epochs {int} -- The number of times to iterate over the training data,
            verbose {bool} -- Whether to log training progress or not,
            batch_size {int} -- Chunk size of data for each training step,
            cuda {bool} -- Whether to use GPU acceleration if available.
        """
        self.module = module
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.cuda = cuda
        # pylint: disable=no-member
        self._device = torch.device('cpu')
        self._batch_first = None
        self._dtype = torch.float



    def _init(self, X, y):
        """
        Initializes internal parameters before fitting, including device, data
        types for network parameters.
        
        Arguments:
            X {torch.Tensor} -- Features
            y {torch.Tensor} -- Targets
        """
        # pylint: disable=no-member
        # Create a linear model if no module provided
        if self.module is None:
            self._device = torch.device('cuda') if \
                           torch.cuda.is_available() and self.cuda \
                           else torch.device('cpu')
            _, _, infeatures = self._get_shape(X)
            _, _, outfeatures = self._get_shape(y)

            class MyModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(infeatures, outfeatures)
                    self.squeeze = len(torch.as_tensor(y).size()) == 1

                def forward(self, x):
                    x = self.linear(x)
                    if self.squeeze:
                        return torch.squeeze(x)
                    return x

            self.module = MyModule()
            self.module.to(self._device)
        else:
            self._device = next(self.module.parameters()).device
        
        self._dtype = next(self.module.parameters()).dtype

        if self.optimizer is None:
            self.optimizer = optim.SGD(self.module.parameters(), lr=0.1)
        if self.loss is None:
            self.loss = nn.MSELoss()


    def fit(self, X: torch.Tensor, y: torch.Tensor) -> 'TorchEstimator':
        """
        Fit target to features.

        Arguments:
        X {torch.Tensor} -- `Tensor` of shape (SeqLen, N, Features) or (N, SeqLen, Features)
            for recurrent modules or (N, Features) for other modules.
        y {torch.Tensor} -- `Tensor` of shape ([SeqLen,] N, OutputFeatures) for recurrent
            modules of (N, OutputFeatures).

        Returns:
        self
        """
        # pylint: disable=no-member
        self._init(X, y)

        if self.verbose:
            print()
        ranger = trange(self.epochs)

        self._batch_first = self._is_batch_first()
        for e in ranger:
            total_loss = 0.

            for instance, target in zip(self._to_batches(X), self._to_batches(y)):
                instance, target = instance.to(self._device), target.to(self._device)
                self.module.zero_grad()
                output = self.module(instance)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if self.verbose:
                ranger.write(f'Epoch {e+1:3d}\tLoss: {total_loss:10.2f}')
        return self


    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict output from inputs.

        Arguments:
        X {torch.Tensor} -- `Tensor` of shape (SeqLen, N, Features) or (N, SeqLen, Features)
            for recurrent modules or (N, Features) for other modules.

        Returns:
        torch.Tensor -- of shape ([SeqLen,] N, OutputFeatures) for recurrent
            modules of (N, OutputFeatures).
        """
        # pylint: disable=no-member
        is_numpy = isinstance(X, np.ndarray)
        X = torch.as_tensor(X, dtype=self._dtype, device=self._device)
        with torch.no_grad():
            result = self.module(X)
        if is_numpy:
            return result.numpy()
        return result


    def score(self, X, y) -> float:
        """
        Measure how well the estimator learned through the coefficient of
        determination.

        Arguments:
        X {torch.Tensor} -- `Tensor` of shape (SeqLen, N, Features) or (N, SeqLen, Features)
            for recurrent modules or (N, Features) for other modules.
        y {torch.Tensor} -- `Tensor` of shape ([SeqLen,] N, OutputFeatures) for recurrent
            modules of (N, OutputFeatures).

        Returns:
        float -- Coefficient of determination.
        """
        y_pred = self.predict(X)
        residual_squares_sum = ((y - y_pred) ** 2).sum()
        total_squares_sum = ((y - y.mean()) ** 2).sum()
        return (1 - residual_squares_sum / total_squares_sum).item()


    def _to_batches(self, X: torch.Tensor) -> Iterator[torch.Tensor]:
        """
        Convert ([SeqLen,] N, Features) to a generator of ([SeqLen,] n, Features)
        mini-batches. So for recurrent layers, training can be done in batches.
        """
        # pylint: disable=no-member
        if isinstance(X, np.ndarray):
            X = X.astype(float)
        X = torch.as_tensor(X, dtype=self._dtype)
        if not self._batch_first:
            # Recurrent layers take inputs of the shape (SeqLen, N, Features...)
            # So if there is any recurrent layer in the module, assume that this
            # is the expected input shape
            # N = X[0].size()[0]
            N = len(X[0])
            nbatches = N // self.batch_size + (1 if N % self.batch_size else 0)
            for i in range(nbatches):
                yield X[:, i*self.batch_size:(i+1)*self.batch_size]
        else:
            # Fully connected layers take inputs of the shape (N, Features...)
            N = len(X)
            nbatches = N // self.batch_size + (1 if N % self.batch_size else 0)
            for i in range(nbatches):
                yield X[i*self.batch_size:(i+1)*self.batch_size]


    def _is_recurrent(self) -> bool:
        """
        Checks whether the network has any recurrent units.
        """
        return any(map(lambda x: isinstance(x, nn.RNNBase), self.module.modules()))


    def _is_batch_first(self) -> bool:
        """
        Checks whether the features arrays are in the shape (Batch, ..., Features) or
        (..., Batch, Features).
        """
        # Default setting is batch_first=False for RNNBase subclasses
        return any(map(lambda x: getattr(x, 'batch_first', True), self.module.modules()))


    def _get_shape(self, t: torch.Tensor) -> Tuple[int, int, int]:
        """
        Get size of each dimension of tensor depending on `batch_first`. The
        size is returned in order of time, batch, features.
        
        Arguments:
            t {torch.Tensor} -- A Tensor

        Returns:
            Tuple[int, int, int] -- A tuple of [time, batch, feature]
            size of the tensor as interpreted by th estimator.
        """
        # pylint: disable=no-member
        if isinstance(t, np.ndarray):
            sz = t.shape
        else:
            t = torch.as_tensor(t)
            sz = t.size()
        ndims = len(sz)
        if ndims == 1:
            return 0, sz[0], 1
        if ndims == 2:
            return 0, sz[0], sz[1]
        elif ndims == 3:
            if self._is_batch_first():
                return sz[1], sz[0], sz[2]
            else:
                return sz[0], sz[1], sz[2]
