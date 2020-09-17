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
from torch.utils.data import DataLoader
from tqdm.auto import trange



# Defining custom types
TensorLike = Union[torch.Tensor, np.ndarray]



class TorchEstimator(BaseEstimator):
    """
    Wraps a `torch.nn.Module` instance with a scikit-learn `Estimator` API.

    *Note:*
        All parameters in the provided module must have the same data type and
        device. The `_init()` method uses the dtype/device of the first element
        in `module.parameters()` to set default casting options.
    """

    def __init__(self, module: nn.Module=None,
                 optimizer: optim.Optimizer=None,
                 loss: nn.modules.loss._Loss=None, epochs: int=2, tol: float=1e-4,
                 max_tol_iter: int=5, verbose=False,
                 early_stopping: bool=False, validation_fraction: float=0.1,
                 batch_size: int=8, cuda=True, return_hidden: bool=False):
        """        
        Parameters
        ----------
        module: torch.nn.Module
            A `nn.Module` describing the neural network,
        optimizer: torch.optim.Optimizer
            An `Optimizer` instance which iteratively modifies weights,
        loss: torch.nn._Loss
            A `_Loss` instance which calculates the loss metric,
        epochs: int
            The number of times to iterate over the training data,
        tol: float
            Tolerance for loss between epochs.
        max_tol_iter: int
            If loss does not change by `tol` for `max_tol_iter`, training is
            stopped.
        early_stopping: bool, optional
            Whether to stop training early if validation score does not improve
            by `tol` for `max_tol_iter` iterations. By default False.
        validation_fraction: float, optional
            The fraction of the training data to be set aside for validation if
            `early_stopping=True`. By default 0.1.
        verbose: bool
            Whether to log training progress or not,
        batch_size: int
            Chunk size of data for each training step,
        cuda: bool
            Whether to use GPU acceleration if available.
        return_hidden: bool
            Whether to return hidden and cell state tuple for recurrent
            networks. (default: False)
        """
        self.module = module
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.tol = tol
        self.max_tol_iter = max_tol_iter
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.batch_size = batch_size
        self.cuda = cuda
        self.return_hidden = return_hidden
        # pylint: disable=no-member
        self._device = torch.device('cpu')
        self._batch_first = None
        self._dtype = torch.float



    def _init(self, X: TensorLike, y: TensorLike):
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
                    self.squeeze = y.ndim == 1

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
        self._batch_first = all(map(lambda x: getattr(x, 'batch_first', True),
                                    self.module.modules()))

        if self.optimizer is None:
            self.optimizer = optim.SGD(self.module.parameters(), lr=0.01)
        if self.loss is None:
            self.loss = nn.MSELoss()


    def parameters(self) -> Iterator[torch.Tensor]:
        """
        Convenience method for `self.module.parameters()`.

        Returns
        -------
        Iterator
            Iterator over a module's parameters.
        """        
        return self.module.parameters()


    def fit(self, X: Union[TensorLike, DataLoader], y: TensorLike=None, \
            **kwargs) -> 'TorchEstimator':
        """
        Fit target to features.

        Arguments:
        X {torch.Tensor} -- `Tensor` of shape (SeqLen, N, Features) or (N, SeqLen, Features)
            for recurrent modules or (N, Features) for other modules.
        y {torch.Tensor} -- `Tensor` of shape ([SeqLen,] N, OutputFeatures) for recurrent
            modules of (N, OutputFeatures). Optional if X is a `DataLoader` which
            already contains features and targets.
        **kwargs -- Keyword arguments passed to `self.module(X, **kwargs)`

        Returns:
        self
        """
        # TODO: Add instance weights to super-/sub-sample training data.
        # pylint: disable=no-member
        self._init(X, y)

        # Setup train/validation split for early stopping
        if self.early_stopping and self.validation_fraction > 0.:
            if self._batch_first:
                vidx = np.random.choice(len(X),
                                        size=int(self.validation_fraction * len(X)))
                Xval, yval = X[vidx], y[vidx]
                idx = np.asarray(list(set(range(len(X))) - set(vidx)), dtype=int)
                X, y = X[idx], y[idx]

            else:
                vidx = np.random.choice(len(X[0]),
                                        size=int(self.validation_fraction * len(X[0])))
                Xval, yval = X[:, vidx], y[:, vidx]
                idx = np.asarray(set(range(len(X[0]))) - set(vidx), dtype=int)
                X, y = X[idx], y[idx]
            Xval = torch.as_tensor(Xval, dtype=self._dtype, device=self._device)
            yval = torch.as_tensor(yval, dtype=self._dtype, device=self._device)

        if self.verbose:
            print()
        ranger = trange(self.epochs, leave=False)
        loss_hist = []
        for e in ranger:
            total_loss = 0.

            if isinstance(X, DataLoader):
                iterable = X
            else:
                iterable = zip(self._to_batches(X), self._to_batches(y))

            for instance, target in iterable:
                instance, target = instance.to(self._device), target.to(self._device)
                total_loss += self.partial_fit(instance, target, **kwargs)

            if self.verbose:
                ranger.write(f'Epoch {e+1:3d}\tLoss: {total_loss:10.2f}')

            if self.early_stopping and self.validation_fraction > 0.:
                output = self.module(Xval, **kwargs)
                total_loss = self.loss(output, yval).item()

            loss_hist.append(total_loss)
            if len(loss_hist) > 1:
                # Get last max_tol_iter + 1 loss values
                arr_loss = np.asarray(loss_hist[-self.max_tol_iter - 1:])
                # Get last max_tol_iter changes in loss
                delta_loss = np.abs(arr_loss[1:] - arr_loss[:-1])
                # Check if last max_tol_iter changes are < tolerance
                thresh_loss = delta_loss < self.tol
                if sum(thresh_loss) == self.max_tol_iter:
                    ranger.close()
                    break
        return self


    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> float:
        """
        Fit a single batch of tensors to the module. No type/device checking is
        done.

        Parameters
        ----------
        X : torch.Tensor
            A tensor containing features.
        y : torch.Tensor
            A tensor containing targets.

        Returns
        -------
        float
            The loss of the targets and module outputs.
        """        
        self.module.zero_grad()
        output = self.module(X, **kwargs)
        # For recurrent networks, the outputs may also return
        # a tuple of hidden states/cell states
        if isinstance(output, tuple):
            output, _ = output
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def predict(self, X: TensorLike, *args, **kwargs) -> torch.Tensor: 
        """
        Predict output from inputs.

        Parameters
        ----------
        X : torch.Tensor
            Tensor` of shape (SeqLen, N, Features) or (N, SeqLen, Features)
            for recurrent modules or (N, Features) for other modules.
        *args
            Positional arguments passed  to `self.module(X, *args, **kwargs)`
        **kwargs
            Keyword arguments passed to `self.module(X, *args, **kwargs)`

        Returns
        -------
        torch.Tensor
            Of shape ([SeqLen,] N, OutputFeatures) for recurrent
            modules of (N, OutputFeatures).
        """
        # pylint: disable=no-member
        is_numpy = isinstance(X, np.ndarray)
        X = torch.as_tensor(X, dtype=self._dtype, device=self._device)
        with torch.no_grad():
            result = self.module(X, *args, **kwargs)
            if isinstance(result, tuple) and not self.return_hidden:
                result = result[0]    # recurrent layers return (output, hidden)

        if is_numpy:
            # TODO: Iterate over arbitrarily nested tuples of returned tensors
            if isinstance(result, tuple):       # If hidden units are returned
                h = result[0].cpu().numpy()
                if isinstance(result[1], tuple):# LSTM case
                    hn = result[1][0].cpu()     # Secondary results are not
                    cn = result[1][1].cpu()     # type-converted
                    return h, (hn, cn)
                else:                           # RNN/GRU case
                    hn = result[1].cpu()
                    return h, hn
            else:                               # If only result is returned
                return result.cpu().numpy()
        return result


    def score(self, X: TensorLike, y: TensorLike, **kwargs) -> float:
        """
        Measure how well the estimator learned through the coefficient of
        determination.

        Arguments:
        X {torch.Tensor} -- `Tensor` of shape (SeqLen, N, Features) or (N, SeqLen, Features)
            for recurrent modules or (N, Features) for other modules.
        y {torch.Tensor} -- `Tensor` of shape ([SeqLen,] N, OutputFeatures) for recurrent
            modules of (N, OutputFeatures).
        **kwargs -- Keyword arguments passed to `self.module(X, **kwargs)`

        Returns:
        float -- Coefficient of determination.
        """
        # pylint: disable=no-member
        X = torch.as_tensor(X, dtype=self._dtype, device=self._device)
        y = torch.as_tensor(y, dtype=self._dtype, device=self._device)
        y_pred = self.predict(X, **kwargs)
        residual_squares_sum = ((y - y_pred) ** 2).sum()
        total_squares_sum = ((y - y.mean()) ** 2).sum()
        return (1 - residual_squares_sum / total_squares_sum).item()


    def _to_batches(self, X: TensorLike) -> Iterator[torch.Tensor]:
        """
        Convert ([SeqLen,] N, Features) to a generator of ([SeqLen,] n, Features)
        mini-batches. So for recurrent layers, training can be done in batches.
        """
        # pylint: disable=no-member
        if isinstance(X, np.ndarray):
            X = X.astype(float)
        X = torch.as_tensor(X, dtype=self._dtype)
        if not self._batch_first and X.ndim > 2:
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


    def _get_shape(self, t: TensorLike) -> Tuple[int, int, int]:
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
            if self._batch_first:
                return sz[1], sz[0], sz[2]
            else:
                return sz[0], sz[1], sz[2]

