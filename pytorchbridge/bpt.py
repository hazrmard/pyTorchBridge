"""
TorchEstimator which implements truncated backpropagation through time.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange

from .pytorchbridge import TorchEstimator



class RecurrentTorchEstimator(TorchEstimator):


    def __init__(self, module: nn.Module=None,
                 optimizer: optim.Optimizer=None,
                 loss: nn.modules.loss._Loss=None, epochs: int=2, verbose=False,
                 batch_size: int=8, cuda=True,
                 bpt_every: int=None, bpt_for: int=None):
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
        super().__init__(
            module=module,
            optimizer=optimizer,
            loss=loss,
            epochs=epochs,
            verbose=verbose,
            batch_size=batch_size,
            cuda=cuda
        )
        
        self.bpt_every = bpt_every
        self.bpt_for = bpt_for


    def _init(self, X, y):
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
                    self.rnn = nn.RNN(infeatures, outfeatures)
                    self.squeeze = y.ndim == 1

                def forward(self, x, h0=None):
                    out, hidden = self.rnn(x, h0)
                    if self.squeeze:
                        return torch.squeeze(out)
                    return out, hidden

            self.module = MyModule()
            self.module.to(self._device)
        else:
            self._device = next(self.module.parameters()).device
        
        self._dtype = next(self.module.parameters()).dtype
        self._batch_first = any(map(lambda x: getattr(x, 'batch_first', False),
                                                      self.module.modules()))
        self._time_dim = 1 if self._batch_first else 0
        self._batch_dim = 0 if self._batch_first else 1

        if self.optimizer is None:
            self.optimizer = optim.SGD(self.module.parameters(), lr=0.1)
        if self.loss is None:
            self.loss = nn.MSELoss()


    def fit(self, X, y, **kwargs):
        # pylint: disable=no-member
        self._init(X, y)

        if self.verbose:
            print()
        ranger = trange(self.epochs, leave=False)
        for e in ranger:
            total_loss = 0.

            for instance, target in zip(self._to_batches(X), self._to_batches(y)):
                instance, target = instance.to(self._device), target.to(self._device)

                hidden_shape = list(target.shape)
                del hidden_shape[self._time_dim]
                post_hidden = instance.new_zeros(hidden_shape)      # batch, feature

                for t in range(0, self._get_shape(instance)[0], self.bpt_every):
                    pre_in = self._slice_time(instance, t, t + self.bpt_every - self.bpt_for)
                    post_in = self._slice_time(instance, t + self.bpt_every - self.bpt_for, t + self.bpt_every)
                    sub_target = self._slice_time(target, t, t + self.bpt_every)
                    if post_in.shape[self._time_dim] < self.bpt_for: break

                    pre_out, pre_hidden = self.module(pre_in, post_hidden, **kwargs)
                    pre_hidden = pre_hidden.detach()
                    post_out, post_hidden = self.module(post_in, pre_hidden, **kwargs)
                    post_hidden = post_hidden.detach()

                    sub_output = torch.cat((pre_out, post_out), dim=self._time_dim)
                    self.module.zero_grad()
                    loss = self.loss(sub_output, sub_target)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

            if self.verbose:
                ranger.write(f'Epoch {e+1:3d}\tLoss: {total_loss:10.2f}')
        return self


    def _slice_time(self, t: torch.Tensor, start: int=None, stop: int=None) \
        -> torch.Tensor:
        if t.ndim > 2:
            if self._time_dim == 1:
                return t[:, start:stop, :]
            else:
                return t[start:stop, :]
        return t