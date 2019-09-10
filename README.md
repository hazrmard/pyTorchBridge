# pyTorchBridge

`pytorchbridge` provides a `scikit-learn Estimator` API for pytorch modules:

```python
from pytorchbridge import TorchEstimator
import torch.nn as nn
import torch.optim as optim

# Define pyTorch module
mymodule = nn.Sequential(
    nn.Conv2d(1,20,5),
    nn.ReLU(),
    nn.Conv2d(20,64,5),
    nn.ReLU()
)
# Cast module to device (if not CPU)
mymodule.to(torch.device('cuda'))
# define optimizer, AFTER casting module to device
myoptim = optim.SGD(mymodule.parameters(), lr=0.1, momentum=0.9)

estimator = TorchEstimator(module=mymodule,
                           optimizer=myoptim,
                           loss=nn.MSELoss(),
                           epochs=5,
                           verbose=False,
                           batch_size=32,
                           cuda=True)

estimator.fit(X, y)
estimator.predict(X)
```