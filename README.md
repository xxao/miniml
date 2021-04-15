# MiniML

The *MiniML* is a super basic Python library for machine learning for beginners (as I am). It is by no means meant as a
stable, production platform, and most probably it is not even useful for anyone except me. In fact, it is rather my
small playground, or a way I am trying to learn this very interesting topic. Hopefully some more functionality will come
as I learn more. There are some simple [examples](https://github.com/xxao/miniml/tree/master/examples) you can check to
have some impression what can be done.

### Example: Non-linear regression

```python
import miniml
import numpy as np
from utils import plot_costs, plot_regression

# Adapted from:
# https://lucidar.me/en/neural-networks/curve-fitting-nonlinear-regression/

# init data
np.random.seed(3)
X = np.linspace(-10, 10, num=1000)
Y = 0.1*X*np.cos(X) + 0.1*np.random.normal(size=1000)

X_train = X.reshape((1, X.shape[0]))
Y_train = Y.reshape((1, Y.shape[0]))

# create model
model = miniml.Model(X_train.shape[0])
model.add(1, 'linear', 'plain')
model.add(64, 'relu', 'he')
model.add(32, 'relu', 'he')
model.add(1, 'linear', 'plain')

# init params
rate = 0.01
epochs = 1000

# train model
optimizer = miniml.Optimizer(cost='mse', epochs=epochs, init_seed=48, store=10, verbose=200)
costs = optimizer.train_adam(model, X_train, Y_train, rate)

# predict by model
plot_costs(costs, rate, epochs)
plot_regression(model, X_train, Y_train)

```

## Requirements

- [Python 3.7+](https://www.python.org)
- [Numpy](https://pypi.org/project/numpy/)
- [ [sklearn](https://scikit-learn.org/stable/) ] Optional, used to load example datasets.
- [ [matplotlib](https://pypi.org/project/matplotlib/) ] Optional, used to display examples.


## Installation

The *MiniML* library is fully implemented in Python. No additional compiler is necessary. After downloading the source
code just run the following command from the *miniml* folder:

```$ python setup.py install```

or

```$ pip install .```


## Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
