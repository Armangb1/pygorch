
# Gorch

Gorch is a lightweight neural network library for educational purposes. It provides basic functionalities for building and training neural networks using automatic differentiation.

## Installation

To install Gorch, clone the repository using the command below:

```sh
git clone https://github.com/Armangb1/pygorch.git
```

To install the required dependencies, use the following command:

```sh
pip install numpy
```


## Usage

After cloning the repository, create a Jupyter notebook and import Gorch as shown below:

```py
import gorch
```

### Creating Tensors

To use backward differentiation, create a `Tensor` object as follows:

```py
t1 = Tensor(data, required_grad=True)
```

Perform any operations on `t1`. At the end, use the `backward` method on the resulting tensor to compute the gradients:

```py
result.backward()
```

## Example

Here is an example of how to use Gorch in a simple neural network:

```py
from pygorch import gorch
import numpy as np

# Generate random input and output data
input = np.random.randn(1, 5)
output = np.random.randn()

# Create input and output tensor
x = gorch.Tensor(input)
y = gorch.Tensor(output)
# Initialize weights and biases
W1_d = np.random.randn(5, 1)
b1_d = np.random.randn(1, 1)
W1 = gorch.Tensor(W1_d, required_grad=True)
b1 = gorch.Tensor(b1_d, required_grad=True)

# Forward pass
net1 = x @ W1 + b1
O1 = net1.tanh()

# Compute error
e = output - O1

# Compute cost
cost = e.transpose()@e

# Backward pass
cost.backward()

# Get gradients
dcost_dW1 = W1.grad 
dcost_db1 = b1.grad
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or inquiries, please contact [arman.ghbn@gmail.com].

