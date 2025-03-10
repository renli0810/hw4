"""The module."""

from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION

        self.weight = init.kaiming_uniform(
            in_features, out_features, device=device, dtype=dtype
        )
        self.weight = Parameter(self.weight)
        if bias:
            bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = ops.reshape(bias, (1, out_features))
            self.bias = Parameter(self.bias)
        else:
            self.bias = None

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias is not None:
            bias = self.bias.broadcast_to(out.shape)
            out = out + bias
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        produce = 1
        if len(X.shape) > 1:
            for shape in X.shape[1:]:
                produce *= shape
            return ops.reshape(X, (X.shape[0], produce))
        else:
            return ops.reshape(X, X.shape)
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        log_sum = ops.logsumexp(logits, axes=1)
        y_one_hot = init.one_hot(logits.shape[1], y, device=y.device)
        z_y = ops.summation(logits * y_one_hot, axes=1)
        return ops.summation(log_sum - z_y) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            Ex_orginal = ops.summation(x, axes=(0,)) / x.shape[0]
            Ex_broadcast = ops.broadcast_to(ops.reshape(Ex_orginal, (1, -1)), x.shape)
            Varx_orginal = (
                ops.summation(ops.power_scalar((x - Ex_broadcast), 2), axes=(0,))
                / x.shape[0]
            )
            Varx_broadcast = ops.broadcast_to(
                ops.reshape(
                    Varx_orginal,
                    (1, -1),
                ),
                x.shape,
            )
            y = ops.broadcast_to(self.weight, x.shape) * (x - Ex_broadcast) / (
                ops.power_scalar(Varx_broadcast + self.eps, 1 / 2)
            ) + ops.broadcast_to(self.bias, x.shape)
            self.running_mean = ops.reshape(
                (1 - self.momentum) * self.running_mean + self.momentum * Ex_orginal,
                (-1),
            )
            self.running_var = ops.reshape(
                (1 - self.momentum) * self.running_var + self.momentum * Varx_orginal,
                (-1),
            )
        else:
            running_mean_broadcast = ops.broadcast_to(self.running_mean, x.shape)
            running_var_broadcast = ops.broadcast_to(self.running_var, x.shape)
            y = ops.broadcast_to(self.weight, x.shape) * (
                x - running_mean_broadcast
            ) / (
                ops.power_scalar(running_var_broadcast + self.eps, 1 / 2)
            ) + ops.broadcast_to(
                self.bias, x.shape
            )
        return y
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Ex = (
            ops.broadcast_to(
                ops.reshape(ops.summation(x, axes=(1,)), (x.shape[0], 1)), x.shape
            )
            / x.shape[1]
        )
        Varx = (
            ops.broadcast_to(
                ops.reshape(
                    ops.summation(ops.power_scalar((x - Ex), 2), axes=(1,)),
                    (x.shape[0], 1),
                ),
                x.shape,
            )
            / x.shape[1]
        )
        y = ops.broadcast_to(self.weight, x.shape) * (x - Ex) / (
            ops.power_scalar(Varx + self.eps, 1 / 2)
        ) + ops.broadcast_to(self.bias, x.shape)

        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training == False:
            return x
        else:
            mask = init.randb(
                *x.shape, p=1 - self.p, device=(x.device), dtype=(x.dtype)
            ) / (1 - self.p)
            return x * mask
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
