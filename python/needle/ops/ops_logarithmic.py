from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Zmax = array_api.max(Z, axis=1, keepdims=True)
        tmpZ = Z - array_api.broadcast_to(Zmax, Z.shape)
        logsumexpZ = array_api.log(
            array_api.sum(array_api.exp(Z - Zmax), axis=1, keepdims=True)
        )
        return tmpZ - logsumexpZ
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].realize_cached_data()
        y = array_api.exp(Z)
        grad = array_api.sum(out_grad.realize_cached_data(), axis=(1,)).reshape(
            (Z.shape[0], 1)
        )
        sum_y = array_api.sum(y, axis=(1,)).reshape((Z.shape[0], 1))

        grad = array_api.broadcast_to(grad / sum_y, Z.shape)
        grad = grad * y
        return out_grad - grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Zmax = array_api.max(Z, axis=self.axes, keepdims=True)
        Z = Z - array_api.broadcast_to(Zmax, Z.shape)
        res = array_api.log(array_api.sum(array_api.exp(Z), axis=self.axes))
        return res + array_api.reshape(Zmax, res.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].realize_cached_data()
        Zmax = array_api.max(Z, axis=self.axes, keepdims=True)
        Z = Z - array_api.broadcast_to(Zmax, Z.shape)
        sum_Z = array_api.sum(array_api.exp(Z), axis=self.axes, keepdims=True)
        res_Z = array_api.exp(Z) / array_api.broadcast_to(sum_Z, Z.shape)

        if res_Z.shape != out_grad.shape:
            new_shape = list(res_Z.shape)
            if self.axes is not None:
                if isinstance(self.axes, Number):
                    self.axes = (self.axes,)
                for axis in self.axes:
                    new_shape[axis] = 1
            else:
                new_shape = [1] * len(new_shape)
            out_grad = reshape(out_grad, new_shape)
            out_grad = broadcast_to(out_grad, res_Z.shape)
        return out_grad * res_Z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
