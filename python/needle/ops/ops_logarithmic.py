from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = Z.max(axis=self.axes)
        shape = tuple(
            1
            if self.axes is None or i in self.axes or i - len(Z.shape) in self.axes
            else s
            for i, s in enumerate(Z.shape)
        )
        Z_max_broadcast = array_api.broadcast_to(Z_max.reshape(shape), Z.shape)
        Z_exp = array_api.exp(Z - Z_max_broadcast)
        return array_api.log(array_api.sum(Z_exp, axis=self.axes)) + Z_max
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Manually deriving this is so hard
        Z = node.inputs[0].realize_cached_data()
        Z_max = Z.max(axis=self.axes)
        shape = tuple(
            1
            if self.axes is None or i in self.axes or i - len(Z.shape) in self.axes
            else s
            for i, s in enumerate(Z.shape)
        )
        Z_max_broadcast = array_api.broadcast_to(Z_max.reshape(shape), Z.shape)
        Z_minus = Z - Z_max_broadcast
        # summing
        Z_exp = array_api.exp(Z_minus)
        Z_sum = array_api.sum(Z_exp, axis=self.axes)
        Z_sum_broadcast = array_api.broadcast_to(Z_sum.reshape(shape), Z.shape)
        G = Z_exp / Z_sum_broadcast
        return out_grad.reshape(shape).broadcast_to(Z.shape) * Tensor(G, device=out_grad.device, dtype=out_grad.dtype)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
