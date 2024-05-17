"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            g = param.grad.detach()
            if self.weight_decay != 0.0:
                g += self.weight_decay * param.detach()
            if self.momentum != 0.0:
                u = self.u.get(param) or ndl.init.zeros_like(g)
                u = self.momentum * u + (1 - self.momentum) * g
                g = u
            param.data -= self.lr * g
            self.u[param] = g
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            g = param.grad.detach()
            norm = (g ** 2).sum().detach()
            if norm > max_norm:
                param.grad = param.grad / (norm / max_norm) ** 0.5
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            g = param.grad
            if self.weight_decay != 0.0:
                g = (g + self.weight_decay * param).detach()
            m = self.m.get(param) or ndl.init.zeros_like(g)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.v.get(param) or ndl.init.zeros_like(g)
            v = self.beta2 * v + (1 - self.beta2) * (g ** 2)
            self.m[param] = m.detach()
            self.v[param] = v.detach()

            # bias correction
            m = m / (1 - self.beta1 ** self.t)
            v = v / (1 - self.beta2 ** self.t)

            g = m / (v ** 0.5 + self.eps)
            param.data -= self.lr * g
        ### END YOUR SOLUTION
