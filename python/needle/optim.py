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
        # self.clip_grad_norm()
        for paramid, param in enumerate(self.params):
            grad = param.grad.detach() + self.weight_decay * param.detach()
            # print(grad)
            u = self.u.get(paramid, 0) * self.momentum + (1 - self.momentum) * grad
            u = ndl.Tensor(u, dtype=param.dtype)
            self.u[paramid] = u
            param.data -= self.lr * u

        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        # params = [p for p in self.params if p.grad.detach() is not None]
        # total_norm = 0
        # for p in params:
        #     total_norm = total_norm + ndl.ops.summation(
        #         ndl.ops.power_scalar(p.grad.detach(), 2)
        #     )
        # print(total_norm)
        # total_norm = ndl.ops.power_scalar(total_norm, 1 / 2).numpy()
        # print(total_norm)
        # clip = max_norm / (total_norm + 1e-6)
        # print("clip:", clip)
        # if clip < 1:
        #     for p in self.params:
        #         if p.grad.data is not None:
        #             # print(type(p.grad))
        #             print("before", p.grad.data)
        #             p.grad.cached_data *= clip
        #             print("ago", p.grad.detach())
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
            grad = param.grad.detach() + self.weight_decay * param.detach()
            grad = ndl.Tensor(grad, dtype=param.dtype)
            u = self.beta1 * self.m.get(id(param), 0) + (1 - self.beta1) * grad
            v = self.beta2 * self.v.get(id(param), 0) + (1 - self.beta2) * grad * grad
            self.m[id(param)] = u.detach()
            self.v[id(param)] = v.detach()
            u_hat = u / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            param.data -= self.lr * u_hat / (v_hat**0.5 + self.eps)

        ### END YOUR SOLUTION
