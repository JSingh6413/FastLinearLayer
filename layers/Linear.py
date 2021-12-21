import torch
from torch import nn

from time import time
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using {} device'.format(device))


class LinearFunctionVanilla(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, k):
        ctx.save_for_backward(X, weight, k)
        return X @ weight.t()

    @staticmethod
    def backward(ctx, grad_output):
        
        X, weight, k = ctx.saved_tensors
        grad_input = grad_weight = X

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        
        if ctx.needs_input_grad[1]:

            # calculate distances
            prob = torch.linalg.norm(grad_output.reshape((grad_output.shape[0], -1)), axis=1) * \
                   torch.linalg.norm(X.reshape((X.shape[0], -1)), axis=1)
            
            prob = prob.flatten() / prob.sum()

            idx = prob.multinomial(num_samples=int(k), replacement=True)
            d = 1.0 / torch.sqrt(int(k) * prob[idx]) 

            A_sketched = (grad_output[idx].T * d).T
            B_sketched = (X[idx].T * d).T

            grad_weight = A_sketched.transpose(-1, -2) @ B_sketched


        return grad_input, grad_weight, None


class LinearFunctionNaive(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, k):

        prob = torch.linalg.norm(X.reshape((X.shape[0], -1)), axis=1)
        prob = prob.flatten() / prob.sum()

        idx = prob.multinomial(num_samples=int(k), replacement=True)
        d = 1.0 / torch.sqrt(int(k) * prob[idx])

        ctx.save_for_backward((X[idx].T * d).T, weight, idx, d)
        return X @ weight.t()

    @staticmethod
    def backward(ctx, grad_output):
        
        X, weight, idx, d = ctx.saved_tensors
        grad_input = grad_weight = X

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        
        if ctx.needs_input_grad[1]:
            A_sketched = (grad_output[idx].T * d).T
            grad_weight = A_sketched.transpose(-1, -2) @ X

        return grad_input, grad_weight, None


class LinearFunctionGauss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, k):

        seed = torch.tensor([hash(time)]).to(device)
        torch.manual_seed(seed)

        gauss = torch.randn((len(X), k)).to(device)
        gauss[gauss >= 0] = 1
        gauss[gauss < 0] = -1

        S = gauss.T @ X.reshape((X.shape[0], -1))
        S = S.reshape((k, *X.shape[1:]))

        ctx.save_for_backward(S / k.to(device), weight, seed, torch.tensor(gauss.shape))
        return X @ weight.t()

    @staticmethod
    def backward(ctx, grad_output):
        
        X, weight, seed, shape = ctx.saved_tensors
        grad_input = grad_weight = X

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        
        if ctx.needs_input_grad[1]:
            torch.manual_seed(seed)

            gauss = torch.randn(tuple(shape)).to(device)
            gauss[gauss >= 0] = 1
            gauss[gauss < 0] = -1

            S = gauss.T @ grad_output.reshape((grad_output.shape[0], -1))
            S = S.reshape((shape[1], *grad_output.shape[1:]))

            grad_weight = S.transpose(-1, -2) @ X

        return grad_input, grad_weight, None



class LinearFunctionTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, k=None):
        ctx.save_for_backward(X, weight)
        return X @ weight.t()

    @staticmethod
    def backward(ctx, grad_output):
        
        X, weight = ctx.saved_tensors
        grad_input = grad_weight = X

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-1, -2) @ X

        return grad_input, grad_weight, None



METHODS = {
    'vanilla': LinearFunctionVanilla,
    'naive': LinearFunctionNaive,
    'gauss': LinearFunctionGauss,
    'torch': LinearFunctionTorch,
}



class Linear(nn.Module):

    def __init__(self, input_shape, output_shape, k: int, method: str = 'vanilla'):
        super().__init__()

        if method not in METHODS:
            raise ValueError(f"Unsupported method {method}! Method must be one of {list(METHODS.keys())}.")

        self.function = METHODS[method]

        self.module = nn.Linear(input_shape, output_shape)
        self.k = torch.IntTensor([k])


    def forward(self, X):
        # return self.function.apply(X, self.module.weight, self.module.bias, self.k)
        return self.function.apply(X, self.module.weight, self.k) + self.module.bias
